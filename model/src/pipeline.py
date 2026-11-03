"""End-to-end inference pipeline: U-Net segmentation -> lesion statistics -> radiomics -> TB classifier.

This module is the single place where images are prepared and features are computed, so the
serving path (WebBack) and the training / evaluation scripts are guaranteed to be identical.

Heavy dependencies (TensorFlow, PyRadiomics, SimpleITK) are imported lazily inside the classes
that need them, so the pure helpers (lesion statistics, zones, risk levels) stay cheap to import
and easy to unit test.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from cxr_tb.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MASK_THRESHOLD,
    MIN_COMPONENT_PX,
    MIN_LESION_AREA_FRACTION,
    QC_ASPECT_RANGE,
    QC_HUGE_ROI_FRACTION,
    QC_LOW_CONFIDENCE,
    QC_MIN_CONTRAST_STD,
    QC_MIN_IMAGE_SIDE,
    QC_TINY_ROI_FRACTION,
    RISK_HIGH_MIN,
    RISK_LOW_MAX,
)
from PIL import Image
from scipy import ndimage

ImageSource = str | Path | bytes | Image.Image | np.ndarray

LABEL_NO_LESION = "no_lesion_detected"
LABEL_TB_SUSPECTED = "tb_suspected"
LABEL_LOW_TB = "low_tb_likelihood"
LABEL_LESION_UNCLASSIFIED = "lesion_detected_unclassified"


# --------------------------------------------------------------------------- image preparation


def load_grayscale(source: ImageSource) -> np.ndarray:
    """Return an 8-bit grayscale array (H, W) from a path, raw bytes, PIL image, or array."""
    if isinstance(source, np.ndarray):
        array = source
        if array.ndim == 3:
            array = np.asarray(Image.fromarray(array.astype(np.uint8)).convert("L"))
        return np.clip(array, 0, 255).astype(np.uint8)
    if isinstance(source, (str, Path)):
        image = Image.open(source)
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    elif isinstance(source, Image.Image):
        image = source
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported image source: {type(source)!r}")
    return np.asarray(image.convert("L"), dtype=np.uint8)


def resize_to_model(gray: np.ndarray, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH) -> np.ndarray:
    """Bilinear resize to the model resolution without antialiasing; returns uint8 (H, W).

    This reproduces ``tf.image.resize(..., method="bilinear")`` (half-pixel centres, no antialias),
    which is what the U-Net saw during training. PIL/OpenCV antialias when downscaling, which
    smooths texture and measurably changes the predicted masks, so a NumPy re-implementation is
    used to keep serving identical to training without importing TensorFlow here.
    """
    in_h, in_w = gray.shape
    if (in_h, in_w) == (height, width):
        return gray.astype(np.uint8)
    ys = np.clip((np.arange(height) + 0.5) * (in_h / height) - 0.5, 0, in_h - 1)
    xs = np.clip((np.arange(width) + 0.5) * (in_w / width) - 0.5, 0, in_w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    src = gray.astype(np.float32)
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    out = top * (1 - wy) + bottom * wy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def to_model_input(resized_u8: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] float32 with a batch and channel axis: (1, H, W, 1)."""
    return (resized_u8.astype(np.float32) / 255.0)[None, ..., None]


# --------------------------------------------------------------------------- lesion statistics


def postprocess_mask(mask: np.ndarray, min_component_px: int = MIN_COMPONENT_PX) -> np.ndarray:
    """Remove connected components smaller than ``min_component_px`` pixels."""
    mask = mask.astype(bool)
    if min_component_px <= 1 or not mask.any():
        return mask
    labeled, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:] = sizes >= min_component_px
    return keep[labeled]


def zone_of(cx_norm: float, cy_norm: float) -> str:
    """Map a normalized centroid to a radiographic lung zone.

    Radiographic convention: the patient's right lung appears on the left side of the image.
    """
    side = "right" if cx_norm < 0.5 else "left"
    if cy_norm < 1 / 3:
        level = "upper"
    elif cy_norm < 2 / 3:
        level = "middle"
    else:
        level = "lower"
    return f"{side} {level} zone"


@dataclass
class LesionComponent:
    area_px: int
    area_fraction: float
    bbox: list[int]  # [x0, y0, x1, y1] in model-resolution pixels
    centroid: list[float]  # [x, y] normalized to [0, 1]
    zone: str
    max_probability: float | None = None


@dataclass
class LesionSummary:
    detected: bool
    mask_area_fraction: float
    num_components: int
    zones: list[str]
    components: list[LesionComponent]
    max_probability: float
    model_resolution: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_lesions(
    mask: np.ndarray,
    probability_map: np.ndarray | None = None,
    min_area_fraction: float = MIN_LESION_AREA_FRACTION,
    max_components_reported: int = 10,
) -> LesionSummary:
    """Describe a binary mask: coverage, connected components, zones, and confidence."""
    mask = mask.astype(bool)
    height, width = mask.shape
    total = float(height * width)
    area_fraction = float(mask.sum() / total)
    max_prob = float(probability_map.max()) if probability_map is not None and probability_map.size else 0.0

    components: list[LesionComponent] = []
    if mask.any():
        labeled, count = ndimage.label(mask)
        for idx in range(1, count + 1):
            region = labeled == idx
            ys, xs = np.nonzero(region)
            area = int(region.sum())
            cx, cy = float(xs.mean()) / width, float(ys.mean()) / height
            comp_prob = float(probability_map[region].max()) if probability_map is not None else None
            components.append(
                LesionComponent(
                    area_px=area,
                    area_fraction=area / total,
                    bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    centroid=[round(cx, 4), round(cy, 4)],
                    zone=zone_of(cx, cy),
                    max_probability=comp_prob,
                )
            )
        components.sort(key=lambda c: c.area_px, reverse=True)

    zones: list[str] = []
    for comp in components:
        if comp.zone not in zones:
            zones.append(comp.zone)

    return LesionSummary(
        detected=area_fraction >= min_area_fraction and bool(components),
        mask_area_fraction=round(area_fraction, 6),
        num_components=len(components),
        zones=zones,
        components=components[:max_components_reported],
        max_probability=round(max_prob, 4),
        model_resolution=[height, width],
    )


def risk_level(tb_probability: float | None) -> str:
    if tb_probability is None:
        return "low"
    if tb_probability < RISK_LOW_MAX:
        return "low"
    if tb_probability < RISK_HIGH_MIN:
        return "moderate"
    return "high"


# --------------------------------------------------------------------------- quality control

QC_BLOCKING_FLAGS = {"image_too_small", "low_contrast", "unusual_aspect", "huge_roi", "nan_features"}
QC_FLAG_DESCRIPTIONS = {
    "image_too_small": "input image is smaller than the model expects",
    "low_contrast": "image has almost no contrast (blank or washed out)",
    "unusual_aspect": "aspect ratio does not look like a frontal chest radiograph",
    "tiny_roi": "segmented region is very small; texture features are unreliable",
    "huge_roi": "segmented region is implausibly large; likely a segmentation failure",
    "low_confidence": "segmentation confidence is low",
    "nan_features": "some radiomics features could not be computed",
}


@dataclass
class QCResult:
    """Quality-control verdict for one analysis.

    ``passed`` means the numbers can be interpreted; ``usable_for_patient`` additionally requires enough
    confidence to show a patient-facing explanation. Either way the flags are surfaced to the LLM and the UI.
    """

    passed: bool
    usable_for_patient: bool
    flags: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "usable_for_patient": self.usable_for_patient,
            "flags": self.flags,
            "flag_descriptions": [QC_FLAG_DESCRIPTIONS.get(f, f) for f in self.flags],
            "details": self.details,
        }


def run_qc(
    gray: np.ndarray,
    lesion: LesionSummary,
    features: dict[str, float] | None,
) -> QCResult:
    """Check the input image, the segmented region, and the feature vector for conditions that make the
    analysis unreliable. Pure function; thresholds live in config."""
    flags: list[str] = []
    height, width = gray.shape[:2]
    contrast = float(np.std(gray))
    aspect = width / max(height, 1)
    if min(height, width) < QC_MIN_IMAGE_SIDE:
        flags.append("image_too_small")
    if contrast < QC_MIN_CONTRAST_STD:
        flags.append("low_contrast")
    if not (QC_ASPECT_RANGE[0] <= aspect <= QC_ASPECT_RANGE[1]):
        flags.append("unusual_aspect")
    if lesion.detected:
        if lesion.mask_area_fraction < QC_TINY_ROI_FRACTION:
            flags.append("tiny_roi")
        if lesion.mask_area_fraction > QC_HUGE_ROI_FRACTION:
            flags.append("huge_roi")
        if lesion.max_probability < QC_LOW_CONFIDENCE:
            flags.append("low_confidence")
    if features:
        bad = [k for k, v in features.items() if v is None or not np.isfinite(v)]
        if bad:
            flags.append("nan_features")
    passed = not any(f in QC_BLOCKING_FLAGS for f in flags)
    usable_for_patient = passed and not any(f in {"tiny_roi", "low_confidence"} for f in flags)
    return QCResult(
        passed=passed,
        usable_for_patient=usable_for_patient,
        flags=flags,
        details={
            "image_size": [int(width), int(height)],
            "contrast_std": round(contrast, 2),
            "aspect_ratio": round(aspect, 3),
            "mask_area_fraction": lesion.mask_area_fraction,
            "max_probability": lesion.max_probability,
            "n_features": len(features or {}),
        },
    )


# --------------------------------------------------------------------------- visualisation


def render_overlay(
    gray_u8: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int] = (220, 38, 38),
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a coloured mask over a grayscale image (both at the same resolution)."""
    base = Image.fromarray(gray_u8).convert("RGB")
    if mask.shape != gray_u8.shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(base.size, Image.NEAREST)) > 127
    tint = Image.new("RGB", base.size, tuple(int(c) for c in color))
    weight = Image.fromarray(mask.astype(np.uint8) * int(round(255 * alpha)))
    return Image.composite(tint, base, weight)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255)


# --------------------------------------------------------------------------- model wrappers


class SegmentationModel:
    """Thin wrapper around the trained Keras U-Net."""

    def __init__(self, model_path: str | Path, threshold: float = MASK_THRESHOLD):
        from cxr_tb.src.model_loader import load_model_compat  # lazy: TensorFlow is heavy

        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model = load_model_compat(self.model_path)

    def predict_proba(self, resized_u8: np.ndarray) -> np.ndarray:
        """Per-pixel lesion probability at model resolution, shape (H, W)."""
        return self.model.predict(to_model_input(resized_u8), verbose=0)[0, ..., 0]

    def predict_mask(self, resized_u8: np.ndarray, min_component_px: int = MIN_COMPONENT_PX) -> np.ndarray:
        return postprocess_mask(self.predict_proba(resized_u8) >= self.threshold, min_component_px)


class RadiomicsExtractor:
    """PyRadiomics feature extraction on in-memory arrays (no temporary files)."""

    def __init__(self, bin_width: float = 25.0, normalize: bool = False):
        import logging

        from cxr_tb.src.radiomics_extract import build_extractor  # lazy: pyradiomics is heavy

        logging.getLogger("radiomics").setLevel(logging.ERROR)  # silence per-call GLCM notices
        self.extractor = build_extractor(bin_width=bin_width, normalize=normalize)

    def extract(self, gray_u8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        import SimpleITK as sitk

        image = sitk.GetImageFromArray(gray_u8.astype(np.float32))
        label = sitk.GetImageFromArray(mask.astype(np.uint8))
        result = self.extractor.execute(image, label, label=1)
        features: dict[str, float] = {}
        for key, value in result.items():
            if key.startswith("diagnostics_"):
                continue
            array = np.asarray(value)
            if array.size != 1:
                continue
            try:
                features[key] = float(array.item())
            except (TypeError, ValueError):
                continue
        return features


@dataclass
class ClassifierBundle:
    """A fitted scikit-learn estimator plus everything needed to apply it consistently."""

    estimator: Any
    feature_names: list[str]
    threshold: float = 0.5
    model_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> ClassifierBundle:
        payload = joblib.load(path)
        return cls(
            estimator=payload["estimator"],
            feature_names=list(payload["feature_names"]),
            threshold=float(payload.get("threshold", 0.5)),
            model_type=str(payload.get("model_type", "unknown")),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "feature_names": self.feature_names,
                "threshold": self.threshold,
                "model_type": self.model_type,
                "metadata": self.metadata,
            },
            path,
            compress=3,
        )

    def vectorize(self, features: dict[str, float]) -> pd.DataFrame:
        # A DataFrame keeps feature names attached, exactly as during training.
        return pd.DataFrame([[features.get(name, np.nan) for name in self.feature_names]], columns=self.feature_names)

    def predict_proba(self, features: dict[str, float]) -> float:
        return float(self.estimator.predict_proba(self.vectorize(features))[0, 1])

    def top_feature_contributions(self, features: dict[str, float], top_k: int = 8) -> list[dict[str, Any]]:
        """Feature importances (RF) or standardized coefficients (logistic) for the given case."""
        importance = self.metadata.get("feature_importance")
        if not importance:
            return []
        ranked = sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        return [
            {"feature": name, "importance": round(float(weight), 6), "value": features.get(name)}
            for name, weight in ranked
        ]


# --------------------------------------------------------------------------- pipeline


@dataclass
class AnalysisResult:
    image_id: str
    image_size: list[int]  # [width, height] of the uploaded image
    lesion: LesionSummary
    radiomics_features: dict[str, float] | None
    ml_results: dict[str, Any]
    timings_ms: dict[str, float]
    mask: np.ndarray = field(repr=False)
    probability_map: np.ndarray = field(repr=False)
    resized_u8: np.ndarray = field(repr=False)
    qc: QCResult | None = None

    def to_case_json(self) -> dict[str, Any]:
        """Serializable description of the case for storage and for the LLM report."""
        return {
            "modality": "xray",
            "image": {
                "id": self.image_id,
                "width": self.image_size[0],
                "height": self.image_size[1],
                "model_resolution": self.lesion.model_resolution,
            },
            "qc": self.qc.to_dict() if self.qc else None,
            "segmentation": self.lesion.to_dict(),
            "radiomics_features": self.radiomics_features,
            "ml_results": self.ml_results,
            "timings_ms": self.timings_ms,
        }

    def overlay_image(self) -> Image.Image:
        return render_overlay(self.resized_u8, self.mask)

    def mask_image(self) -> Image.Image:
        return mask_to_image(self.mask)


class TBPipeline:
    """Segmentation -> lesion summary -> radiomics -> classifier, with a 'no lesion' gate."""

    def __init__(
        self,
        segmentation: SegmentationModel,
        extractor: RadiomicsExtractor | None,
        classifier: ClassifierBundle | None,
        min_area_fraction: float = MIN_LESION_AREA_FRACTION,
        min_component_px: int = MIN_COMPONENT_PX,
    ):
        self.segmentation = segmentation
        self.extractor = extractor
        self.classifier = classifier
        self.min_area_fraction = min_area_fraction
        self.min_component_px = min_component_px

    @classmethod
    def from_paths(
        cls,
        segmentation_model_path: str | Path,
        classifier_path: str | Path | None = None,
        with_radiomics: bool = True,
        **kwargs: Any,
    ) -> TBPipeline:
        classifier = None
        if classifier_path and Path(classifier_path).exists():
            classifier = ClassifierBundle.load(classifier_path)
        extractor = RadiomicsExtractor() if with_radiomics else None
        return cls(SegmentationModel(segmentation_model_path), extractor, classifier, **kwargs)

    def analyze(self, source: ImageSource, image_id: str = "image") -> AnalysisResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        gray = load_grayscale(source)
        resized = resize_to_model(gray)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        probability_map = self.segmentation.predict_proba(resized)
        mask = postprocess_mask(probability_map >= self.segmentation.threshold, self.min_component_px)
        timings["segmentation"] = (time.perf_counter() - t0) * 1000

        lesion = summarize_lesions(mask, probability_map, self.min_area_fraction)

        features: dict[str, float] | None = None
        ml: dict[str, Any] = {
            "primary_label": LABEL_NO_LESION,
            "tb_probability": None,
            "risk_level": "low",
            "classifier": None,
            "decision_basis": (
                "No lesion region above the minimum area threshold was segmented; "
                "radiomics and the TB classifier were not run."
            ),
        }

        if lesion.detected and self.extractor is not None:
            t0 = time.perf_counter()
            features = self.extractor.extract(resized, mask)
            timings["radiomics"] = (time.perf_counter() - t0) * 1000

            if self.classifier is not None:
                t0 = time.perf_counter()
                prob = self.classifier.predict_proba(features)
                timings["classifier"] = (time.perf_counter() - t0) * 1000
                label = LABEL_TB_SUSPECTED if prob >= self.classifier.threshold else LABEL_LOW_TB
                ml = {
                    "primary_label": label,
                    "tb_probability": round(prob, 4),
                    "risk_level": risk_level(prob),
                    "classifier": self.classifier.model_type,
                    "threshold": self.classifier.threshold,
                    "top_features": self.classifier.top_feature_contributions(features),
                    "decision_basis": (
                        "Radiomics features of the segmented region scored by the TB classifier; "
                        "probability is relative to the training distribution (see model metadata)."
                    ),
                }
            else:
                ml = {
                    "primary_label": LABEL_LESION_UNCLASSIFIED,
                    "tb_probability": None,
                    "risk_level": "moderate",
                    "classifier": None,
                    "decision_basis": "A lesion region was segmented but no classifier is loaded.",
                }
        elif lesion.detected:
            ml = {
                "primary_label": LABEL_LESION_UNCLASSIFIED,
                "tb_probability": None,
                "risk_level": "moderate",
                "classifier": None,
                "decision_basis": "A lesion region was segmented; radiomics extraction is disabled.",
            }

        qc = run_qc(gray, lesion, features)
        if not qc.passed:
            ml["qc_warning"] = "quality control failed; interpret with caution: " + ", ".join(qc.flags)
        elif qc.flags:
            ml["qc_warning"] = "quality control notes: " + ", ".join(qc.flags)

        timings = {k: round(v, 1) for k, v in timings.items()}
        return AnalysisResult(
            image_id=image_id,
            image_size=[int(gray.shape[1]), int(gray.shape[0])],
            lesion=lesion,
            radiomics_features=features,
            ml_results=ml,
            timings_ms=timings,
            mask=mask,
            probability_map=probability_map,
            resized_u8=resized,
            qc=qc,
        )


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
