"""Serving-side wrapper around ``cxr_tb.src.pipeline.TBPipeline``.

Responsibilities that belong to the web layer, not to the model package:
  - lazily build the pipeline from settings (TensorFlow is loaded once, at startup),
  - persist overlay / mask PNGs under ``static/results`` and turn them into public URLs,
  - expose a ``FakeInferenceService`` so tests and UI work do not need model weights.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class AnalysisArtifacts:
    """What ``/analyze`` stores and returns for one image."""

    case: dict[str, Any]  # serializable case JSON (segmentation, radiomics, ml_results)
    overlay_path: str | None  # relative to static dir, e.g. "results/<id>_overlay.png"
    mask_path: str | None

    @property
    def ml_results(self) -> dict[str, Any]:
        return self.case.get("ml_results", {})


class InferenceService(Protocol):
    name: str

    def analyze(self, image_path: Path, image_id: str) -> AnalysisArtifacts: ...

    def analyze_ct(self, volume_path: Path, mask_path: Path, image_id: str) -> AnalysisArtifacts: ...

    def model_metadata(self) -> dict[str, Any]: ...


def analyze_ct_files(settings: Settings, volume_path: Path, mask_path: Path, image_id: str) -> AnalysisArtifacts:
    """CT path shared by both backends: no model is involved, only QC + 3D radiomics on the given mask."""
    from cxr_tb.src.ct import CTPipeline

    result = CTPipeline().analyze(volume_path, mask_path, image_id=image_id)
    overlay_rel = f"results/{image_id}_overlay.png"
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    result.overlay_image().save(settings.static_dir / overlay_rel)
    return AnalysisArtifacts(case=result.to_case_json(), overlay_path=overlay_rel, mask_path=None)


class RealInferenceService:
    name = "real"

    def __init__(self, settings: Settings):
        from cxr_tb.src.pipeline import TBPipeline  # heavy import kept local

        if not settings.segmentation_model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model not found at {settings.segmentation_model_path}. "
                "Download it from the GitHub release (scripts/download_models.sh) or train it."
            )
        self.settings = settings
        self.pipeline = TBPipeline.from_paths(settings.segmentation_model_path, settings.classifier_path)
        if self.pipeline.classifier is None:
            logger.warning(
                "No classifier bundle at %s; lesions will be reported unclassified.", settings.classifier_path
            )
        settings.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, image_path: Path, image_id: str) -> AnalysisArtifacts:
        result = self.pipeline.analyze(image_path, image_id=image_id)
        overlay_rel = f"results/{image_id}_overlay.png"
        mask_rel = f"results/{image_id}_mask.png"
        result.overlay_image().save(self.settings.static_dir / overlay_rel)
        result.mask_image().save(self.settings.static_dir / mask_rel)
        return AnalysisArtifacts(case=result.to_case_json(), overlay_path=overlay_rel, mask_path=mask_rel)

    def analyze_ct(self, volume_path: Path, mask_path: Path, image_id: str) -> AnalysisArtifacts:
        return analyze_ct_files(self.settings, volume_path, mask_path, image_id)

    def model_metadata(self) -> dict[str, Any]:
        classifier = self.pipeline.classifier
        return {
            "segmentation_model": str(self.settings.segmentation_model_path.name),
            "ct": "mask ingestion only (no CT segmentation model); 3D radiomics of the tumor label",
            "classifier": None
            if classifier is None
            else {
                "model_type": classifier.model_type,
                "threshold": classifier.threshold,
                "n_features": len(classifier.feature_names),
                "dataset": classifier.metadata.get("dataset"),
                "cross_validation": classifier.metadata.get("cross_validation", {}).get(classifier.model_type),
                "feature_importance_top10": classifier.metadata.get("feature_importance_top10"),
                "group_importance": classifier.metadata.get("group_importance"),
                "limitations": classifier.metadata.get("limitations"),
            },
        }


class FakeInferenceService:
    """Deterministic stand-in: a synthetic blob mask and fixed features. No TensorFlow needed."""

    name = "fake"

    def __init__(self, settings: Settings):
        self.settings = settings
        settings.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, image_path: Path, image_id: str) -> AnalysisArtifacts:
        from cxr_tb.src.pipeline import load_grayscale, render_overlay, resize_to_model, summarize_lesions

        resized = resize_to_model(load_grayscale(image_path))
        yy, xx = np.mgrid[0:512, 0:512]
        mask = ((xx - 150) ** 2 + (yy - 140) ** 2) < 45**2
        probability = np.where(mask, 0.9, 0.05).astype(np.float32)
        lesion = summarize_lesions(mask, probability)
        features = {
            "original_glcm_Imc1": -0.41,
            "original_firstorder_Maximum": 231.0,
            "original_firstorder_Range": 198.0,
        }
        case = {
            "modality": "xray",
            "image": {"id": image_id, "width": 512, "height": 512, "model_resolution": [512, 512]},
            "qc": {"passed": True, "usable_for_patient": True, "flags": [], "flag_descriptions": [], "details": {}},
            "segmentation": lesion.to_dict(),
            "radiomics_features": features,
            "ml_results": {
                "primary_label": "tb_suspected",
                "tb_probability": 0.81,
                "risk_level": "high",
                "classifier": "fake",
                "threshold": 0.5,
                "top_features": [{"feature": k, "importance": 0.1, "value": v} for k, v in features.items()],
                "decision_basis": "FAKE inference backend (INFERENCE_BACKEND=fake); values are fixed.",
            },
            "timings_ms": {"segmentation": 0.0},
        }
        overlay_rel = f"results/{image_id}_overlay.png"
        mask_rel = f"results/{image_id}_mask.png"
        render_overlay(resized, mask).save(self.settings.static_dir / overlay_rel)
        from cxr_tb.src.pipeline import mask_to_image

        mask_to_image(mask).save(self.settings.static_dir / mask_rel)
        return AnalysisArtifacts(case=case, overlay_path=overlay_rel, mask_path=mask_rel)

    def analyze_ct(self, volume_path: Path, mask_path: Path, image_id: str) -> AnalysisArtifacts:
        return analyze_ct_files(self.settings, volume_path, mask_path, image_id)

    def model_metadata(self) -> dict[str, Any]:
        return {"segmentation_model": "fake", "classifier": {"model_type": "fake", "threshold": 0.5}}


def build_inference_service(settings: Settings) -> InferenceService:
    if settings.inference_backend == "fake":
        logger.warning("Using FAKE inference backend; results are not real predictions.")
        return FakeInferenceService(settings)
    return RealInferenceService(settings)


def new_image_id() -> str:
    return uuid.uuid4().hex
