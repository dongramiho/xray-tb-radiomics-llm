"""CT path: ingest an externally produced segmentation mask with its volume and characterise the lesion.

There is no CT segmentation model in this repository (the presentation's nnU-Net was never shipped), so this
module takes a NIfTI volume plus a NIfTI label map (0 background, 1 liver, 2 tumor by convention) — the
output format of nnU-Net — and runs the same QC -> radiomics -> structured case JSON flow as the X-ray path,
in 3D. A missing tumor label is reported as "no tumor label" and never as "no tumor".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LABEL_BACKGROUND, LABEL_LIVER, LABEL_TUMOR = 0, 1, 2
MIN_TUMOR_VOXELS = 50
MAX_SLICE_SPACING_MM = 5.0
GEOMETRY_TOLERANCE = 1e-3
LIVER_WINDOW_HU = (-160.0, 240.0)

CT_QC_DESCRIPTIONS = {
    "geometry_mismatch": "mask and volume differ in size or spacing",
    "tumor_label_missing": "no tumor label (2) in the mask — this is NOT evidence of no tumor",
    "liver_label_missing": "no liver label (1) in the mask",
    "tiny_roi": "tumor region has very few voxels; texture features are unreliable",
    "coarse_slices": "slice spacing is large; 3D texture features are less reliable",
    "nan_features": "some radiomics features could not be computed",
    "not_3d": "volume is not three-dimensional",
}
CT_QC_BLOCKING = {"geometry_mismatch", "not_3d", "nan_features"}


@dataclass
class CTQCResult:
    passed: bool
    usable_for_patient: bool
    flags: list[str]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "usable_for_patient": self.usable_for_patient,
            "flags": self.flags,
            "flag_descriptions": [CT_QC_DESCRIPTIONS.get(f, f) for f in self.flags],
            "details": self.details,
        }


def load_image(path: str | Path):
    import SimpleITK as sitk

    return sitk.ReadImage(str(path))


def _geometry(image) -> dict[str, Any]:
    return {
        "size": [int(v) for v in image.GetSize()],
        "spacing_mm": [round(float(v), 4) for v in image.GetSpacing()],
        "origin": [round(float(v), 3) for v in image.GetOrigin()],
    }


def qc_ct(volume, mask, tumor_voxels: int, liver_voxels: int, features: dict[str, float] | None) -> CTQCResult:
    flags: list[str] = []
    if volume.GetDimension() != 3 or mask.GetDimension() != 3:
        flags.append("not_3d")
    else:
        if tuple(volume.GetSize()) != tuple(mask.GetSize()) or any(
            abs(a - b) > GEOMETRY_TOLERANCE for a, b in zip(volume.GetSpacing(), mask.GetSpacing(), strict=True)
        ):
            flags.append("geometry_mismatch")
        if max(volume.GetSpacing()) > MAX_SLICE_SPACING_MM:
            flags.append("coarse_slices")
    if tumor_voxels == 0:
        flags.append("tumor_label_missing")
    elif tumor_voxels < MIN_TUMOR_VOXELS:
        flags.append("tiny_roi")
    if liver_voxels == 0:
        flags.append("liver_label_missing")
    if features:
        if any(v is None or not np.isfinite(v) for v in features.values()):
            flags.append("nan_features")
    passed = not any(f in CT_QC_BLOCKING for f in flags)
    usable = passed and "tiny_roi" not in flags and "tumor_label_missing" not in flags
    return CTQCResult(
        passed=passed,
        usable_for_patient=usable,
        flags=flags,
        details={
            "volume": _geometry(volume),
            "mask": _geometry(mask),
            "tumor_voxels": tumor_voxels,
            "liver_voxels": liver_voxels,
        },
    )


def extract_ct_features(volume, mask, label: int = LABEL_TUMOR, bin_width: float = 25.0) -> dict[str, float]:
    """3D PyRadiomics features (shape, first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM) for one label."""
    import logging

    import SimpleITK as sitk
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=bin_width, resampledPixelSpacing=None)
    extractor.disableAllFeatures()
    for cls in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        extractor.enableFeatureClassByName(cls)
    result = extractor.execute(sitk.Cast(volume, sitk.sitkFloat32), sitk.Cast(mask, sitk.sitkUInt8), label=label)
    features: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        array = np.asarray(value)
        if array.size == 1:
            try:
                features[key] = float(array.item())
            except (TypeError, ValueError):
                continue
    return features


def lesion_summary_3d(
    volume, mask_array: np.ndarray, spacing: tuple[float, float, float], label: int
) -> dict[str, Any]:
    """Volume, extent and intensity statistics of one label (array axes are z, y, x)."""
    region = mask_array == label
    voxels = int(region.sum())
    voxel_ml = float(np.prod(spacing)) / 1000.0
    out: dict[str, Any] = {"label": label, "voxels": voxels, "volume_ml": round(voxels * voxel_ml, 2)}
    if voxels == 0:
        return out
    zs, ys, xs = np.nonzero(region)
    extent_mm = [
        round(float((axis.max() - axis.min() + 1) * sp), 1)
        for axis, sp in ((xs, spacing[0]), (ys, spacing[1]), (zs, spacing[2]))
    ]
    import SimpleITK as sitk

    intensities = sitk.GetArrayFromImage(volume)[region].astype(np.float64)
    out.update(
        {
            "extent_mm": extent_mm,  # x, y, z
            "max_extent_mm": max(extent_mm),
            "slices_with_label": int(len(np.unique(zs))),
            "centroid_voxel": [round(float(xs.mean()), 1), round(float(ys.mean()), 1), round(float(zs.mean()), 1)],
            "mean_hu": round(float(intensities.mean()), 1),
            "std_hu": round(float(intensities.std()), 1),
            "min_hu": round(float(intensities.min()), 1),
            "max_hu": round(float(intensities.max()), 1),
        }
    )
    return out


def window_to_u8(slice_hu: np.ndarray, window: tuple[float, float] = LIVER_WINDOW_HU) -> np.ndarray:
    lo, hi = window
    scaled = np.clip((slice_hu.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def render_ct_overlay(
    volume_array: np.ndarray, mask_array: np.ndarray, label: int = LABEL_TUMOR
) -> tuple[Image.Image, int]:
    """Axial slice with the largest label area (or the middle slice), mask tinted. Returns (image, slice index)."""
    from cxr_tb.src.pipeline import render_overlay

    region = mask_array == label
    per_slice = region.reshape(region.shape[0], -1).sum(axis=1)
    index = int(np.argmax(per_slice)) if per_slice.max() > 0 else volume_array.shape[0] // 2
    base = window_to_u8(volume_array[index])
    overlay = render_overlay(base, region[index])
    liver = mask_array[index] == LABEL_LIVER
    if liver.any():
        overlay = render_overlay(np.asarray(overlay.convert("L")), liver, color=(59, 130, 246), alpha=0.18)
        overlay = render_overlay(np.asarray(overlay.convert("L")), region[index])
    return overlay, index


@dataclass
class CTAnalysisResult:
    image_id: str
    qc: CTQCResult
    tumor: dict[str, Any]
    liver: dict[str, Any]
    radiomics_features: dict[str, float] | None
    ml_results: dict[str, Any]
    timings_ms: dict[str, float]
    overlay: Image.Image = field(repr=False)
    overlay_slice: int = 0
    geometry: dict[str, Any] = field(default_factory=dict)

    def to_case_json(self) -> dict[str, Any]:
        return {
            "modality": "ct",
            "image": {"id": self.image_id, **self.geometry, "overlay_slice": self.overlay_slice},
            "qc": self.qc.to_dict(),
            "segmentation": {
                "detected": self.tumor["voxels"] > 0,
                "labels_present": [
                    lbl for lbl, info in (("liver", self.liver), ("tumor", self.tumor)) if info["voxels"] > 0
                ],
                "tumor": self.tumor,
                "liver": self.liver,
                "zones": [],
                "num_components": 1 if self.tumor["voxels"] > 0 else 0,
                "mask_area_fraction": None,
            },
            "radiomics_features": self.radiomics_features,
            "ml_results": self.ml_results,
            "timings_ms": self.timings_ms,
        }

    def overlay_image(self) -> Image.Image:
        return self.overlay


class CTPipeline:
    """QC -> 3D radiomics of the tumor label -> case JSON. No classifier: characterisation only."""

    def analyze(self, volume_path: str | Path, mask_path: str | Path, image_id: str = "ct") -> CTAnalysisResult:
        import SimpleITK as sitk

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        volume = load_image(volume_path)
        mask = load_image(mask_path)
        volume_array = sitk.GetArrayFromImage(volume)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.int32)
        spacing = tuple(float(v) for v in volume.GetSpacing())
        timings["load"] = (time.perf_counter() - t0) * 1000

        tumor = lesion_summary_3d(volume, mask_array, spacing, LABEL_TUMOR)
        liver = lesion_summary_3d(volume, mask_array, spacing, LABEL_LIVER)
        features: dict[str, float] | None = None
        geometry_ok = tuple(volume.GetSize()) == tuple(mask.GetSize())
        if tumor["voxels"] >= MIN_TUMOR_VOXELS and geometry_ok:
            t0 = time.perf_counter()
            features = extract_ct_features(volume, mask, LABEL_TUMOR)
            timings["radiomics"] = (time.perf_counter() - t0) * 1000
        qc = qc_ct(volume, mask, tumor["voxels"], liver["voxels"], features)

        if tumor["voxels"] == 0:
            ml = {
                "primary_label": "ct_tumor_label_missing",
                "tb_probability": None,
                "risk_level": "low",
                "classifier": None,
                "decision_basis": (
                    "The mask contains no tumor label. This means the segmentation produced no tumor region, "
                    "not that there is no tumor; clinician review of the images is required."
                ),
            }
        else:
            ml = {
                "primary_label": "ct_lesion_characterized",
                "tb_probability": None,
                "risk_level": "moderate",
                "classifier": None,
                "decision_basis": (
                    "A tumor-labelled region was characterised with 3D radiomics (volume, extent, intensity, texture). "
                    "No CT classifier is available; the report describes the region and defers to the clinician."
                ),
            }
        if not qc.passed:
            ml["qc_warning"] = "quality control failed; interpret with caution: " + ", ".join(qc.flags)
        elif qc.flags:
            ml["qc_warning"] = "quality control notes: " + ", ".join(qc.flags)

        overlay, index = render_ct_overlay(volume_array, mask_array)
        return CTAnalysisResult(
            image_id=image_id,
            qc=qc,
            tumor=tumor,
            liver=liver,
            radiomics_features=features,
            ml_results=ml,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            overlay=overlay,
            overlay_slice=index,
            geometry=_geometry(volume),
        )


def make_synthetic_ct(
    out_dir: str | Path, size: tuple[int, int, int] = (64, 64, 48), with_tumor: bool = True, seed: int = 0
) -> tuple[Path, Path]:
    """Write a small synthetic abdominal CT phantom + label map (liver ellipsoid, optional tumor sphere).

    Used by tests and by the fake inference backend; returns (volume_path, mask_path) as .nii.gz files.
    """
    import SimpleITK as sitk

    rng = np.random.default_rng(seed)
    nx, ny, nz = size
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    volume = rng.normal(-30, 15, size=(nz, ny, nx)).astype(np.float32)  # soft tissue / fat background
    liver = ((x - nx * 0.45) / (nx * 0.3)) ** 2 + ((y - ny * 0.5) / (ny * 0.28)) ** 2 + (
        (z - nz * 0.5) / (nz * 0.35)
    ) ** 2 <= 1
    volume[liver] = rng.normal(60, 8, size=int(liver.sum()))
    labels = np.zeros((nz, ny, nx), dtype=np.uint8)
    labels[liver] = LABEL_LIVER
    if with_tumor:
        tumor = ((x - nx * 0.5) / (nx * 0.09)) ** 2 + ((y - ny * 0.52) / (ny * 0.09)) ** 2 + (
            (z - nz * 0.5) / (nz * 0.1)
        ) ** 2 <= 1
        tumor &= liver
        volume[tumor] = rng.normal(25, 10, size=int(tumor.sum()))
        labels[tumor] = LABEL_TUMOR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vol_img = sitk.GetImageFromArray(volume)
    vol_img.SetSpacing((0.8, 0.8, 2.5))
    lab_img = sitk.GetImageFromArray(labels)
    lab_img.CopyInformation(vol_img)
    vol_path, mask_path = out_dir / "ct_volume.nii.gz", out_dir / "ct_mask.nii.gz"
    sitk.WriteImage(vol_img, str(vol_path))
    sitk.WriteImage(lab_img, str(mask_path))
    return vol_path, mask_path
