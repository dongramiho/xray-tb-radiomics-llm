"""Extract radiomics features from chest X-ray images and binary masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
    from radiomics import featureextractor
except ImportError as exc:  # pragma: no cover - dependency hint for users
    raise SystemExit(
        "Missing dependencies for radiomics. Please install with:\n"
        "  python3 -m pip install --upgrade pip\n"
        "  python3 -m pip install pyradiomics SimpleITK\n"
    ) from exc

from cxr_tb.config import MODEL_ROOT as PROJECT_ROOT

DEFAULT_PRED_MASK_DIR = PROJECT_ROOT / "outputs" / "predictions" / "batch" / "masks"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "radiomics" / "features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PyRadiomics features for images in a CSV manifest.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset.csv",
        help="CSV with image_path/mask_path columns.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination CSV file for extracted features.",
    )
    parser.add_argument(
        "--pred_mask_dir",
        type=Path,
        default=DEFAULT_PRED_MASK_DIR,
        help="Directory containing predicted masks named <stem>_mask.png.",
    )
    parser.add_argument(
        "--mask_source",
        choices=["auto", "pred", "gt"],
        default="auto",
        help="Which masks to use: predicted, ground-truth, or auto (pred then gt).",
    )
    parser.add_argument(
        "--image_column",
        default="image_path",
        help="CSV column containing image paths.",
    )
    parser.add_argument(
        "--mask_column",
        default="mask_path",
        help="CSV column containing ground-truth mask paths.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Label value inside the mask to analyze.",
    )
    parser.add_argument(
        "--bin_width",
        type=float,
        default=25.0,
        help="Gray level bin width for intensity discretization.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply z-score normalization before feature extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )
    return parser.parse_args()


def build_extractor(bin_width: float, normalize: bool) -> featureextractor.RadiomicsFeatureExtractor:
    """Configure PyRadiomics extractor with common feature classes."""
    settings = {
        "binWidth": bin_width,
        "normalize": normalize,
        "normalizeScale": 100,
        "resampledPixelSpacing": None,
        "interpolator": sitk.sitkBSpline,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("shape2D")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")
    return extractor


def pick_mask(
    row: pd.Series,
    image_column: str,
    mask_column: str,
    mask_source: str,
    pred_mask_dir: Path,
) -> Path | None:
    """Choose predicted or ground-truth mask based on availability and user preference."""
    image_path = Path(row[image_column])
    pred_mask = pred_mask_dir / f"{image_path.stem}_mask.png"

    if mask_source in ("pred", "auto") and pred_mask.exists():
        return pred_mask

    if mask_source in ("gt", "auto"):
        if mask_column not in row:
            return None
        gt_mask = Path(row[mask_column])
        if gt_mask.exists():
            return gt_mask

    return None


def sanitize_features(result: dict[str, Any]) -> dict[str, Any]:
    """Drop diagnostic keys and cast numpy scalars to native Python types."""
    clean: dict[str, Any] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        if isinstance(value, (np.generic,)):
            value = value.item()
        clean[key] = value
    return clean


def ensure_binary_mask(mask_path: Path) -> Path:
    """Force mask values to 0/1 so PyRadiomics interprets labels correctly."""
    mask_img = sitk.ReadImage(str(mask_path))
    binary = sitk.BinaryThreshold(
        mask_img, lowerThreshold=0.5, upperThreshold=sys.maxsize, insideValue=1, outsideValue=0
    )
    tmp_path = mask_path.parent / f"{mask_path.stem}_bin{mask_path.suffix}"
    sitk.WriteImage(binary, str(tmp_path))
    return tmp_path


def ensure_scalar_image(image_path: Path) -> Path:
    """Convert multi-channel images (e.g., RGB PNG) to single-channel float for PyRadiomics."""
    image = sitk.ReadImage(str(image_path))
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorMagnitude(image)
    image = sitk.Cast(image, sitk.sitkFloat32)
    tmp_path = image_path.parent / f"{image_path.stem}_scalar.nrrd"
    sitk.WriteImage(image, str(tmp_path))
    return tmp_path


def extract_features(
    df: pd.DataFrame,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    image_column: str,
    mask_column: str,
    mask_source: str,
    pred_mask_dir: Path,
    label: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        image_path = Path(row[image_column])
        if not image_path.exists():
            print(f"[skip] Missing image: {image_path}")
            continue

        mask_path = pick_mask(row, image_column, mask_column, mask_source, pred_mask_dir)
        if mask_path is None:
            print(f"[skip] No mask found for: {image_path.name}")
            continue
        if not mask_path.exists():
            print(f"[skip] Mask file does not exist: {mask_path}")
            continue

        try:
            scalar_image_path = ensure_scalar_image(image_path)
            bin_mask_path = ensure_binary_mask(mask_path)
            result = extractor.execute(str(scalar_image_path), str(bin_mask_path), label=label)
            features = sanitize_features(result)
            rows.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    **features,
                }
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"[skip] Extraction failed for {image_path.name}: {exc}")
            continue
        finally:
            if "scalar_image_path" in locals() and Path(scalar_image_path).exists():
                Path(scalar_image_path).unlink(missing_ok=True)
            if "bin_mask_path" in locals() and Path(bin_mask_path).exists():
                Path(bin_mask_path).unlink(missing_ok=True)

    return rows


def main() -> None:
    args = parse_args()
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_csv} already exists. Use --overwrite to replace it.")

    df = pd.read_csv(args.csv_path)
    if args.image_column not in df.columns:
        raise ValueError(f"CSV is missing column: {args.image_column}")
    if args.mask_source in ("gt", "auto") and args.mask_column not in df.columns:
        raise ValueError(f"CSV is missing column: {args.mask_column}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(bin_width=args.bin_width, normalize=args.normalize)

    results = extract_features(
        df=df,
        extractor=extractor,
        image_column=args.image_column,
        mask_column=args.mask_column,
        mask_source=args.mask_source,
        pred_mask_dir=args.pred_mask_dir,
        label=args.label,
    )

    if not results:
        raise SystemExit("No features were extracted. Check image/mask paths.")

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"Saved radiomics features to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
