"""Run the full pipeline over a manifest and write one row per image.

For every image: U-Net mask -> lesion summary -> (if a lesion is detected) radiomics features.
The output CSV feeds both ``evaluate_screening.py`` (image-level detection metrics) and
``train_classifier.py`` (TB-vs-normal classifier on detected regions). Because the same
``TBPipeline`` code is used at serving time, the classifier sees identical features in production.

Example:
    python -m cxr_tb.src.extract_dataset_features --csv_path data/dataset.csv \
        --output_csv outputs/evaluation/pipeline_features.csv --save_overlays
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from cxr_tb.config import (
    CSV_PATH,
    DEFAULT_SEGMENTATION_MODEL,
    EVALUATION_DIR,
    MIN_COMPONENT_PX,
    MIN_LESION_AREA_FRACTION,
    PREDICTION_DIR,
)
from cxr_tb.src.pipeline import RadiomicsExtractor, SegmentationModel, TBPipeline

META_COLUMNS = [
    "image_id",
    "label",
    "detected",
    "mask_area_fraction",
    "num_components",
    "max_probability",
    "zones",
    "largest_component_fraction",
    "width",
    "height",
    "seconds",
    "qc_passed",
    "qc_usable_for_patient",
    "qc_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pipeline features for every image in a manifest.")
    parser.add_argument("--csv_path", type=Path, default=CSV_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument("--output_csv", type=Path, default=EVALUATION_DIR / "pipeline_features.csv")
    parser.add_argument("--min_area_fraction", type=float, default=MIN_LESION_AREA_FRACTION)
    parser.add_argument("--min_component_px", type=int, default=MIN_COMPONENT_PX)
    parser.add_argument(
        "--save_overlays", action="store_true", help="Also write masks/overlays under outputs/predictions/pipeline."
    )
    parser.add_argument("--limit", type=int, default=-1, help="Process only the first N rows (debugging).")
    parser.add_argument("--no_radiomics", action="store_true", help="Segmentation statistics only (fast).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    if args.limit > 0:
        df = df.head(args.limit)
    if "image_path" not in df.columns:
        raise ValueError("Manifest must contain an 'image_path' column.")

    pipeline = TBPipeline(
        segmentation=SegmentationModel(args.model_path),
        extractor=None if args.no_radiomics else RadiomicsExtractor(),
        classifier=None,
        min_area_fraction=args.min_area_fraction,
        min_component_px=args.min_component_px,
    )

    out_dir = PREDICTION_DIR / "pipeline"
    if args.save_overlays:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started = time.time()
    for _, row in df.iterrows():
        image_path = Path(row["image_path"])
        if not image_path.exists():
            print(f"[skip] missing image: {image_path}")
            continue
        t0 = time.time()
        result = pipeline.analyze(image_path, image_id=image_path.stem)
        seconds = time.time() - t0

        lesion = result.lesion
        record: dict[str, Any] = {
            "image_id": result.image_id,
            "label": row.get("label"),
            "detected": int(lesion.detected),
            "mask_area_fraction": lesion.mask_area_fraction,
            "num_components": lesion.num_components,
            "max_probability": lesion.max_probability,
            "zones": "|".join(lesion.zones),
            "largest_component_fraction": lesion.components[0].area_fraction if lesion.components else 0.0,
            "width": result.image_size[0],
            "height": result.image_size[1],
            "seconds": round(seconds, 2),
            "qc_passed": int(result.qc.passed) if result.qc else 1,
            "qc_usable_for_patient": int(result.qc.usable_for_patient) if result.qc else 1,
            "qc_flags": "|".join(result.qc.flags) if result.qc else "",
        }
        if result.radiomics_features:
            record.update(result.radiomics_features)
        rows.append(record)

        if args.save_overlays:
            result.mask_image().save(out_dir / "masks" / f"{result.image_id}_mask.png")
            result.overlay_image().save(out_dir / "overlays" / f"{result.image_id}_overlay.png")

        done = len(rows)
        if done % 25 == 0 or done == len(df):
            elapsed = time.time() - started
            print(f"[{done}/{len(df)}] {elapsed / 60:.1f} min elapsed, {elapsed / done:.1f}s per image", flush=True)

    if not rows:
        raise SystemExit("No rows processed.")

    out = pd.DataFrame(rows)
    feature_cols = [c for c in out.columns if c not in META_COLUMNS]
    out = out[META_COLUMNS + feature_cols]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    summary = {
        "rows": int(len(out)),
        "detected": int(out["detected"].sum()),
        "by_label": {str(k): int(v) for k, v in out.groupby("label")["detected"].sum().items()}
        if "label" in out
        else {},
        "feature_columns": len(feature_cols),
        "model_path": str(args.model_path),
        "min_area_fraction": args.min_area_fraction,
        "min_component_px": args.min_component_px,
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
