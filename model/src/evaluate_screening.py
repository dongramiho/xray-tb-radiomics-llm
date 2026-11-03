"""Image-level screening metrics for the segmentation stage.

Given ``pipeline_features.csv`` (from ``extract_dataset_features.py``), treat the segmented lesion area
fraction (and the maximum pixel probability) as a TB-vs-normal score and report ROC-AUC, average
precision, and sensitivity/specificity at the deployed detection threshold. This is the honest,
label-only evaluation that does not depend on synthetic normals.

Example:
    python -m cxr_tb.src.evaluate_screening --features_csv outputs/evaluation/pipeline_features.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cxr_tb.config import EVALUATION_DIR, MIN_LESION_AREA_FRACTION
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-level detection metrics from pipeline features.")
    parser.add_argument("--features_csv", type=Path, default=EVALUATION_DIR / "pipeline_features.csv")
    parser.add_argument("--output_json", type=Path, default=EVALUATION_DIR / "screening_metrics.json")
    parser.add_argument("--area_threshold", type=float, default=MIN_LESION_AREA_FRACTION)
    return parser.parse_args()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "accuracy": round(float((tp + tn) / max(len(y_true), 1)), 4),
    }


def score_summary(y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(y_true, score)
    youden = tpr - fpr
    best = int(np.argmax(youden))
    return {
        "roc_auc": round(float(roc_auc_score(y_true, score)), 4),
        "average_precision": round(float(average_precision_score(y_true, score)), 4),
        "youden_threshold": float(thresholds[best]) if np.isfinite(thresholds[best]) else None,
        "youden_sensitivity": round(float(tpr[best]), 4),
        "youden_specificity": round(float(1 - fpr[best]), 4),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features_csv)
    if "label" not in df.columns or df["label"].isna().any():
        raise SystemExit("features CSV must contain a complete 'label' column (1 = TB, 0 = normal).")

    y = df["label"].astype(int).to_numpy()
    area = df["mask_area_fraction"].astype(float).to_numpy()
    max_prob = df["max_probability"].astype(float).to_numpy()

    report: dict[str, Any] = {
        "n_images": int(len(df)),
        "n_tb": int(y.sum()),
        "n_normal": int(len(y) - y.sum()),
        "detection_threshold_area_fraction": args.area_threshold,
        "at_deployed_threshold": binary_metrics(y, (area >= args.area_threshold).astype(int)),
        "score_mask_area_fraction": score_summary(y, area),
        "score_max_probability": score_summary(y, max_prob),
        "mask_area_fraction_percentiles": {
            label_name: {
                str(p): round(float(np.percentile(area[y == label_value], p)), 5) for p in (10, 25, 50, 75, 90)
            }
            for label_name, label_value in (("normal", 0), ("tb", 1))
            if (y == label_value).any()
        },
        "source_csv": str(args.features_csv),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
