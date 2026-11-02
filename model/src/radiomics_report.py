"""Generate JSON report from radiomics features with simple classifiers and importance summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build radiomics-only classifiers and emit a JSON report.")
    parser.add_argument(
        "--features_csv",
        type=Path,
        required=True,
        help="CSV with radiomics features (image_id/image_path/mask_path + feature columns).",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("outputs/radiomics/classify_report.json"),
        help="Destination JSON report file.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio (default 0.2).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Assign TB=1 if image_id contains '_1', else 0."""
    return df["image_id"].str.contains("_1").astype(int)


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"image_id", "image_path", "mask_path", "label"}
    return [c for c in df.columns if c not in drop]


def group_importance(feature_importance: pd.Series) -> dict[str, float]:
    groups = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape2D"]
    scores = {}
    for g in groups:
        cols = [c for c in feature_importance.index if f"_{g}_" in c]
        if cols:
            scores[g] = float(feature_importance[cols].sum())
    return scores


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features_csv)
    if "image_id" not in df.columns:
        raise ValueError("CSV must contain image_id.")

    df["label"] = build_labels(df)
    feats = feature_columns(df)
    X = df[feats]
    y = df["label"]

    if y.nunique() < 2:
        raise SystemExit("Need at least two classes (TB/Normal).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, n_jobs=-1))
    logreg.fit(X_train, y_train)
    proba_lr = logreg.predict_proba(X_test)[:, 1]
    pred_lr = (proba_lr >= 0.5).astype(int)
    auc_lr = roc_auc_score(y_test, proba_lr)
    report_lr = classification_report(y_test, pred_lr, output_dict=True)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=args.random_state,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    pred_rf = (proba_rf >= 0.5).astype(int)
    auc_rf = roc_auc_score(y_test, proba_rf)
    report_rf = classification_report(y_test, pred_rf, output_dict=True)

    importances = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    top10 = (
        importances.head(10)
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
        .to_dict(orient="records")
    )

    result: dict[str, Any] = {
        "dataset": {
            "file": str(args.features_csv),
            "samples": int(len(df)),
            "features": int(len(feats)),
            "tb": int(y.sum()),
            "normal": int(len(y) - y.sum()),
            "note": "Normal 100 samples are synthetic; analysis only, not clinical.",
        },
        "logreg": {"roc_auc": float(auc_lr), "report": report_lr},
        "random_forest": {
            "roc_auc": float(auc_rf),
            "report": report_rf,
            "feature_importance_top10": top10,
            "group_importance": group_importance(importances),
        },
        "disclaimer": [
            "Synthetic normals used; external validation required.",
            "Not for clinical decision-making.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(f"Wrote JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
