"""Train and evaluate the radiomics TB classifier with cross-validation, then export a bundle.

Input: a features CSV produced by ``extract_dataset_features.py`` (one row per image, with ``label``
and ``detected`` columns plus ``original_*`` radiomics columns). Only rows with a detected lesion
region are used, because radiomics features are undefined for empty masks; images without a
detected region are handled by the "no lesion detected" gate in ``TBPipeline``.

Outputs (``outputs/classifier/`` by default):
  - ``tb_classifier.joblib``   the fitted estimator + feature names + metadata (loaded by the backend)
  - ``classifier_report.json`` cross-validated metrics, feature importances, dataset description

Example:
    python -m cxr_tb.src.train_classifier --features_csv outputs/evaluation/pipeline_features.csv --model auto
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from cxr_tb import __version__ as package_version
from cxr_tb.config import CLASSIFIER_DIR, EVALUATION_DIR, RANDOM_SEED
from cxr_tb.src.pipeline import ClassifierBundle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_GROUPS = ["shape2D", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
DEFAULT_DATASET_NOTE = (
    "Rows are images with a segmented lesion region; normals here are false-positive regions on normal CXRs."
)
NON_FEATURE_COLUMNS = {
    "image_id",
    "image_path",
    "mask_path",
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
    "group",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate and export the TB radiomics classifier.")
    parser.add_argument("--features_csv", type=Path, default=EVALUATION_DIR / "pipeline_features.csv")
    parser.add_argument("--output_dir", type=Path, default=CLASSIFIER_DIR)
    parser.add_argument(
        "--model",
        choices=["logreg", "rf", "auto"],
        default="auto",
        help="Estimator to export; 'auto' keeps the one with the higher CV ROC-AUC.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--group_column",
        default=None,
        help="Optional column identifying the patient, to keep a patient's images in one fold.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=RANDOM_SEED)
    parser.add_argument("--dataset_note", default="", help="Free-text provenance note stored in the bundle.")
    return parser.parse_args()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c.startswith("original_")]


def build_estimator(kind: str, random_state: int) -> Pipeline:
    if kind == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"),
        )
    if kind == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    raise ValueError(kind)


def cross_validate(
    kind: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None, folds: int, threshold: float, seed: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Return CV metrics and the out-of-fold probabilities (aligned with X)."""
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)

    oof = np.zeros(len(y), dtype=float)
    fold_auc: list[float] = []
    for train_idx, test_idx in split_iter:
        est = build_estimator(kind, seed)
        est.fit(X.iloc[train_idx], y[train_idx])
        proba = est.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = proba
        fold_auc.append(float(roc_auc_score(y[test_idx], proba)))

    pred = (oof >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "folds": folds,
        "roc_auc_mean": round(float(np.mean(fold_auc)), 4),
        "roc_auc_std": round(float(np.std(fold_auc)), 4),
        "roc_auc_oof": round(float(roc_auc_score(y, oof)), 4),
        "average_precision_oof": round(float(average_precision_score(y, oof)), 4),
        "brier_oof": round(float(brier_score_loss(y, oof)), 4),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "threshold": threshold,
    }, oof


def end_to_end_metrics(
    full: pd.DataFrame, detected_index: pd.Index, oof: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Image-level metrics of gate + classifier: images without a detected region count as predicted normal."""
    score = pd.Series(0.0, index=full.index)
    score.loc[detected_index] = oof
    y_all = full["label"].astype(int).to_numpy()
    s_all = score.to_numpy()
    pred = (s_all >= threshold).astype(int)
    tp = int(((pred == 1) & (y_all == 1)).sum())
    tn = int(((pred == 0) & (y_all == 0)).sum())
    fp = int(((pred == 1) & (y_all == 0)).sum())
    fn = int(((pred == 0) & (y_all == 1)).sum())
    return {
        "n_images": int(len(y_all)),
        "roc_auc": round(float(roc_auc_score(y_all, s_all)), 4),
        "average_precision": round(float(average_precision_score(y_all, s_all)), 4),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "accuracy": round((tp + tn) / len(y_all), 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "undetected_images": int((~full.index.isin(detected_index)).sum()),
    }


def importances(kind: str, estimator: Pipeline, features: list[str]) -> dict[str, float]:
    final = estimator.steps[-1][1]
    if kind == "rf":
        values = final.feature_importances_
    else:
        values = final.coef_[0]
    return {name: float(v) for name, v in zip(features, values, strict=True)}


def group_importance(imp: dict[str, float]) -> dict[str, float]:
    total = sum(abs(v) for v in imp.values()) or 1.0
    return {g: round(sum(abs(v) for k, v in imp.items() if f"_{g}_" in k) / total, 4) for g in FEATURE_GROUPS}


def main() -> None:
    args = parse_args()
    full = pd.read_csv(args.features_csv)
    if "label" not in full.columns:
        raise SystemExit("features CSV must contain 'label'.")
    full = full.dropna(subset=["label"])
    df = full[full["detected"] == 1].copy() if "detected" in full.columns else full.copy()
    features = feature_columns(df)
    if not features:
        raise SystemExit("No radiomics feature columns (original_*) found.")
    X = df[features].astype(float)
    y = df["label"].astype(int).to_numpy()
    groups = df[args.group_column].to_numpy() if args.group_column else None

    counts = {"tb": int(y.sum()), "normal": int(len(y) - y.sum())}
    if min(counts.values()) < args.folds:
        raise SystemExit(f"Not enough samples per class for {args.folds}-fold CV: {counts}")

    cv_results: dict[str, Any] = {}
    oof_by_kind: dict[str, np.ndarray] = {}
    for kind in ("logreg", "rf"):
        cv_results[kind], oof_by_kind[kind] = cross_validate(
            kind, X, y, groups, args.folds, args.threshold, args.random_state
        )
    chosen = args.model
    if chosen == "auto":
        chosen = max(cv_results, key=lambda k: cv_results[k]["roc_auc_oof"])
    end_to_end = (
        end_to_end_metrics(full, df.index, oof_by_kind[chosen], args.threshold) if "detected" in full.columns else None
    )

    estimator = build_estimator(chosen, args.random_state)
    estimator.fit(X, y)
    imp = importances(chosen, estimator, features)
    top10 = sorted(imp.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]

    metadata: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": package_version,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "features_csv": str(args.features_csv),
        "dataset": {
            "samples": int(len(y)),
            "tb": counts["tb"],
            "normal": counts["normal"],
            "features": len(features),
            "note": args.dataset_note or DEFAULT_DATASET_NOTE,
        },
        "cross_validation": cv_results,
        "end_to_end_image_level": end_to_end,
        "selected_model": chosen,
        "feature_importance": imp,
        "feature_importance_top10": [{"feature": k, "importance": round(v, 6)} for k, v in top10],
        "group_importance": group_importance(imp),
        "limitations": [
            "Single-centre public dataset (Shenzhen); external validation required.",
            "Image-level labels only; lesion masks come from the U-Net, not radiologist annotations.",
            "Research prototype, not for clinical decision-making.",
        ],
    }

    bundle = ClassifierBundle(
        estimator=estimator,
        feature_names=features,
        threshold=args.threshold,
        model_type=chosen,
        metadata=metadata,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle.save(args.output_dir / "tb_classifier.joblib")
    (args.output_dir / "classifier_report.json").write_text(json.dumps(metadata, indent=2))

    summary = {
        "selected_model": chosen,
        "cross_validation": cv_results,
        "end_to_end_image_level": end_to_end,
        "dataset": metadata["dataset"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved bundle to {args.output_dir / 'tb_classifier.joblib'}")


if __name__ == "__main__":
    main()
