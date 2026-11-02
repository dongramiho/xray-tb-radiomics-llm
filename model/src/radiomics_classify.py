"""Train simple radiomics-only classifiers (LogReg, RandomForest) and report feature/group importances."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Radiomics-only TB/Normal classifier and importance report.")
    parser.add_argument(
        "--features_csv",
        type=Path,
        required=True,
        help="CSV with radiomics features (columns include image_id, image_path, mask_path).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for split and models.",
    )
    return parser.parse_args()


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Assign TB=1 if image_id or filename contains '_1', else 0."""
    return df["image_id"].str.contains("_1").astype(int)


def feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = {"image_id", "image_path", "mask_path", "label"}
    return [c for c in df.columns if c not in drop_cols]


def group_importance(feature_importance: pd.Series) -> pd.Series:
    groups = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape2D"]
    scores = {}
    for g in groups:
        cols = [c for c in feature_importance.index if f"_{g}_" in c]
        if cols:
            scores[g] = float(feature_importance[cols].sum())
    return pd.Series(scores).sort_values(ascending=False)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features_csv)
    if "image_id" not in df.columns:
        raise ValueError("CSV must contain an image_id column.")

    df["label"] = build_labels(df)
    feats = feature_columns(df)
    X = df[feats]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    print("=== Dataset ===")
    print(f"Samples: {len(df)}, Features: {len(feats)}, Positives (TB): {int(y.sum())}, Negatives: {len(y)-int(y.sum())}")
    if y.nunique() < 2:
        print("Only one class present; need both TB=1 and Normal=0 samples for classification.")
        return

    # Logistic Regression baseline
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, n_jobs=-1))
    logreg.fit(X_train, y_train)
    proba_lr = logreg.predict_proba(X_test)[:, 1]
    pred_lr = (proba_lr >= 0.5).astype(int)
    auc_lr = roc_auc_score(y_test, proba_lr)

    # Random Forest
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

    print("\n=== Logistic Regression ===")
    print(f"ROC-AUC: {auc_lr:.4f}")
    print(classification_report(y_test, pred_lr, digits=4))

    print("=== Random Forest ===")
    print(f"ROC-AUC: {auc_rf:.4f}")
    print(classification_report(y_test, pred_rf, digits=4))

    importances = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    print("Top 10 feature importances (RF):")
    print(importances.head(10))

    group_imp = group_importance(importances)
    print("\nGroup importances (sum of RF importances):")
    print(group_imp)

    print("\nNote: 내부 데이터 전용 결과이며, 외부 검증이 필요합니다. 임상적 판단에 직접 사용해서는 안 됩니다.")


if __name__ == "__main__":
    main()
