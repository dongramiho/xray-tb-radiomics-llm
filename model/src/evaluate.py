"""Evaluate a trained segmentation model on the held-out test set."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINT_DIR  # noqa: E402
from src.dataset import build_datasets
from src.losses import SMOOTH, dice_coefficient
from src.model_loader import load_model_compat


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Intersection-over-Union metric for binary segmentation."""
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred >= threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return tf.reduce_mean(iou)


def evaluate_model(model_path: Path) -> Tuple[float, float, float]:
    """Load dataset/model, run predictions, and compute summary metrics."""
    datasets = build_datasets()
    test_ds = datasets["test"]
    if test_ds is None:
        raise ValueError("Test split is empty. Increase TEST_SPLIT or provide separate CSV rows.")

    model = load_model_compat(model_path)

    dice_scores = []
    iou_scores = []
    binary_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    for images, masks in test_ds:
        preds = model.predict(images, verbose=0)
        dice_scores.append(float(dice_coefficient(masks, preds).numpy()))
        iou_scores.append(float(iou_score(masks, preds).numpy()))
        binary_accuracy_metric.update_state(masks, preds)

    dice_mean = float(np.mean(dice_scores))
    iou_mean = float(np.mean(iou_scores))
    binary_acc = float(binary_accuracy_metric.result().numpy())
    return dice_mean, iou_mean, binary_acc


def main() -> None:
    """Script entrypoint."""
    model_path = Path(CHECKPOINT_DIR) / "best_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find saved model at {model_path}")

    dice_mean, iou_mean, binary_acc = evaluate_model(model_path)
    print("Test metrics:")
    print(f"  Dice coefficient : {dice_mean:.4f}")
    print(f"  IoU              : {iou_mean:.4f}")
    print(f"  Binary accuracy  : {binary_acc:.4f}")


if __name__ == "__main__":
    main()
