"""Training script for chest X-ray lesion segmentation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import tensorflow as tf
from cxr_tb.config import (
    CHANNELS,
    CHECKPOINT_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    LOG_DIR,
    NUM_EPOCHS,
    OUTPUT_DIR,
)
from cxr_tb.src.dataset import build_datasets
from cxr_tb.src.losses import binary_crossentropy_dice_loss, dice_coefficient
from cxr_tb.src.model import build_unet


def prepare_output_dirs() -> None:
    """Create necessary output directories."""
    for directory in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Entrypoint for model training."""
    prepare_output_dirs()
    datasets = build_datasets()
    train_ds = datasets["train"]
    val_ds = datasets["val"]

    model = build_unet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=binary_crossentropy_dice_loss,
        metrics=[dice_coefficient, "binary_accuracy"],
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = Path(LOG_DIR) / f"run_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = "val_dice_coefficient" if val_ds is not None else "dice_coefficient"
    best_model_path = Path(CHECKPOINT_DIR) / "best_model.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=run_log_dir),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = Path(CHECKPOINT_DIR) / "final_model.h5"
    model.save(str(final_model_path))
    print(f"Training completed. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
