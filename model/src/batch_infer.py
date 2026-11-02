"""Generate masks and overlays for every image listed in the dataset CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CSV_PATH, PREDICTION_DIR, RANDOM_SEED  # noqa: E402
from src.infer import preprocess_image, save_mask, save_overlay  # noqa: E402
from src.model_loader import load_model_compat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for all images defined in the CSV manifest.")
    parser.add_argument(
        "--model_path",
        default=str(Path("outputs/checkpoints/best_model.h5")),
        help="Path to the trained TensorFlow model (.h5). Defaults to best checkpoint.",
    )
    parser.add_argument(
        "--csv_path",
        default=str(Path(CSV_PATH)),
        help="Path to CSV containing image_path/mask_path columns. Defaults to config CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(PREDICTION_DIR) / "batch"),
        help="Directory where masks and overlays will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binarizing the predicted mask.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of random samples to process. Use -1 to process all rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of skipping them.",
    )
    return parser.parse_args()


def prepare_model(model_path: Path) -> tf.keras.Model:
    return load_model_compat(model_path)


def run_batch_inference(
    model: tf.keras.Model,
    data_frame: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    overwrite: bool,
) -> Tuple[int, int]:
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for _, row in data_frame.iterrows():

        image_path = Path(row["image_path"])
        if not image_path.exists():
            print(f"[skip] Missing image: {image_path}")
            skipped += 1
            continue

        stem = image_path.stem
        mask_path = masks_dir / f"{stem}_mask.png"
        overlay_path = overlays_dir / f"{stem}_overlay.png"

        if not overwrite and mask_path.exists() and overlay_path.exists():
            skipped += 1
            continue

        batched_image, image_np = preprocess_image(image_path)
        prediction = model.predict(batched_image, verbose=0)[0, ..., 0]
        binary_mask = (prediction >= threshold).astype(np.float32)

        save_mask(binary_mask, mask_path)
        save_overlay(image_np, binary_mask, overlay_path)
        processed += 1

        if processed % 25 == 0:
            print(f"[info] Processed {processed} samples...")

    return processed, skipped


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.count >= 0:
        df = df.sample(n=min(args.count, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain an 'image_path' column.")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = prepare_model(model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed, skipped = run_batch_inference(
        model=model,
        data_frame=df,
        output_dir=output_dir,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )

    print(f"Batch inference complete. Generated overlays for {processed} images (skipped {skipped}).")
    print(f"Outputs saved under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
