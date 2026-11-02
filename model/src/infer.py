"""Run inference on a single chest X-ray and save predictions/visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from src.model_loader import load_model_compat


def preprocess_image(image_path: Path) -> tuple[tf.Tensor, np.ndarray]:
    """Load the input image, normalize, resize, and return both batched and raw arrays."""
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=CHANNELS, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), method=tf.image.ResizeMethod.BILINEAR)
    image = tf.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    batched = tf.expand_dims(image, axis=0)
    return batched, image.numpy()


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save predicted mask as an 8-bit grayscale PNG."""
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(path)


def save_overlay(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    """Save overlay visualization with the predicted mask in red."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image.squeeze(), cmap="gray")
    plt.imshow(mask, cmap="Reds", alpha=0.4)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U-Net inference on a single X-ray image.")
    parser.add_argument("--image_path", required=True, help="Path to the input X-ray image.")
    parser.add_argument("--model_path", required=True, help="Path to a trained TensorFlow model (.h5).")
    parser.add_argument(
        "--output_path",
        required=True,
        help="File path (PNG) for saving the overlay visualization. A *_mask.png will also be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    model_path = Path(args.model_path)
    overlay_path = Path(args.output_path)
    if overlay_path.suffix == "":
        overlay_path = overlay_path.with_suffix(".png")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    mask_path = overlay_path.with_name(f"{overlay_path.stem}_mask.png")

    model = load_model_compat(model_path)

    batched_image, image_np = preprocess_image(image_path)
    prediction = model.predict(batched_image, verbose=0)[0, ..., 0]
    binary_mask = (prediction >= 0.5).astype(np.float32)

    save_mask(binary_mask, mask_path)
    save_overlay(image_np, binary_mask, overlay_path)

    print(f"Saved raw mask to {mask_path}")
    print(f"Saved overlay visualization to {overlay_path}")


if __name__ == "__main__":
    main()
