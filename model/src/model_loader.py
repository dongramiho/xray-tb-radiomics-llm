"""Utilities for loading trained Keras models with backwards-compatible layers."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from cxr_tb.src.losses import binary_crossentropy_dice_loss, dice_coefficient


def load_model_compat(model_path: Path) -> tf.keras.Model:
    """
    Load a Keras model while tolerating legacy configs (e.g., Conv2DTranspose with 'groups').

    Some saved models include a 'groups' field that older TF/Keras builds reject. This wrapper
    strips the unsupported argument and registers custom losses/metrics.
    """

    class Conv2DTransposeCompat(tf.keras.layers.Conv2DTranspose):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    custom_objects = {
        "dice_coefficient": dice_coefficient,
        "binary_crossentropy_dice_loss": binary_crossentropy_dice_loss,
        "Conv2DTranspose": Conv2DTransposeCompat,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
