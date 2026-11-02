"""Custom losses and metrics for segmentation."""

from __future__ import annotations

import tensorflow as tf

SMOOTH = 1e-6


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute the Sørensen–Dice coefficient."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)
    return tf.reduce_mean(dice)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice loss = 1 - Dice coefficient."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def binary_crossentropy_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combine BCE and Dice loss for stable training."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)
