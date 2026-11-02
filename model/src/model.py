"""Model definitions for lesion segmentation."""

from __future__ import annotations

import tensorflow as tf


def _conv_block(inputs: tf.Tensor, filters: int, use_batchnorm: bool = True) -> tf.Tensor:
    """Two-layer Conv -> BN -> ReLU block used across the U-Net."""
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(inputs)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_unet(
    input_shape: tuple[int, int, int],
    base_filters: int = 32,
    depth: int = 4,
    use_batchnorm: bool = True,
) -> tf.keras.Model:
    """Construct a configurable 2D U-Net model."""
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = inputs
    skips: list[tf.Tensor] = []

    for d in range(depth):
        filters = base_filters * (2**d)
        x = _conv_block(x, filters, use_batchnorm=use_batchnorm)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    bottleneck_filters = base_filters * (2**depth)
    x = _conv_block(x, bottleneck_filters, use_batchnorm=use_batchnorm)

    for d in reversed(range(depth)):
        filters = base_filters * (2**d)
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, skips[d]])
        x = _conv_block(x, filters, use_batchnorm=use_batchnorm)

    outputs = tf.keras.layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="mask")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="unet_segmentation")
    return model
