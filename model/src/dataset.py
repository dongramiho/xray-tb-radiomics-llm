"""Utilities for reading CSV manifests and building tf.data pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from config import (
    BATCH_SIZE,
    CHANNELS,
    CSV_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
    SHUFFLE_BUFFER_SIZE,
    TEST_SPLIT,
    VALIDATION_SPLIT,
)


@dataclass
class DatasetSplits:
    """Container holding train/validation/test pandas DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_dataframe(csv_path: str | tf.Tensor = CSV_PATH) -> pd.DataFrame:
    """Load the CSV manifest that contains image_path/mask_path columns."""
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "mask_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path' and 'mask_path' columns.")
    return df


def split_dataframe(df: pd.DataFrame) -> DatasetSplits:
    """Split the dataframe into train/validation/test sets using configured ratios."""
    if VALIDATION_SPLIT + TEST_SPLIT >= 1.0:
        raise ValueError("VALIDATION_SPLIT + TEST_SPLIT must be < 1.")

    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    total_records = len(df)
    test_size = int(total_records * TEST_SPLIT)
    val_size = int(total_records * VALIDATION_SPLIT)

    test_df = df.iloc[:test_size]
    val_df = df.iloc[test_size : test_size + val_size]
    train_df = df.iloc[test_size + val_size :]

    if train_df.empty:
        raise ValueError("Not enough samples for the requested splits.")

    return DatasetSplits(train=train_df, val=val_df, test=test_df)


def _decode_image(path: tf.Tensor) -> tf.Tensor:
    """Read and decode an image from disk as float32 tensor."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=CHANNELS, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([None, None, CHANNELS])
    return image


def _decode_mask(path: tf.Tensor) -> tf.Tensor:
    """Read and decode a binary mask as float32 tensor."""
    mask_bytes = tf.io.read_file(path)
    mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.clip_by_value(mask, 0.0, 1.0)
    mask.set_shape([None, None, 1])
    return mask


def _resize_pair(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Resize image and mask to the configured spatial size."""
    image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, mask


def _augment(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply simple spatial and intensity augmentations."""
    flip_cond = tf.random.uniform(shape=[]) > 0.5
    image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip_cond, lambda: tf.image.flip_left_right(mask), lambda: mask)

    angle = tf.random.uniform([], minval=-0.08, maxval=0.08)
    image = tfa.image.rotate(image, angles=angle, fill_mode="reflect")
    mask = tfa.image.rotate(mask, angles=angle, fill_mode="reflect")

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, mask


def _load_example(image_path: tf.Tensor, mask_path: tf.Tensor, augment: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load, preprocess, and optionally augment a single sample."""
    image = _decode_image(image_path)
    mask = _decode_mask(mask_path)
    image, mask = _resize_pair(image, mask)
    if augment:
        image, mask = _augment(image, mask)
    return image, mask


def dataframe_to_dataset(df: pd.DataFrame, augment: bool, shuffle: bool) -> tf.data.Dataset:
    """Convert a dataframe split into a tf.data pipeline."""
    paths_ds = tf.data.Dataset.from_tensor_slices((df["image_path"].astype(str).values, df["mask_path"].astype(str).values))
    if shuffle:
        paths_ds = paths_ds.shuffle(buffer_size=min(SHUFFLE_BUFFER_SIZE, len(df)), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    def _load_map(image_path: tf.Tensor, mask_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return _load_example(image_path, mask_path, augment=augment)

    dataset = paths_ds.map(_load_map, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_datasets() -> Dict[str, tf.data.Dataset]:
    """Prepare train/validation/test datasets based on the configured CSV."""
    df = load_dataframe()
    splits = split_dataframe(df)
    datasets = {
        "train": dataframe_to_dataset(splits.train, augment=True, shuffle=True),
        "val": dataframe_to_dataset(splits.val, augment=False, shuffle=False) if not splits.val.empty else None,
        "test": dataframe_to_dataset(splits.test, augment=False, shuffle=False) if not splits.test.empty else None,
    }
    return datasets
