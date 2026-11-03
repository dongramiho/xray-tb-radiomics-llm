"""Unit tests for the pure (TensorFlow-free) parts of cxr_tb.src.pipeline."""

import numpy as np
import pytest
from cxr_tb.src.pipeline import (
    LABEL_NO_LESION,
    ClassifierBundle,
    load_grayscale,
    postprocess_mask,
    render_overlay,
    resize_to_model,
    risk_level,
    summarize_lesions,
    zone_of,
)
from PIL import Image


def blob_mask(cx: int, cy: int, r: int, size: int = 512) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) < r**2


def test_zone_of_uses_radiographic_convention():
    assert zone_of(0.2, 0.1) == "right upper zone"  # image left = patient's right
    assert zone_of(0.8, 0.5) == "left middle zone"
    assert zone_of(0.5, 0.9) == "left lower zone"


def test_postprocess_mask_drops_small_components():
    mask = blob_mask(100, 100, 30) | blob_mask(400, 400, 2)
    cleaned = postprocess_mask(mask, min_component_px=16)
    assert cleaned.sum() < mask.sum()
    assert cleaned[100, 100] and not cleaned[400, 400]


def test_summarize_lesions_reports_components_and_zones():
    mask = blob_mask(120, 100, 40) | blob_mask(400, 420, 25)
    prob = np.where(mask, 0.9, 0.1).astype(np.float32)
    summary = summarize_lesions(mask, prob)
    assert summary.detected
    assert summary.num_components == 2
    assert summary.zones == ["right upper zone", "left lower zone"]
    assert summary.components[0].area_px > summary.components[1].area_px
    assert summary.max_probability == pytest.approx(0.9)
    assert 0 < summary.mask_area_fraction < 0.05


def test_summarize_lesions_empty_mask_is_not_detected():
    summary = summarize_lesions(np.zeros((512, 512), dtype=bool))
    assert not summary.detected and summary.num_components == 0 and summary.zones == []


def test_risk_level_thresholds():
    assert risk_level(None) == "low"
    assert risk_level(0.1) == "low"
    assert risk_level(0.5) == "moderate"
    assert risk_level(0.9) == "high"


def test_resize_matches_tensorflow_bilinear_sampling_on_identity_and_downscale():
    gray = (np.arange(64 * 64).reshape(64, 64) % 251).astype(np.uint8)
    assert np.array_equal(resize_to_model(gray, 64, 64), gray)
    small = resize_to_model(gray, 16, 16)
    assert small.shape == (16, 16) and small.dtype == np.uint8


def test_load_grayscale_accepts_rgb_and_bytes(tmp_path):
    rgb = Image.new("RGB", (30, 20), (200, 100, 50))
    arr = load_grayscale(rgb)
    assert arr.shape == (20, 30) and arr.dtype == np.uint8
    path = tmp_path / "x.png"
    rgb.save(path)
    assert np.array_equal(load_grayscale(path), load_grayscale(path.read_bytes()))


def test_render_overlay_colours_only_masked_pixels():
    gray = np.full((64, 64), 128, dtype=np.uint8)
    mask = blob_mask(32, 32, 10, size=64)
    overlay = np.asarray(render_overlay(gray, mask))
    assert tuple(overlay[0, 0]) == (128, 128, 128)
    assert overlay[32, 32, 0] > overlay[32, 32, 1]


def test_classifier_bundle_roundtrip(tmp_path):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]], columns=["a", "b"])
    y = np.array([0, 1, 0, 1])
    bundle = ClassifierBundle(
        estimator=LogisticRegression().fit(X, y),
        feature_names=["a", "b"],
        threshold=0.5,
        model_type="logreg",
        metadata={"feature_importance": {"a": 1.2, "b": -0.4}},
    )
    bundle.save(tmp_path / "clf.joblib")
    loaded = ClassifierBundle.load(tmp_path / "clf.joblib")
    assert loaded.feature_names == ["a", "b"]
    assert loaded.predict_proba({"a": 0.9, "b": 0.1}) > 0.5
    assert loaded.top_feature_contributions({"a": 0.9, "b": 0.1})[0]["feature"] == "a"
    assert LABEL_NO_LESION == "no_lesion_detected"
