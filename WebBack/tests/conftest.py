"""Test fixtures: isolated SQLite DB, fake inference + fake LLM, no TensorFlow / OpenAI needed."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def tmp_static(tmp_path_factory):
    return tmp_path_factory.mktemp("static")


@pytest.fixture(scope="session")
def settings(tmp_path_factory, tmp_static):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    os.environ.update(
        {
            "DATABASE_URL": f"sqlite:///{db_path}",
            "STATIC_DIR": str(tmp_static),
            "INFERENCE_BACKEND": "fake",
            "LLM_BACKEND": "fake",
            "OPENAI_API_KEY": "",
            "PUBLIC_BASE_URL": "http://testserver",
        }
    )
    from settings import Settings, get_settings

    get_settings.cache_clear()
    settings = Settings()
    assert settings.inference_backend == "fake"
    return settings


@pytest.fixture(scope="session")
def client(settings):
    from main import create_app

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_png(tmp_path) -> Path:
    path = tmp_path / "cxr.png"
    Image.effect_noise((640, 600), 40).convert("L").save(path)
    return path
