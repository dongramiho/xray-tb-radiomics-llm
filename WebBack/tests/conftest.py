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

SEED_EMAIL = "doctor@test.com"
SEED_PASSWORD = "test-password"
ADMIN_EMAIL = "admin@test.com"
ADMIN_PASSWORD = "admin-password"


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
            "SEED_DOCTOR_EMAIL": SEED_EMAIL,
            "SEED_DOCTOR_PASSWORD": SEED_PASSWORD,
            "SEED_ADMIN_EMAIL": ADMIN_EMAIL,
            "SEED_ADMIN_PASSWORD": ADMIN_PASSWORD,
            "FRONTEND_BASE_URL": "http://front.test",
            "RATE_LIMIT_ENABLED": "false",
            "MAX_UPLOAD_MB": "1",
            "PUBMED_ENABLED": "false",
            "GRAPH_CHECKPOINT_PATH": str(tmp_path_factory.mktemp("graph") / "checkpoints.sqlite"),
        }
    )
    from settings import Settings, get_settings

    get_settings.cache_clear()
    settings = Settings()
    assert settings.inference_backend == "fake"
    return settings


@pytest.fixture(scope="session")
def app(settings):
    from main import create_app

    return create_app(settings)


@pytest.fixture(scope="session")
def anon_client(app):
    """Client without credentials."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def token(anon_client) -> str:
    res = anon_client.post("/auth/login", json={"email": SEED_EMAIL, "password": SEED_PASSWORD})
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


@pytest.fixture(scope="session")
def client(app, token):
    """Client authenticated as the seeded clinician."""
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as test_client:
        yield test_client


@pytest.fixture
def sample_png(tmp_path) -> Path:
    path = tmp_path / "cxr.png"
    Image.effect_noise((640, 600), 40).convert("L").save(path)
    return path


@pytest.fixture(scope="session")
def admin_client(app, anon_client):
    """Client authenticated as the seeded unit administrator."""
    res = anon_client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert res.status_code == 200, res.text
    with TestClient(app, headers={"Authorization": f"Bearer {res.json()['access_token']}"}) as test_client:
        yield test_client
