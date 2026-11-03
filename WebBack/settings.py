"""Application settings, loaded from environment variables / a local .env file.

Every value that differs between a laptop, Docker Compose, and a real deployment lives here.
See ``.env.example`` for documentation of each variable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- service -------------------------------------------------------------------------
    app_name: str = "X-ray TB Radiomics CDS API"
    environment: str = Field(default="development", description="development | staging | production")
    public_base_url: str = Field(default="http://127.0.0.1:8000", description="URL clients use to reach this API")
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"])

    # --- storage -------------------------------------------------------------------------
    database_url: str = Field(default=f"sqlite:///{BACKEND_DIR / 'xray_app.db'}")
    static_dir: Path = Field(default=BACKEND_DIR / "static")
    max_upload_mb: int = 20

    # --- models --------------------------------------------------------------------------
    segmentation_model_path: Path = Field(default=REPO_ROOT / "model" / "outputs" / "checkpoints" / "best_model.h5")
    classifier_path: Path = Field(default=REPO_ROOT / "model" / "outputs" / "classifier" / "tb_classifier.joblib")
    inference_backend: str = Field(default="real", description="real | fake (tests / UI work without TensorFlow)")

    # --- LLM -----------------------------------------------------------------------------
    openai_api_key: str = Field(default="", description="Required unless llm_backend=fake")
    llm_backend: str = Field(default="openai", description="openai | fake")
    report_model: str = Field(default="gpt-4.1", description="Stage 1: structured clinical report")
    summary_model: str = Field(default="gpt-4.1-mini", description="Stage 2: clinician-facing summary")
    kb_dir: Path = Field(default=BACKEND_DIR / "documents")

    @property
    def upload_dir(self) -> Path:
        return self.static_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.static_dir / "results"


@lru_cache
def get_settings() -> Settings:
    return Settings()
