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

    # --- auth ----------------------------------------------------------------------------
    jwt_secret: str = Field(default="dev-only-change-me", description="HS256 signing key; set a long random value")
    access_token_expire_minutes: int = 60 * 12
    seed_doctor_email: str = Field(default="doctor@test.com", description="Operator-provisioned clinician login")
    seed_doctor_password: str = Field(default="changeme")
    seed_doctor_name: str = Field(default="Dr. Kim")

    # --- request handling ----------------------------------------------------------------
    rate_limit_analyze: str = Field(default="30/minute", description="slowapi limit string for POST /analyze")
    rate_limit_enabled: bool = True
    log_json: bool = Field(default=False, description="Emit one JSON object per log line (container friendly)")

    # --- storage -------------------------------------------------------------------------
    database_url: str = Field(default=f"sqlite:///{BACKEND_DIR / 'xray_app.db'}")
    auto_migrate: bool = Field(
        default=True, description="Apply Alembic migrations at startup (disable when the entrypoint does it)"
    )
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
    report_fallback_model: str = Field(
        default="gpt-4.1-mini", description="Used when the report model fails after retries"
    )
    summary_fallback_model: str = Field(
        default="gpt-4.1", description="Used when the summary model fails after retries"
    )
    llm_timeout_s: float = Field(default=60.0, description="Per-request timeout for OpenAI calls")
    llm_max_retries: int = Field(default=2, description="Retries on rate limits / transient errors before fallback")
    embedding_model: str = Field(default="text-embedding-3-small")
    rag_top_k: int = Field(default=6, description="Knowledge-base chunks injected into the report prompt")

    def validate_for_environment(self) -> None:
        """Refuse obviously unsafe production configuration early (at startup, not at first request)."""
        if self.environment != "production":
            return
        problems = []
        if self.jwt_secret == "dev-only-change-me" or len(self.jwt_secret) < 32:
            problems.append("JWT_SECRET must be a random string of at least 32 characters")
        if "*" in self.cors_origins:
            problems.append("CORS_ORIGINS must not contain '*'")
        if self.llm_backend == "fake" or self.inference_backend == "fake":
            problems.append("fake backends are not allowed in production")
        if problems:
            raise RuntimeError("Unsafe production settings: " + "; ".join(problems))

    @property
    def upload_dir(self) -> Path:
        return self.static_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.static_dir / "results"


@lru_cache
def get_settings() -> Settings:
    return Settings()
