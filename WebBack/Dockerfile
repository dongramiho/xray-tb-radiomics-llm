# syntax=docker/dockerfile:1.7
# Backend image: FastAPI + cxr_tb pipeline (TensorFlow CPU, PyRadiomics). Build context = repository root.
#   docker build -f WebBack/Dockerfile -t xray-tb-backend .

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends curl git build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Model package first (cached unless model/ changes); pyradiomics compiles its C extensions here.
COPY model/pyproject.toml model/__init__.py model/config.py /app/model/
COPY model/src /app/model/src
RUN pip install --upgrade pip && pip install "numpy<2" \
    && pip install --no-build-isolation -e "/app/model[inference]"

COPY WebBack/requirements.txt /app/WebBack/requirements.txt
RUN grep -vE '^-e' /app/WebBack/requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY WebBack /app/WebBack
COPY model/outputs/classifier /app/model/outputs/classifier
COPY scripts/download_models.sh /app/scripts/download_models.sh

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/WebBack/static /app/model/outputs/checkpoints \
    && chown -R app:app /app
USER app
WORKDIR /app/WebBack
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["/app/WebBack/docker-entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
