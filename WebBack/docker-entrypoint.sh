#!/usr/bin/env bash
# 1) fetch U-Net weights if the volume is empty, 2) apply DB migrations, 3) start the app.
set -euo pipefail

WEIGHTS="${SEGMENTATION_MODEL_PATH:-/app/model/outputs/checkpoints/best_model.h5}"
if [ ! -s "${WEIGHTS}" ] && [ "${INFERENCE_BACKEND:-real}" = "real" ]; then
  echo "[entrypoint] weights missing at ${WEIGHTS}; downloading from GitHub release ${MODEL_RELEASE_TAG:-models-v1}"
  MODEL_DIR="$(dirname "${WEIGHTS}")" /app/scripts/download_models.sh "${MODEL_RELEASE_TAG:-models-v1}"
fi

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "[entrypoint] alembic upgrade head"
  alembic upgrade head
fi

exec "$@"
