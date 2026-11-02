#!/usr/bin/env bash
# Fetch model artifacts that are not stored in git (U-Net weights) from the GitHub release.
#   usage: scripts/download_models.sh [tag]          (default tag: models-v1)
#   env:   MODEL_REPO=owner/name  overrides the repository, MODEL_DIR overrides the target folder
set -euo pipefail

TAG="${1:-models-v1}"
REPO="${MODEL_REPO:-dongramiho/xray-tb-radiomics-llm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/model/outputs/checkpoints}"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"

mkdir -p "${MODEL_DIR}"
for asset in best_model.h5; do
  target="${MODEL_DIR}/${asset}"
  if [ -s "${target}" ]; then
    echo "exists: ${target}"
    continue
  fi
  echo "downloading ${BASE_URL}/${asset}"
  curl -sS -f -L --retry 3 -o "${target}" "${BASE_URL}/${asset}"
  echo "saved: ${target} ($(du -h "${target}" | cut -f1))"
done
