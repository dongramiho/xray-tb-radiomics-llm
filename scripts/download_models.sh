#!/usr/bin/env bash
# Fetch model artifacts that are not stored in git (U-Net weights) from the GitHub release.
#   usage: scripts/download_models.sh [tag]          (default tag: models-v1)
#   env:   MODEL_REPO=owner/name   overrides the repository
#          MODEL_DIR=path          overrides the target folder (default model/outputs/checkpoints)
#          GITHUB_TOKEN=...        needed while the repository is private (repo read scope)
# Uses `gh release download` when the gh CLI is available and logged in; otherwise the REST API via curl.
set -euo pipefail

TAG="${1:-models-v1}"
REPO="${MODEL_REPO:-dongramiho/xray-tb-radiomics-llm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/model/outputs/checkpoints}"
ASSETS=(best_model.h5)

mkdir -p "${MODEL_DIR}"
for asset in "${ASSETS[@]}"; do
  target="${MODEL_DIR}/${asset}"
  if [ -s "${target}" ]; then
    echo "exists: ${target}"
    continue
  fi

  if [ -z "${GITHUB_TOKEN:-}" ] && command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    echo "downloading ${asset} from ${REPO} release ${TAG} via gh"
    gh release download "${TAG}" --repo "${REPO}" --pattern "${asset}" --dir "${MODEL_DIR}" --clobber \
      || echo "gh download failed (active account may lack access); falling back to the GitHub API"
  fi
  if [ ! -s "${target}" ]; then
    auth=()
    if [ -n "${GITHUB_TOKEN:-}" ]; then
      auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    fi
    echo "downloading ${asset} from ${REPO} release ${TAG} via the GitHub API"
    asset_id="$(curl -sS -f -L ${auth[@]+"${auth[@]}"} -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/${REPO}/releases/tags/${TAG}" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next(a["id"] for a in d["assets"] if a["name"]==sys.argv[1]))' "${asset}")"
    curl -sS -f -L ${auth[@]+"${auth[@]}"} -H "Accept: application/octet-stream" --retry 3 \
      -o "${target}" "https://api.github.com/repos/${REPO}/releases/assets/${asset_id}"
  fi
  echo "saved: ${target} ($(du -h "${target}" | cut -f1))"
done
