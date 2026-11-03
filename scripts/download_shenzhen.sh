#!/usr/bin/env bash
# Download the public Shenzhen Hospital chest X-ray set (NIH/NLM) into model/data/shenzhen.
#   662 PNGs (326 normal: CHNCXR_0001_0 .. CHNCXR_0326_0, 336 TB: CHNCXR_0327_1 .. CHNCXR_0662_1)
#   + clinical readings (sex, age, finding) as text files.
# Usage: scripts/download_shenzhen.sh [dest_dir] [parallelism]
set -euo pipefail

BASE_URL="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set"
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/model/data/shenzhen}"
PARALLEL="${2:-12}"

download_one() {
  local id="$1"
  curl -sS -f -L --retry 3 -m 180 -o "CXR_png/${id}.png" "${BASE_URL}/CXR_png/${id}.png" || echo "FAIL png ${id}"
  curl -sS -f -L --retry 3 -m 60 -o "ClinicalReadings/${id}.txt" "${BASE_URL}/ClinicalReadings/${id}.txt" || echo "FAIL txt ${id}"
}
export -f download_one
export BASE_URL

mkdir -p "${DEST}/CXR_png" "${DEST}/ClinicalReadings"
cd "${DEST}"
{
  for i in $(seq 1 326); do printf 'CHNCXR_%04d_0\n' "$i"; done
  for i in $(seq 327 662); do printf 'CHNCXR_%04d_1\n' "$i"; done
} | xargs -P "${PARALLEL}" -n 1 bash -c 'download_one "$0"'

echo "images: $(ls CXR_png | wc -l | tr -d ' ')  readings: $(ls ClinicalReadings | wc -l | tr -d ' ')  size: $(du -sh CXR_png | cut -f1)"
