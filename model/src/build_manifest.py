"""Build ``data/dataset.csv`` from a Shenzhen-style image folder.

Columns: image_id, image_path, label (1 = TB, 0 = normal), mask_path (optional, when a mask directory
is given), sex, age, finding (parsed from the ClinicalReadings text files when present).

Example:
    python -m cxr_tb.src.build_manifest --images data/shenzhen/CXR_png \
        --readings data/shenzhen/ClinicalReadings --masks data/shenzhen_combined_masks \
        --output data/dataset.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from cxr_tb.config import CSV_PATH, DATA_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the dataset manifest CSV.")
    parser.add_argument("--images", type=Path, default=DATA_DIR / "shenzhen" / "CXR_png")
    parser.add_argument("--readings", type=Path, default=DATA_DIR / "shenzhen" / "ClinicalReadings")
    parser.add_argument("--masks", type=Path, default=None, help="Optional directory with <image_id>.png lesion masks.")
    parser.add_argument("--output", type=Path, default=CSV_PATH)
    return parser.parse_args()


def label_from_id(image_id: str) -> int | None:
    """Shenzhen naming: CHNCXR_<index>_<label> where label 1 = TB, 0 = normal."""
    match = re.search(r"_(\d)$", image_id)
    return int(match.group(1)) if match else None


def parse_reading(path: Path) -> dict[str, str | None]:
    """Parse 'Male , 32yrs\\nsecondary PTB in the right upper field' style readings."""
    info: dict[str, str | None] = {"sex": None, "age": None, "finding": None}
    if not path.exists():
        return info
    lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    if not lines:
        return info
    head = lines[0]
    sex_match = re.search(r"(male|female)", head, flags=re.I)
    age_match = re.search(r"(\d+)\s*yr", head, flags=re.I)
    info["sex"] = sex_match.group(1).lower() if sex_match else None
    info["age"] = age_match.group(1) if age_match else None
    info["finding"] = " ".join(lines[1:]) if len(lines) > 1 else None
    return info


def build_rows(images: Path, readings: Path | None, masks: Path | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_path in sorted(images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_id = image_path.stem
        row: dict[str, object] = {
            "image_id": image_id,
            "image_path": str(image_path.resolve()),
            "label": label_from_id(image_id),
        }
        if masks is not None:
            mask_path = masks / f"{image_id}.png"
            row["mask_path"] = str(mask_path.resolve()) if mask_path.exists() else ""
        if readings is not None:
            row.update(parse_reading(readings / f"{image_id}.txt"))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if not args.images.exists():
        raise FileNotFoundError(f"Image directory not found: {args.images}")
    rows = build_rows(args.images, args.readings if args.readings.exists() else None, args.masks)
    if not rows:
        raise SystemExit("No images found.")
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    counts = df["label"].value_counts(dropna=False).to_dict()
    print(f"Wrote {len(df)} rows to {args.output} (labels: {counts})")


if __name__ == "__main__":
    main()
