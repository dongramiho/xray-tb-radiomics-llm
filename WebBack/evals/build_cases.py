"""Build the LLM evaluation cases from real images with the real pipeline.

Each case file is the exact case JSON the backend sends to the stage-1 LLM, plus the ground-truth
image label, so prompt / model changes can be evaluated on fixed inputs without TensorFlow.

    python evals/build_cases.py --images CHNCXR_0345_1 CHNCXR_0002_0 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
CASES_DIR = BACKEND_DIR / "evals" / "cases"
DEFAULT_IMAGE_DIR = BACKEND_DIR.parent / "model" / "data" / "shenzhen" / "CXR_png"

DEFAULT_IDS = [
    # TB (label 1): varied lesion burden
    "CHNCXR_0329_1",
    "CHNCXR_0345_1",
    "CHNCXR_0400_1",
    "CHNCXR_0450_1",
    "CHNCXR_0500_1",
    "CHNCXR_0550_1",
    "CHNCXR_0600_1",
    "CHNCXR_0662_1",
    # Normal (label 0): false-positive regions of different sizes, plus undetected ones
    "CHNCXR_0001_0",
    "CHNCXR_0002_0",
    "CHNCXR_0010_0",
    "CHNCXR_0050_0",
    "CHNCXR_0100_0",
    "CHNCXR_0200_0",
    "CHNCXR_0300_0",
    "CHNCXR_0326_0",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out", type=Path, default=CASES_DIR)
    parser.add_argument(
        "--degraded",
        default="CHNCXR_0345_1",
        help="Also build a QC-failing case from a 160px downscale of this image ('' to skip)",
    )
    args = parser.parse_args()

    from cxr_tb.config import DEFAULT_CLASSIFIER, DEFAULT_SEGMENTATION_MODEL
    from cxr_tb.src.pipeline import TBPipeline

    pipeline = TBPipeline.from_paths(DEFAULT_SEGMENTATION_MODEL, DEFAULT_CLASSIFIER)
    args.out.mkdir(parents=True, exist_ok=True)

    def write_case(image_id: str, source, label: int) -> None:
        result = pipeline.analyze(source, image_id=image_id)
        case = result.to_case_json()
        case.pop("timings_ms", None)
        payload = {"image_id": image_id, "label": label, "case": case}
        (args.out / f"{image_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        ml = case["ml_results"]
        qc = case.get("qc") or {}
        print(
            f"{image_id}: {ml['primary_label']} p={ml.get('tb_probability')} zones={case['segmentation']['zones']} "
            f"qc={'pass' if qc.get('passed', True) else 'FAIL'} {qc.get('flags', [])}"
        )

    for image_id in args.images:
        write_case(image_id, args.image_dir / f"{image_id}.png", int(image_id.endswith("_1")))

    if args.degraded:
        from PIL import Image

        src = Image.open(args.image_dir / f"{args.degraded}.png").convert("L")
        tiny = src.resize((160, int(160 * src.height / src.width)), Image.BILINEAR)
        write_case(f"{args.degraded}_degraded", tiny, int(args.degraded.endswith("_1")))


if __name__ == "__main__":
    main()
