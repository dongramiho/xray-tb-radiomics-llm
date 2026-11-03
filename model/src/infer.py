"""Run the full pipeline on a single chest X-ray and print the case JSON.

Example:
    python -m cxr_tb.src.infer --image path/to/cxr.png --output_dir outputs/predictions/single
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cxr_tb.config import DEFAULT_CLASSIFIER, DEFAULT_SEGMENTATION_MODEL, PREDICTION_DIR
from cxr_tb.src.pipeline import TBPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment, extract radiomics, and classify one X-ray.")
    parser.add_argument("--image", type=Path, required=True, help="Input X-ray (PNG/JPEG).")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument("--classifier_path", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--output_dir", type=Path, default=PREDICTION_DIR / "single")
    parser.add_argument("--no_radiomics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TBPipeline.from_paths(args.model_path, args.classifier_path, with_radiomics=not args.no_radiomics)
    result = pipeline.analyze(args.image, image_id=args.image.stem)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / f"{result.image_id}_mask.png"
    overlay_path = args.output_dir / f"{result.image_id}_overlay.png"
    json_path = args.output_dir / f"{result.image_id}_case.json"
    result.mask_image().save(mask_path)
    result.overlay_image().save(overlay_path)
    case = result.to_case_json()
    json_path.write_text(json.dumps(case, indent=2, ensure_ascii=False))

    brief = {
        "segmentation": {k: v for k, v in case["segmentation"].items() if k != "components"},
        "ml_results": case["ml_results"],
        "timings_ms": case["timings_ms"],
    }
    print(json.dumps(brief, indent=2, ensure_ascii=False))
    print(f"Saved {mask_path}, {overlay_path}, {json_path}")


if __name__ == "__main__":
    main()
