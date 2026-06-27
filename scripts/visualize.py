#!/usr/bin/env python3
"""
scripts/visualize.py — Generate 3-D and 2-D visualisations for any case.

Usage:
    # Visualise a single case with prediction and GT
    python scripts/visualize.py \
        --image data/processed/images/case_001_img.nii.gz \
        --pred  outputs/predictions/case_001_pred.nii.gz \
        --gt    data/processed/labels/case_001_lbl.nii.gz \
        --case_id case_001

    # Regenerate all visualisations for the test set
    python scripts/visualize.py --test_set
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.visualize import visualize_case
from src.preprocessing.split import load_split
from src.utils.helpers import load_config, get_logger, ensure_dirs

logger = get_logger("visualize_script")


def parse_args():
    p = argparse.ArgumentParser(description="Generate CBCT visualisations")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--image", type=str, help="Path to image NIfTI")
    p.add_argument("--pred", type=str, help="Path to predicted mask NIfTI")
    p.add_argument("--gt", type=str, help="Path to ground-truth mask NIfTI")
    p.add_argument("--case_id", type=str, default="case")
    p.add_argument("--out_dir", type=str, help="Override output directory")
    p.add_argument("--test_set", action="store_true",
                   help="Visualise all test-set predictions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = args.out_dir or cfg["paths"]["visualizations"]
    ensure_dirs(out_dir)

    if args.test_set:
        test_data = load_split(cfg["dataset"]["splits_dir"], "test")
        pred_dir = Path(cfg["paths"]["predictions"])
        results_path = pred_dir / "test_results.json"

        per_case_metrics = {}
        if results_path.exists():
            with open(results_path) as f:
                per_case_metrics = json.load(f).get("per_case", {})

        for item in test_data:
            case_id = Path(item["image"]).stem.replace("_img", "")
            pred_path = str(pred_dir / f"{case_id}_pred.nii.gz")
            metrics = per_case_metrics.get(case_id)

            visualize_case(
                image_path=item["image"],
                pred_path=pred_path if Path(pred_path).exists() else None,
                gt_path=item.get("label"),
                out_dir=out_dir,
                case_id=case_id,
                metrics=metrics,
            )
    else:
        if not args.image:
            logger.error("Provide --image or --test_set")
            sys.exit(1)

        visualize_case(
            image_path=args.image,
            pred_path=args.pred,
            gt_path=args.gt,
            out_dir=out_dir,
            case_id=args.case_id,
        )

    logger.info(f"✅ Visualisations saved to {out_dir}")
