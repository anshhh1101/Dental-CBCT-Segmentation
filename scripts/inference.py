#!/usr/bin/env python3
"""
scripts/inference.py — Run inference on a single volume or a full test set.

Usage:
    # Single file
    python scripts/inference.py --input data/raw/patient_001.nii.gz \
                                --output outputs/predictions/patient_001_pred.nii.gz \
                                --checkpoint outputs/checkpoints/best_model.pth

    # Full test set (from splits)
    python scripts/inference.py --test_set \
                                --checkpoint outputs/checkpoints/best_model.pth \
                                --visualize

    # Evaluate (requires ground-truth labels)
    python scripts/inference.py --test_set --evaluate --visualize
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from src.models.architectures import build_model
from src.preprocessing.dataloader import get_val_transforms
from src.preprocessing.preprocess import clip_and_normalize, resample_volume
from src.postprocessing.postprocess import (
    postprocess_prediction,
    save_prediction_nifti,
    evaluate_case,
    evaluate_dataset,
)
from src.visualization.visualize import visualize_case
from src.utils.helpers import load_config, get_logger, ensure_dirs

logger = get_logger("inference")


# ─────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, cfg: Dict) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    dice = ckpt.get("val_dice", "?")
    logger.info(f"Loaded checkpoint: epoch={epoch}, best_val_dice={dice}")
    return model, device


# ─────────────────────────────────────────────
# Single-volume inference
# ─────────────────────────────────────────────

def infer_volume(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    cfg: Dict,
) -> np.ndarray:
    """
    Run sliding-window inference on a single CBCT volume.
    Returns the foreground probability map as a numpy array.
    """
    # Load + preprocess on the fly (if not already preprocessed)
    img_sitk = sitk.ReadImage(str(image_path))
    img_sitk = resample_volume(img_sitk, cfg["preprocessing"]["target_spacing"])
    img_sitk = clip_and_normalize(img_sitk, cfg["preprocessing"]["clip_hu"])
    img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

    # Add batch + channel dims: (1, 1, D, H, W)
    tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(device)

    patch_size = tuple(cfg["dataset"]["patch_size"])

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=patch_size,
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian",     # weighted averaging at patch borders
        )

    # Softmax → foreground probability (channel 1)
    probs = torch.softmax(logits, dim=1)
    prob_fg = probs[0, 1].cpu().numpy()  # shape (D, H, W)
    return prob_fg, img_sitk


def run_single(args, cfg):
    model, device = load_checkpoint(args.checkpoint, cfg)
    prob_fg, ref_img = infer_volume(args.input, model, device, cfg)
    raw_mask, refined_mask = postprocess_prediction(prob_fg, cfg)

    out_path = args.output or str(Path(args.input).stem + "_pred.nii.gz")
    save_prediction_nifti(refined_mask, ref_img, out_path)
    logger.info(f"Prediction saved → {out_path}")

    if args.visualize:
        case_id = Path(args.input).stem
        visualize_case(
            image_path=args.input,
            pred_path=out_path,
            gt_path=None,
            out_dir=cfg["paths"]["visualizations"],
            case_id=case_id,
        )


# ─────────────────────────────────────────────
# Full test-set inference
# ─────────────────────────────────────────────

def run_test_set(args, cfg):
    from src.preprocessing.split import load_split

    test_data = load_split(cfg["dataset"]["splits_dir"], "test")
    logger.info(f"Running inference on {len(test_data)} test cases.")

    model, device = load_checkpoint(args.checkpoint, cfg)
    pred_dir = Path(cfg["paths"]["predictions"])
    ensure_dirs(str(pred_dir))

    predictions = []

    for item in test_data:
        case_id = Path(item["image"]).stem.replace("_img", "")
        logger.info(f"Processing {case_id}...")

        prob_fg, ref_img = infer_volume(item["image"], model, device, cfg)
        raw_mask, refined_mask = postprocess_prediction(prob_fg, cfg)

        pred_path = str(pred_dir / f"{case_id}_pred.nii.gz")
        save_prediction_nifti(refined_mask, ref_img, pred_path)

        metrics = None
        if args.evaluate and "label" in item and Path(item["label"]).exists():
            gt_sitk = sitk.ReadImage(item["label"])
            gt_arr = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)
            # Align shapes if needed (may differ due to resampling)
            if gt_arr.shape != refined_mask.shape:
                logger.warning(
                    f"Shape mismatch for {case_id}: pred={refined_mask.shape}, gt={gt_arr.shape}. "
                    "Skipping metric computation for this case."
                )
            else:
                metrics = evaluate_case(refined_mask, gt_arr, spacing=cfg["preprocessing"]["target_spacing"])
                logger.info(f"  {case_id} | {metrics}")
                predictions.append((case_id, refined_mask, gt_arr))

        if args.visualize:
            visualize_case(
                image_path=item["image"],
                pred_path=pred_path,
                gt_path=item.get("label"),
                out_dir=cfg["paths"]["visualizations"],
                case_id=case_id,
                metrics=metrics,
            )

    # Aggregate evaluation
    if args.evaluate and predictions:
        summary = evaluate_dataset(predictions, spacing=cfg["preprocessing"]["target_spacing"])
        results_path = Path(cfg["paths"]["predictions"]) / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nTest set summary:\n{json.dumps({k: v for k, v in summary.items() if k != 'per_case'}, indent=2)}")
        logger.info(f"Full results → {results_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dental CBCT Segmentation — Inference")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    p.add_argument("--input", type=str, help="Single input .nii.gz file")
    p.add_argument("--output", type=str, help="Output path for single prediction")
    p.add_argument("--test_set", action="store_true", help="Run on full test split")
    p.add_argument("--evaluate", action="store_true", help="Compute metrics against GT labels")
    p.add_argument("--visualize", action="store_true", help="Generate visualisations")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    ensure_dirs(
        cfg["paths"]["predictions"],
        cfg["paths"]["visualizations"],
    )

    if args.test_set:
        run_test_set(args, cfg)
    elif args.input:
        run_single(args, cfg)
    else:
        logger.error("Provide either --input <file> or --test_set")
        sys.exit(1)
