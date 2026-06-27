"""
postprocessing/postprocess.py — Refine raw model predictions.

Raw softmax outputs contain noise, spurious blobs, and rough boundaries.
Post-processing improves both quantitative metrics and clinical usability.

Steps applied (configurable):
  1. Threshold softmax → binary mask
  2. Remove small connected components (noise suppression)
  3. Morphological closing (fill narrow gaps)
  4. Hole filling
  5. Optional Gaussian smoothing before thresholding
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    label as ndlabel,
)

from src.utils.helpers import get_logger, dice_score, iou_score, hausdorff_95

logger = get_logger("postprocess")


# ─────────────────────────────────────────────
# Core ops
# ─────────────────────────────────────────────

def threshold(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probability map to binary mask."""
    return (prob_map >= threshold).astype(np.uint8)


def remove_small_objects(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Remove connected components smaller than min_size voxels.
    Keeps only the N largest components; small blobs are noise.
    """
    labeled, n_comps = ndlabel(mask)
    if n_comps == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    too_small = sizes < min_size
    remove_mask = too_small[labeled]
    cleaned = mask.copy()
    cleaned[remove_mask] = 0
    n_removed = remove_mask.sum()
    if n_removed:
        logger.debug(f"Removed {int((mask - cleaned).sum())} voxels in {too_small.sum() - 1} small components")
    return cleaned


def morphological_close(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Fill narrow gaps with binary closing (dilation then erosion)."""
    struct = np.ones((3, 3, 3), dtype=bool)
    return binary_closing(mask.astype(bool), structure=struct, iterations=iterations).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes slice-by-slice (faster than 3-D hole fill for large volumes)."""
    filled = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        filled[z] = binary_fill_holes(mask[z]).astype(np.uint8)
    return filled


def smooth_probability(prob_map: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Gaussian-smooth probability map before thresholding to reduce salt-and-pepper noise."""
    return gaussian_filter(prob_map.astype(np.float32), sigma=sigma)


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def postprocess_prediction(
    prob_map: np.ndarray,
    cfg: Dict,
    ref_image: Optional[sitk.Image] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the full post-processing chain.

    Args:
        prob_map:   Foreground probability map, shape (D, H, W), values [0, 1].
        cfg:        Config dict.
        ref_image:  Optional SimpleITK reference for metadata transfer.

    Returns:
        (raw_mask, refined_mask) both as uint8 numpy arrays.
    """
    pp = cfg.get("postprocessing", {})
    threshold_val = pp.get("confidence_threshold", 0.5)

    # Optional smoothing before threshold
    if pp.get("smooth_predictions", True):
        prob_map = smooth_probability(prob_map, sigma=0.5)

    raw_mask = threshold(prob_map, threshold_val)
    refined = raw_mask.copy()

    if pp.get("remove_small_objects", True):
        min_size = pp.get("min_object_size_voxels", 50)
        refined = remove_small_objects(refined, min_size)

    if pp.get("closing_iterations", 2) > 0:
        refined = morphological_close(refined, pp["closing_iterations"])

    if pp.get("fill_holes", True):
        refined = fill_holes(refined)

    return raw_mask, refined


def save_prediction_nifti(
    mask: np.ndarray,
    ref_image: sitk.Image,
    out_path: str,
) -> None:
    """Save a binary mask as NIfTI with the reference image's geometry."""
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(ref_image)
    sitk.WriteImage(out, str(out_path))
    logger.debug(f"Saved prediction → {out_path}")


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_case(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
) -> Dict[str, float]:
    """Return a dict of segmentation metrics for a single case."""
    return {
        "dice": round(dice_score(pred, gt), 4),
        "iou": round(iou_score(pred, gt), 4),
        "hd95": round(hausdorff_95(pred, gt, spacing=spacing), 2),
        "pred_volume_ml": round(float(pred.sum()) * np.prod(spacing) / 1000, 3),
        "gt_volume_ml": round(float(gt.sum()) * np.prod(spacing) / 1000, 3),
    }


def evaluate_dataset(
    predictions: List[Tuple[str, np.ndarray, np.ndarray]],
    spacing: Tuple = (0.4, 0.4, 0.4),
) -> Dict:
    """
    Aggregate metrics across the test set.

    Args:
        predictions: List of (case_id, pred_mask, gt_mask) tuples.

    Returns:
        Dict with per-case and aggregate metrics.
    """
    import json

    per_case = {}
    dices, ious, hds = [], [], []

    for case_id, pred, gt in predictions:
        m = evaluate_case(pred, gt, spacing)
        per_case[case_id] = m
        dices.append(m["dice"])
        ious.append(m["iou"])
        if not np.isnan(m["hd95"]):
            hds.append(m["hd95"])

    summary = {
        "n_cases": len(predictions),
        "mean_dice": round(float(np.mean(dices)), 4),
        "std_dice": round(float(np.std(dices)), 4),
        "median_dice": round(float(np.median(dices)), 4),
        "mean_iou": round(float(np.mean(ious)), 4),
        "mean_hd95": round(float(np.mean(hds)) if hds else float("nan"), 2),
        "per_case": per_case,
    }

    logger.info(
        f"Test set results | "
        f"Dice={summary['mean_dice']:.4f}±{summary['std_dice']:.4f} | "
        f"IoU={summary['mean_iou']:.4f} | "
        f"HD95={summary['mean_hd95']:.2f}mm"
    )
    return summary
