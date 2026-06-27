"""
utils/helpers.py — Shared utility functions for the dental segmentation pipeline.
"""

import os
import random
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import yaml
import torch


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:           # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return as nested dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def flatten_config(cfg: Dict, prefix: str = "", sep: str = ".") -> Dict:
    """Flatten nested config dict for logging."""
    out = {}
    for k, v in cfg.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_config(v, full_key, sep))
        else:
            out[full_key] = v
    return out


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    """Resolve device from preference string."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ─────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────

def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice similarity coefficient for binary arrays."""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = (pred & target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection-over-Union for binary arrays."""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return (intersection + smooth) / (union + smooth)


def hausdorff_95(pred: np.ndarray, target: np.ndarray, spacing: Tuple = (1, 1, 1)) -> float:
    """
    Approximate 95th-percentile Hausdorff distance using scipy.
    Falls back to NaN if either mask is empty.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        if not pred.any() or not target.any():
            return float("nan")
        dt_pred = distance_transform_edt(~pred.astype(bool), sampling=spacing)
        dt_target = distance_transform_edt(~target.astype(bool), sampling=spacing)
        d1 = dt_target[pred.astype(bool)]
        d2 = dt_pred[target.astype(bool)]
        return float(np.percentile(np.concatenate([d1, d2]), 95))
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────

class Timer:
    """Simple context-manager timer."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start

    def __str__(self):
        return f"{self.elapsed:.2f}s"


# ─────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def find_files(root: str, extensions: Tuple[str, ...] = (".nii", ".nii.gz", ".mha", ".nrrd")) -> list:
    """Recursively find medical image files under root."""
    found = []
    for ext in extensions:
        found.extend(sorted(Path(root).rglob(f"*{ext}")))
    return found
