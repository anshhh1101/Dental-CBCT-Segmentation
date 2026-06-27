"""
preprocessing/split.py — Reproducible train / validation / test splits.

Writes three JSON files:
  data/splits/train.json
  data/splits/val.json
  data/splits/test.json

Each file is a list of {"image": ..., "label": ...} dicts
compatible with MONAI Dataset / CacheDataset.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.helpers import get_logger, load_config

logger = get_logger("split")


def load_manifest(processed_dir: str) -> List[Dict]:
    manifest_path = Path(processed_dir) / "dataset.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"dataset.json not found in {processed_dir}. Run preprocess.py first."
        )
    with open(manifest_path) as f:
        data = json.load(f)
    return data["cases"]


def stratified_split(
    cases: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split cases into train / val / test.
    Optionally stratifies on foreground_ratio if available.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)
    shuffled = cases.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train: n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def to_monai_format(cases: List[Dict]) -> List[Dict]:
    """Convert manifest cases to MONAI-style list of dicts."""
    return [{"image": c["image"], "label": c["label"]} for c in cases]


def create_splits(cfg: Dict, seed: int = 42) -> None:
    cases = load_manifest(cfg["dataset"]["processed"])
    logger.info(f"Total cases available: {len(cases)}")

    train_cases, val_cases, test_cases = stratified_split(
        cases,
        cfg["dataset"]["train_ratio"],
        cfg["dataset"]["val_ratio"],
        cfg["dataset"]["test_ratio"],
        seed=seed,
    )

    logger.info(f"Split → train={len(train_cases)} | val={len(val_cases)} | test={len(test_cases)}")

    splits_dir = Path(cfg["dataset"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train", train_cases), ("val", val_cases), ("test", test_cases)]:
        data = to_monai_format(subset)
        out = splits_dir / f"{name}.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {name} split → {out}")

    # Summary
    summary = {
        "seed": seed,
        "total": len(cases),
        "train": len(train_cases),
        "val": len(val_cases),
        "test": len(test_cases),
    }
    with open(splits_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_split(splits_dir: str, split: str) -> List[Dict]:
    """Load a specific split JSON as a list of MONAI-style dicts."""
    path = Path(splits_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}. Run split.py first.")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    create_splits(cfg, seed=args.seed)
