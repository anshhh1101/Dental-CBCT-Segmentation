#!/usr/bin/env python3
"""
scripts/preprocess.py — Run the full data preparation pipeline.

Steps:
  1. (Optional) Download dataset
  2. Preprocess all volumes
  3. Create train/val/test splits

Usage:
    # Download + preprocess ToothFairy2
    python scripts/preprocess.py --download toothfairy2

    # Preprocess a local dataset already in data/raw/
    python scripts/preprocess.py

    # Just create splits from already-preprocessed data
    python scripts/preprocess.py --splits_only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, get_logger
from src.preprocessing.download import download_dataset, list_datasets
from src.preprocessing.preprocess import DentalPreprocessor
from src.preprocessing.split import create_splits

logger = get_logger("preprocess_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Data preparation pipeline")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--download", type=str, metavar="DATASET",
                   help="Download a dataset (toothfairy2 | mmdental | tufts)")
    p.add_argument("--list_datasets", action="store_true",
                   help="List available datasets and exit")
    p.add_argument("--splits_only", action="store_true",
                   help="Skip preprocessing; only (re-)create splits")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_datasets:
        list_datasets()
        sys.exit(0)

    cfg = load_config(args.config)

    if args.download:
        logger.info(f"Downloading dataset: {args.download}")
        download_dataset(args.download, raw_dir=cfg["dataset"]["root"])

    if not args.splits_only:
        logger.info("Starting preprocessing...")
        preprocessor = DentalPreprocessor(cfg)
        preprocessor.run()

    logger.info("Creating train/val/test splits...")
    create_splits(cfg, seed=args.seed)
    logger.info("✅ Data preparation complete.")
