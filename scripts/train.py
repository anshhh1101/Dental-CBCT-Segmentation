#!/usr/bin/env python3
"""
scripts/train.py — Entry point for the dental segmentation training pipeline.

Usage:
    python scripts/train.py --config config.yaml
    python scripts/train.py --config config.yaml --arch segresnet --epochs 100
    python scripts/train.py --config config.yaml --resume outputs/checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, set_seed, get_device, ensure_dirs, get_logger
from src.preprocessing.split import load_split, create_splits
from src.preprocessing.dataloader import build_dataloaders
from src.models.architectures import build_model, model_summary
from src.models.losses import build_loss
from src.training.trainer import Trainer

logger = get_logger("train", log_dir="outputs/logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Dental CBCT Segmentation — Training")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--arch", type=str, help="Override model architecture")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.arch:
        cfg["model"]["architecture"] = args.arch
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    set_seed(args.seed)
    cfg["project"]["seed"] = args.seed

    ensure_dirs(
        cfg["paths"]["checkpoints"],
        cfg["paths"]["predictions"],
        cfg["paths"]["visualizations"],
        cfg["paths"].get("logs", "outputs/logs"),
    )

    # ── Load splits ──────────────────────────────────────────
    splits_dir = cfg["dataset"]["splits_dir"]
    try:
        train_data = load_split(splits_dir, "train")
        val_data = load_split(splits_dir, "val")
        test_data = load_split(splits_dir, "test")
    except FileNotFoundError:
        logger.info("Splits not found — running split creation...")
        create_splits(cfg, seed=args.seed)
        train_data = load_split(splits_dir, "train")
        val_data = load_split(splits_dir, "val")
        test_data = load_split(splits_dir, "test")

    logger.info(f"Data | train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    # ── DataLoaders ───────────────────────────────────────────
    loaders = build_dataloaders(train_data, val_data, test_data, cfg)

    # ── Model ─────────────────────────────────────────────────
    model = build_model(cfg)
    model_summary(model)

    # ── Loss ──────────────────────────────────────────────────
    loss_fn = build_loss(cfg)

    # ── Trainer ───────────────────────────────────────────────
    trainer = Trainer(model, loss_fn, cfg, loaders)

    if args.resume:
        import torch
        ckpt = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, dice={ckpt['val_dice']:.4f})")

    # ── Train ─────────────────────────────────────────────────
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
