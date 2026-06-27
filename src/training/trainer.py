"""
training/trainer.py — Main training loop for dental segmentation.

Features:
  - Mixed-precision training (AMP) for faster GPU utilisation
  - Sliding-window inference for full-volume validation
  - Dice metric tracking with best-checkpoint saving
  - Cosine / StepLR / ReduceLROnPlateau schedulers
  - Early stopping
  - TensorBoard logging
  - CSV metrics export
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter

from src.utils.helpers import get_logger, dice_score, iou_score, hausdorff_95

logger = get_logger("trainer")


class Trainer:
    """
    End-to-end training manager.

    Usage:
        trainer = Trainer(model, cfg, loaders)
        trainer.train()
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, cfg: Dict, loaders: Dict):
        self.cfg = cfg
        self.loaders = loaders
        self.device = self._resolve_device()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        t_cfg = cfg["training"]
        self.epochs = t_cfg["epochs"]
        self.val_interval = t_cfg.get("val_interval", 5)
        self.amp = t_cfg.get("amp", True) and self.device.type == "cuda"
        self.grad_clip = t_cfg.get("gradient_clip", 1.0)
        self.patience = t_cfg.get("early_stopping_patience", 30)
        self.patch_size = tuple(cfg["dataset"]["patch_size"])

        self.ckpt_dir = Path(cfg["paths"]["checkpoints"])
        self.log_dir = Path(cfg["paths"].get("logs", "outputs/logs"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Optimiser
        self.optimizer = AdamW(
            model.parameters(),
            lr=t_cfg["learning_rate"],
            weight_decay=t_cfg.get("weight_decay", 1e-5),
        )

        # Scheduler
        self.scheduler = self._build_scheduler(t_cfg)

        # AMP scaler
        self.scaler = GradScaler() if self.amp else None

        # MONAI metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.post_pred = AsDiscrete(argmax=True, to_onehot=cfg["model"]["out_channels"])
        self.post_label = AsDiscrete(to_onehot=cfg["model"]["out_channels"])

        # Tracking
        self.best_val_dice = 0.0
        self.epochs_no_improve = 0
        self.history = []

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        logger.info(f"TensorBoard logs → {self.log_dir / 'tensorboard'}")

    def _resolve_device(self) -> torch.device:
        pref = self.cfg.get("project", {}).get("device", "auto")
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(pref)

    def _build_scheduler(self, t_cfg: Dict):
        sched = t_cfg.get("scheduler", "cosine").lower()
        if sched == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-7)
        elif sched == "step":
            return StepLR(self.optimizer, step_size=50, gamma=0.5)
        elif sched == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="max", patience=10, factor=0.5)
        else:
            raise ValueError(f"Unknown scheduler: {sched}")

    # ─────────────────────────────────────────
    # One epoch of training
    # ─────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in self.loaders["train"]:
            # Patch sampling returns (batch_size * num_patches, C, D, H, W) for some MONAI versions
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            if self.amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        return avg_loss

    # ─────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────

    def validate(self, epoch: int) -> float:
        self.model.eval()
        self.dice_metric.reset()

        with torch.no_grad():
            for batch in self.loaders["val"]:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Sliding-window inference handles large volumes
                preds = sliding_window_inference(
                    inputs=images,
                    roi_size=self.patch_size,
                    sw_batch_size=4,
                    predictor=self.model,
                    overlap=0.25,
                )

                preds_disc = [self.post_pred(i) for i in preds]
                labels_disc = [self.post_label(i) for i in labels]

                self.dice_metric(y_pred=preds_disc, y=labels_disc)

        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        self.writer.add_scalar("Dice/val", mean_dice, epoch)
        return mean_dice

    # ─────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────

    def train(self) -> None:
        logger.info(f"Training on {self.device} | AMP={self.amp}")
        logger.info(f"Epochs={self.epochs} | Val every {self.val_interval} epochs")

        csv_path = self.log_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "lr", "elapsed_s"])

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LR", lr, epoch)

            val_dice = 0.0
            if epoch % self.val_interval == 0 or epoch == self.epochs:
                val_dice = self.validate(epoch)
                self._maybe_save(val_dice, epoch)
                self._maybe_stop()
                logger.info(
                    f"Epoch {epoch:>4}/{self.epochs} | "
                    f"loss={train_loss:.4f} | val_dice={val_dice:.4f} | "
                    f"lr={lr:.2e} | {elapsed:.1f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch:>4}/{self.epochs} | loss={train_loss:.4f} | {elapsed:.1f}s"
                )

            # Step scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_dice)
            else:
                self.scheduler.step()

            # Log to CSV
            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_dice": round(val_dice, 6),
                "lr": lr,
                "elapsed_s": round(elapsed, 2),
            })
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_dice, lr, elapsed])

            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        self.writer.close()
        self._save_history()
        logger.info(f"Training complete. Best val Dice = {self.best_val_dice:.4f}")

    # ─────────────────────────────────────────
    # Checkpoint management
    # ─────────────────────────────────────────

    def _maybe_save(self, val_dice: float, epoch: int) -> None:
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            self.epochs_no_improve = 0
            path = self.ckpt_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_dice": val_dice,
                    "cfg": self.cfg,
                },
                path,
            )
            logger.info(f"  ✓ Best model saved (dice={val_dice:.4f}) → {path}")
        else:
            self.epochs_no_improve += 1

    def _maybe_stop(self) -> None:
        pass  # check happens in main loop

    def load_best(self) -> None:
        path = self.ckpt_dir / "best_model.pth"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model (epoch={ckpt['epoch']}, dice={ckpt['val_dice']:.4f})")

    def _save_history(self) -> None:
        path = self.log_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history → {path}")
