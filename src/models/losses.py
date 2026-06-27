"""
models/losses.py — Segmentation loss functions.

Dental CBCT segmentation is heavily class-imbalanced (foreground structures
such as cavities or implants occupy <5% of the volume). Pure cross-entropy
would collapse to the background class. We therefore default to Dice + CE.
"""

from typing import Dict

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss

from src.utils.helpers import get_logger

logger = get_logger("loss")


def build_loss(cfg: Dict) -> nn.Module:
    """
    Build the training loss function from config.

    Options:
      dice       — Pure Dice loss (handles class imbalance, smooth gradients)
      ce         — Cross-entropy (fast but imbalance-sensitive)
      dice_ce    — Dice + CE weighted sum (DEFAULT, best overall)
      focal      — Focal loss (down-weights easy negatives)
      dice_focal — Dice + Focal (aggressive imbalance handling)
    """
    loss_name = cfg["training"].get("loss", "dice_ce").lower()
    out_channels = cfg["model"]["out_channels"]
    softmax = out_channels > 1   # use softmax for multi-class, sigmoid for binary

    if loss_name == "dice":
        loss_fn = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=softmax,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            reduction="mean",
        )

    elif loss_name == "ce":
        loss_fn = nn.CrossEntropyLoss()

    elif loss_name == "dice_ce":
        lambda_dice = cfg["training"].get("dice_ce_lambda_dice", 0.5)
        lambda_ce = cfg["training"].get("dice_ce_lambda_ce", 0.5)
        loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=softmax,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )

    elif loss_name == "focal":
        loss_fn = FocalLoss(
            include_background=False,
            to_onehot_y=True,
            gamma=2.0,
            reduction="mean",
        )

    elif loss_name == "dice_focal":
        loss_fn = DiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=softmax,
            gamma=2.0,
            lambda_dice=0.5,
            lambda_focal=0.5,
        )

    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            "Choose: dice | ce | dice_ce | focal | dice_focal"
        )

    logger.info(f"Loss function: {loss_fn.__class__.__name__}")
    return loss_fn
