"""
models/architectures.py — Segmentation model factory.

Supported architectures:
  - SegResNet    : NVIDIA's residual encoder-decoder (default, fast)
  - UNet3D       : Classic 3-D U-Net with skip connections
  - SwinUNETR    : Transformer-based segmentation (heavy, best accuracy)

All models use MONAI implementations for reliability and GPU efficiency.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR, UNet

from src.utils.helpers import get_logger

logger = get_logger("model")


# ─────────────────────────────────────────────
# SegResNet (recommended default)
# ─────────────────────────────────────────────

def build_segresnet(cfg: Dict) -> nn.Module:
    """
    SegResNet — Encoder-decoder with residual blocks.

    Reference: Myronenko (2019), "3D MRI brain tumor segmentation using
    autoencoder regularization", MICCAI BrainLes Workshop.

    Advantages for dental CBCT:
      - Lightweight (~4M params with init_filters=16)
      - Native 3-D operations
      - Fast convergence
    """
    model_cfg = cfg["model"]
    model = SegResNet(
        spatial_dims=3,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        init_filters=model_cfg.get("init_filters", 16),
        dropout_prob=model_cfg.get("dropout_prob", 0.2),
        act="relu",
        norm="batch",
        use_conv_final=True,
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"SegResNet built | params={params:.2f}M")
    return model


# ─────────────────────────────────────────────
# 3D U-Net (interpretable baseline)
# ─────────────────────────────────────────────

def build_unet3d(cfg: Dict) -> nn.Module:
    """
    Classic 3-D U-Net (Çiçek et al. 2016).

    Advantages:
      - Skip connections preserve spatial detail
      - Well-understood architecture for medical segmentation
    """
    model_cfg = cfg["model"]
    channels = tuple(model_cfg.get("channels", [16, 32, 64, 128, 256]))
    strides = tuple(model_cfg.get("strides", [2, 2, 2, 2]))

    model = UNet(
        spatial_dims=3,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=channels,
        strides=strides,
        num_res_units=2,
        norm="batch",
        act="leakyrelu",
        dropout=model_cfg.get("dropout_prob", 0.1),
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"UNet3D built | channels={channels} | params={params:.2f}M")
    return model


# ─────────────────────────────────────────────
# SwinUNETR (transformer-based, highest accuracy)
# ─────────────────────────────────────────────

def build_swinunetr(cfg: Dict, img_size: Tuple[int, int, int] = (64, 64, 64)) -> nn.Module:
    """
    Swin Transformer U-shaped network.

    Reference: Tang et al. (2022), "Self-supervised pre-training of
    swin transformers for 3D medical image analysis", CVPR.

    Best suited for fine-grained structures (cavities, small lesions).
    Requires more GPU memory (~24 GB for 96³ patches).
    """
    model_cfg = cfg["model"]
    model = SwinUNETR(
        img_size=img_size,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        feature_size=48,
        use_checkpoint=True,   # gradient checkpointing to save memory
        spatial_dims=3,
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"SwinUNETR built | feature_size=48 | params={params:.2f}M")
    return model


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_model(cfg: Dict) -> nn.Module:
    """Return the model specified in cfg['model']['architecture']."""
    arch = cfg["model"]["architecture"].lower()
    patch_size = tuple(cfg["dataset"]["patch_size"])

    if arch == "segresnet":
        return build_segresnet(cfg)
    elif arch == "unet3d":
        return build_unet3d(cfg)
    elif arch == "swinunetr":
        return build_swinunetr(cfg, img_size=patch_size)
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            "Choose from: segresnet | unet3d | swinunetr"
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_summary(model: nn.Module, input_shape: Tuple = (1, 1, 64, 64, 64)) -> None:
    """Print a brief model summary."""
    counts = count_parameters(model)
    logger.info(
        f"Model: {model.__class__.__name__} | "
        f"Total params: {counts['total'] / 1e6:.2f}M | "
        f"Trainable: {counts['trainable'] / 1e6:.2f}M"
    )
    try:
        dummy = torch.zeros(*input_shape)
        out = model(dummy)
        logger.info(f"Input shape: {tuple(dummy.shape)} → Output: {tuple(out.shape)}")
    except Exception as e:
        logger.warning(f"Forward pass check failed: {e}")
