"""
preprocessing/dataloader.py — MONAI-based DataLoaders with augmentation.

Patch-based training strategy is used because dental CBCT volumes are too
large to fit in GPU memory as a whole (typically 400–600 voxels per axis).
Random patches of size 64³ are sampled per forward pass.
"""

from typing import Dict, List, Optional

import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    ToTensord,
)

from src.utils.helpers import get_logger

logger = get_logger("dataloader")


# ─────────────────────────────────────────────
# Transform factories
# ─────────────────────────────────────────────

def get_base_transforms(orientation: str = "RAS") -> Compose:
    """Transforms applied at load time for all splits."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=orientation),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_train_transforms(cfg: Dict) -> Compose:
    """Full training transform chain including patch sampling and augmentation."""
    aug = cfg.get("augmentation", {})
    patch_size = cfg["dataset"]["patch_size"]
    pos_neg_ratio = 1  # balanced positive/negative patches

    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image", "label"]),

        # Pad if volume is smaller than patch size
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),

        # Sample random 3D patches: ensures each patch has at least one foreground voxel
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=pos_neg_ratio,
            neg=1,
            num_samples=cfg["dataset"].get("num_patches_per_volume", 4),
            image_key="image",
        ),
    ]

    if aug.get("enabled", True):
        transforms += [
            RandFlipd(
                keys=["image", "label"],
                prob=aug.get("rand_flip_prob", 0.5),
                spatial_axis=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=aug.get("rand_flip_prob", 0.5),
                spatial_axis=1,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=aug.get("rand_flip_prob", 0.5),
                spatial_axis=2,
            ),
            RandAffined(
                keys=["image", "label"],
                prob=aug.get("rand_rotate_prob", 0.3),
                rotate_range=(0.3, 0.3, 0.3),
                scale_range=(
                    aug.get("rand_scale_range", [0.9, 1.1])[0] - 1,
                    aug.get("rand_scale_range", [0.9, 1.1])[1] - 1,
                ),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandScaleIntensityd(
                keys=["image"],
                factors=0.1,
                prob=aug.get("rand_brightness_prob", 0.3),
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=aug.get("rand_brightness_prob", 0.3),
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=aug.get("gaussian_noise_prob", 0.2),
                std=0.05,
            ),
        ]

    transforms.append(ToTensord(keys=["image", "label"]))
    return Compose(transforms)


def get_val_transforms(cfg: Dict) -> Compose:
    """Minimal transforms for validation and test sets (full volume, no augmentation)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


# ─────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────

def build_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: Optional[List[Dict]],
    cfg: Dict,
) -> Dict[str, DataLoader]:
    """
    Build and return train / val / (test) DataLoaders.

    Uses MONAI CacheDataset to cache preprocessed volumes in RAM
    for fast epoch iteration.
    """
    cache_rate = cfg["preprocessing"].get("cache_rate", 1.0)
    num_workers = min(4, torch.get_num_threads())
    batch_size = cfg["training"]["batch_size"]

    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    logger.info(f"Building datasets | cache_rate={cache_rate} | workers={num_workers}")

    train_ds = CacheDataset(
        data=train_data,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,   # CacheDataset is already parallel at load time
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        ),
    }

    if test_data:
        test_ds = CacheDataset(
            data=test_data,
            transform=val_transforms,
            cache_rate=cache_rate,
        )
        loaders["test"] = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    logger.info(
        f"DataLoaders ready | train={len(train_ds)} | val={len(val_ds)} "
        f"| test={len(test_data) if test_data else 0}"
    )
    return loaders
