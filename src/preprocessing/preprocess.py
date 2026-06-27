"""
preprocessing/preprocess.py — CBCT volume preprocessing pipeline.

Steps:
  1. Load NIfTI / NRRD / MHA volumes + labels via SimpleITK
  2. Resample to isotropic spacing
  3. HU clipping + normalisation
  4. Optional foreground cropping
  5. Save as compressed NIfTI (.nii.gz) for fast loading
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from src.utils.helpers import get_logger, find_files

logger = get_logger("preprocess")


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def load_volume(path: str) -> sitk.Image:
    """Load any SimpleITK-supported medical image."""
    img = sitk.ReadImage(str(path))
    logger.debug(f"Loaded {Path(path).name} | size={img.GetSize()} spacing={img.GetSpacing()}")
    return img


def save_volume(img: sitk.Image, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


# ─────────────────────────────────────────────
# Core preprocessing ops
# ─────────────────────────────────────────────

def resample_volume(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
    is_label: bool = False,
) -> sitk.Image:
    """
    Resample image to target isotropic spacing.
    Uses linear interpolation for images, nearest-neighbour for labels.
    """
    original_spacing = np.array(img.GetSpacing())
    original_size = np.array(img.GetSize())
    target_spacing = np.array(target_spacing)

    new_size = (original_size * (original_spacing / target_spacing)).astype(int).tolist()

    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(img)


def clip_and_normalize(
    img: sitk.Image,
    clip_range: Tuple[float, float] = (-1000, 3000),
    method: str = "z_score",
) -> sitk.Image:
    """Clip HU values then normalise."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    # Clip
    arr = np.clip(arr, clip_range[0], clip_range[1])

    # Normalise
    if method == "z_score":
        mean = arr[arr > clip_range[0]].mean()
        std = arr[arr > clip_range[0]].std() + 1e-8
        arr = (arr - mean) / std
    elif method == "min_max":
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo + 1e-8)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def foreground_crop(
    img: sitk.Image,
    label: sitk.Image,
    margin: int = 10,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Crop both image and label to the bounding box of the label foreground,
    with an optional margin (voxels).
    """
    arr_lbl = sitk.GetArrayFromImage(label)
    fg = np.where(arr_lbl > 0)

    if len(fg[0]) == 0:
        logger.warning("Empty label — skipping foreground crop.")
        return img, label

    z0, z1 = max(0, fg[0].min() - margin), min(arr_lbl.shape[0], fg[0].max() + margin + 1)
    y0, y1 = max(0, fg[1].min() - margin), min(arr_lbl.shape[1], fg[1].max() + margin + 1)
    x0, x1 = max(0, fg[2].min() - margin), min(arr_lbl.shape[2], fg[2].max() + margin + 1)

    crop = sitk.CropImageFilter()
    # SimpleITK uses (x, y, z) order; array is (z, y, x)
    lower = [x0, y0, z0]
    upper = [
        arr_lbl.shape[2] - x1,
        arr_lbl.shape[1] - y1,
        arr_lbl.shape[0] - z1,
    ]
    crop.SetLowerBoundaryCropSize(lower)
    crop.SetUpperBoundaryCropSize(upper)

    return crop.Execute(img), crop.Execute(label)


def binarize_label(label: sitk.Image, foreground_classes: Optional[List[int]] = None) -> sitk.Image:
    """
    Convert multi-class label to binary mask.
    If foreground_classes is None, all non-zero voxels become foreground.
    """
    arr = sitk.GetArrayFromImage(label).astype(np.uint8)
    if foreground_classes is not None:
        mask = np.zeros_like(arr, dtype=np.uint8)
        for c in foreground_classes:
            mask[arr == c] = 1
        arr = mask
    else:
        arr = (arr > 0).astype(np.uint8)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(label)
    return out


# ─────────────────────────────────────────────
# Dataset-level pipeline
# ─────────────────────────────────────────────

class DentalPreprocessor:
    """
    End-to-end preprocessing pipeline for dental CBCT datasets.

    Expected raw layout (ToothFairy2 / nnUNet style):
        raw_dir/
          imagesTr/   *.nii.gz    (images)
          labelsTr/   *.nii.gz    (labels)
          imagesTs/   *.nii.gz    (test images, optional)

    Output layout:
        processed_dir/
          images/   <case_id>_img.nii.gz
          labels/   <case_id>_lbl.nii.gz
          dataset.json
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.raw_dir = Path(cfg["dataset"]["root"])
        self.proc_dir = Path(cfg["dataset"]["processed"])
        self.target_spacing = tuple(cfg["preprocessing"]["target_spacing"])
        self.clip_hu = tuple(cfg["preprocessing"]["clip_hu"])
        self.normalize = cfg["preprocessing"]["normalize"]
        self.fg_crop = cfg["preprocessing"]["foreground_crop"]

    def _find_pairs(self) -> List[Dict[str, Path]]:
        """Discover image / label pairs from common layout variants."""
        pairs = []

        # nnUNet / ToothFairy layout
        img_dirs = ["imagesTr", "images", "Images", "scans"]
        lbl_dirs = ["labelsTr", "labels", "Labels", "masks"]

        for id_, ld_ in zip(img_dirs, lbl_dirs):
            idir = self.raw_dir / id_
            ldir = self.raw_dir / ld_
            if idir.exists() and ldir.exists():
                imgs = sorted(find_files(str(idir)))
                for img_path in imgs:
                    stem = img_path.name.replace(".nii.gz", "").replace(".nii", "")
                    # Try common label naming conventions
                    for ext in [".nii.gz", ".nii"]:
                        lbl_path = ldir / (stem + ext)
                        if lbl_path.exists():
                            pairs.append({"id": stem, "image": img_path, "label": lbl_path})
                            break
                if pairs:
                    logger.info(f"Found {len(pairs)} image/label pairs in {idir.name}/{ldir.name}")
                    return pairs

        # Flat layout: same folder, img_ / lbl_ prefix
        all_imgs = find_files(str(self.raw_dir))
        for f in all_imgs:
            if "img" in f.stem.lower() or "scan" in f.stem.lower():
                stem = f.stem.replace("_img", "").replace("_scan", "").replace(".nii", "")
                for lbl_name in [f"{stem}_lbl", f"{stem}_mask", f"{stem}_label"]:
                    for ext in [".nii.gz", ".nii"]:
                        lp = f.parent / (lbl_name + ext)
                        if lp.exists():
                            pairs.append({"id": stem, "image": f, "label": lp})
                            break

        logger.info(f"Found {len(pairs)} image/label pairs (flat layout)")
        return pairs

    def process_one(self, pair: Dict) -> Optional[Dict]:
        """Process a single image/label pair. Returns metadata dict or None on error."""
        case_id = pair["id"]
        out_img = self.proc_dir / "images" / f"{case_id}_img.nii.gz"
        out_lbl = self.proc_dir / "labels" / f"{case_id}_lbl.nii.gz"

        if out_img.exists() and out_lbl.exists():
            logger.debug(f"[{case_id}] Already processed, skipping.")
            return {"id": case_id, "image": str(out_img), "label": str(out_lbl)}

        try:
            img = load_volume(pair["image"])
            lbl = load_volume(pair["label"])

            # 1. Resample
            img = resample_volume(img, self.target_spacing, is_label=False)
            lbl = resample_volume(lbl, self.target_spacing, is_label=True)

            # 2. Binarize label (all non-zero → foreground)
            lbl = binarize_label(lbl)

            # 3. Foreground crop
            if self.fg_crop:
                img, lbl = foreground_crop(img, lbl)

            # 4. Clip + normalise
            img = clip_and_normalize(img, self.clip_hu, self.normalize)

            # 5. Save
            save_volume(img, str(out_img))
            save_volume(lbl, str(out_lbl))

            arr_lbl = sitk.GetArrayFromImage(lbl)
            fg_ratio = float(arr_lbl.mean())
            return {
                "id": case_id,
                "image": str(out_img),
                "label": str(out_lbl),
                "shape": list(sitk.GetArrayFromImage(img).shape),
                "foreground_ratio": round(fg_ratio, 5),
            }

        except Exception as e:
            logger.error(f"[{case_id}] Failed: {e}")
            return None

    def run(self) -> List[Dict]:
        """Process all cases and return metadata list."""
        (self.proc_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.proc_dir / "labels").mkdir(parents=True, exist_ok=True)

        pairs = self._find_pairs()
        if not pairs:
            raise FileNotFoundError(
                f"No image/label pairs found under {self.raw_dir}. "
                "Please download the dataset first (see README.md)."
            )

        metadata = []
        for pair in tqdm(pairs, desc="Preprocessing"):
            result = self.process_one(pair)
            if result:
                metadata.append(result)

        # Save dataset manifest
        manifest = {
            "total_cases": len(metadata),
            "target_spacing": list(self.target_spacing),
            "normalization": self.normalize,
            "cases": metadata,
        }
        manifest_path = self.proc_dir / "dataset.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Preprocessing complete. {len(metadata)} cases saved → {self.proc_dir}")
        logger.info(f"Manifest: {manifest_path}")
        return metadata


if __name__ == "__main__":
    import argparse
    from src.utils.helpers import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocessor = DentalPreprocessor(cfg)
    preprocessor.run()
