"""
preprocessing/download.py — Dataset acquisition helpers.

Supports:
  - ToothFairy2 (via Zenodo)
  - MMDental Multimodal Dataset
  - Tufts Dental Dataset (panoramic X-rays + segmentation)
  - Custom path
"""

import os
import shutil
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

from src.utils.helpers import get_logger

logger = get_logger("download")

# ─────────────────────────────────────────────
# Public dataset registry
# ─────────────────────────────────────────────
DATASETS = {
    "toothfairy2": {
        "description": "ToothFairy2 — multi-class 3D CBCT (teeth, implants, restorations, mandible, maxilla)",
        "url": "https://zenodo.org/records/10934857/files/Dataset112_ToothFairy2.zip",
        "filename": "ToothFairy2.zip",
        "format": "zip",
        "notes": (
            "Labels include: teeth instances (1-32), implants, crowns, bridges, "
            "mandibular & maxillary bones. ~443 CBCT volumes."
        ),
    },
    "mmdental": {
        "description": "MMDental — multimodal dental dataset with panoramic X-rays and CBCT",
        "url": "https://zenodo.org/records/7812090/files/MMDental.zip",
        "filename": "MMDental.zip",
        "format": "zip",
        "notes": "Contains both panoramic radiographs and CBCT with tooth-level segmentations.",
    },
    "tufts": {
        "description": "Tufts Dental Dataset — 1000 panoramic X-rays with expert segmentations",
        "url": "https://tdd.ece.tufts.edu/",
        "filename": None,
        "format": "manual",
        "notes": (
            "Requires manual download + registration. "
            "Provides bounding-box and pixel-level annotations for 32 tooth classes."
        ),
    },
}


def list_datasets() -> None:
    """Print available datasets."""
    print("\n📦  Available Datasets\n" + "=" * 60)
    for key, info in DATASETS.items():
        print(f"\n[{key}]")
        print(f"  {info['description']}")
        print(f"  Notes: {info['notes']}")
        if info["url"]:
            print(f"  URL:   {info['url']}")
    print()


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream-download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} → {dest}")

    def _reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  Progress: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline after progress
    logger.info(f"Saved to {dest}")


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract zip or tar archive."""
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive.name} → {dest}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)
    elif archive.name.endswith(".tar"):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive.suffix}")
    logger.info("Extraction complete.")


def download_dataset(name: str, raw_dir: str = "data/raw", keep_archive: bool = False) -> Path:
    """
    Download and extract a registered dataset.

    Args:
        name:         Key from DATASETS registry.
        raw_dir:      Root folder to store the data.
        keep_archive: Whether to keep the archive after extraction.

    Returns:
        Path to the extracted dataset directory.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")

    info = DATASETS[name]
    raw_dir = Path(raw_dir)

    if info["format"] == "manual":
        print(f"\n⚠️  Manual download required for '{name}'.")
        print(f"    Visit: {info['url']}")
        print(f"    Place the extracted data under: {raw_dir / name}")
        return raw_dir / name

    archive_path = raw_dir / info["filename"]
    extract_path = raw_dir / name

    if extract_path.exists() and any(extract_path.iterdir()):
        logger.info(f"Dataset already found at {extract_path}. Skipping download.")
        return extract_path

    download_file(info["url"], archive_path)
    extract_archive(archive_path, extract_path)

    if not keep_archive:
        archive_path.unlink(missing_ok=True)
        logger.info(f"Removed archive {archive_path.name}")

    return extract_path


def use_custom_dataset(src: str, raw_dir: str = "data/raw", name: str = "custom") -> Path:
    """
    Copy or symlink a custom local dataset into the raw data directory.

    Args:
        src:     Existing path to dataset folder.
        raw_dir: Destination root.
        name:    Subdirectory name to use.
    """
    src = Path(src)
    dest = Path(raw_dir) / name
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")
    if not dest.exists():
        os.symlink(src.resolve(), dest)
        logger.info(f"Symlinked {src} → {dest}")
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download dental datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--dataset", type=str, default="toothfairy2", help="Dataset key to download")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--keep_archive", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    else:
        out = download_dataset(args.dataset, args.raw_dir, args.keep_archive)
        print(f"\n✅  Dataset ready at: {out}")
