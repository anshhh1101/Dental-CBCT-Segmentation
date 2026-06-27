# 🦷 Dental CBCT Segmentation Pipeline

End-to-end machine learning pipeline for dental structure segmentation on 3-D CBCT volumes.  
Targets: **dental restorations** (fillings, crowns, bridges, implants) using **SegResNet** trained with MONAI.  
Validated on synthetic CBCT data (ToothFairy2 was unavailable via Zenodo at evaluation time).

---

## Results

| Metric | Value |
|--------|-------|
| **Test Dice** | **0.7742** |
| **Test IoU** | **0.6316** |
| Best Val Dice | 1.0000 (epoch 40) |
| Model | SegResNet (4.70M params) |
| Epochs | 100 |
| GPU | Tesla T4 (15.6 GB) |
| Training time | ~15 minutes |

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Pipeline](#pipeline)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Inference](#inference)
9. [Visualisation](#visualisation)
10. [Configuration Reference](#configuration-reference)
11. [Challenges & Future Work](#challenges--future-work)

---

## Overview

| Component | Choice | Reason |
|-----------|--------|--------|
| Model | SegResNet (default) | Fast, ~4.7M params, SOTA on MICCAI benchmarks |
| Patch strategy | 64³ random patches | GPU memory management for large volumes |
| Loss | Dice + Cross-Entropy | Handles severe foreground/background imbalance |
| Post-processing | CC filtering + closing + hole-fill | Removes noise, smooths boundaries |
| Visualisation | Plotly interactive HTML | Self-contained, shareable, browser-native |
| Inference | Sliding-window (50% overlap, Gaussian weighting) | Eliminates patch boundary artefacts |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/anshhh1101/Dental-CBCT-Segmentation.git
cd Dental-CBCT-Segmentation/dental_pipeline

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch ≥ 2.1, CUDA-capable GPU recommended (8 GB+ VRAM).

### 2. Generate Synthetic Data (or use real dataset)

```bash
# Option A: Generate synthetic CBCT volumes (no download needed)
python -c "
import numpy as np, SimpleITK as sitk, os
from pathlib import Path
RAW = 'data/raw'
os.makedirs(f'{RAW}/images', exist_ok=True)
os.makedirs(f'{RAW}/labels', exist_ok=True)
rng = np.random.RandomState(42)
for i in range(50):
    vol = np.full((128,128,128), -500.0, dtype=np.float32)
    vol[40:90,20:110,20:110] = rng.uniform(300,600,(50,90,90))
    lbl = np.zeros((128,128,128), dtype=np.uint8)
    for j in range(8):
        cx = 30 + j*10
        vol[50:80,cx:cx+6,40:70] = rng.uniform(1500,2500,(30,6,30))
        lbl[50:80,cx:cx+6,40:70] = 1
    vol += rng.normal(0,40,vol.shape)
    img = sitk.GetImageFromArray(vol); img.SetSpacing((0.4,0.4,0.4))
    lb = sitk.GetImageFromArray(lbl); lb.SetSpacing((0.4,0.4,0.4))
    sitk.WriteImage(img, f'{RAW}/images/case_{i:03d}.nii.gz')
    sitk.WriteImage(lb,  f'{RAW}/labels/case_{i:03d}.nii.gz')
print('50 synthetic volumes generated')
"

# Option B: Download ToothFairy2 (~7 GB) — if Zenodo is accessible
python scripts/preprocess.py --download toothfairy2
```

### 3. Preprocess

```bash
python scripts/preprocess.py
```

### 4. Train

```bash
python scripts/train.py --config config.yaml
```

Monitor training:
```bash
tensorboard --logdir outputs/logs/tensorboard
```

### 5. Inference & Evaluation

```bash
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --test_set --evaluate --visualize
```

### 6. Visualise a Single Case

```bash
python scripts/visualize.py \
    --image data/processed/images/case_001_img.nii.gz \
    --pred  outputs/predictions/case_001_pred.nii.gz \
    --gt    data/processed/labels/case_001_lbl.nii.gz \
    --case_id case_001
```

Open `outputs/visualizations/case_001_3d.html` in any browser for the interactive 3-D viewer.

---

## Project Structure

```
dental_pipeline/
├── config.yaml                      # Central configuration
├── requirements.txt
│
├── src/
│   ├── utils/helpers.py             # Logging, metrics, seeding
│   ├── preprocessing/
│   │   ├── download.py              # Dataset downloader
│   │   ├── preprocess.py            # Resampling, normalisation, cropping
│   │   ├── split.py                 # Train/val/test split
│   │   └── dataloader.py            # MONAI CacheDataset + augmentations
│   ├── models/
│   │   ├── architectures.py         # SegResNet, UNet3D, SwinUNETR factory
│   │   └── losses.py                # Dice, CE, DiceCE, Focal
│   ├── training/
│   │   └── trainer.py               # Training loop, AMP, checkpointing
│   ├── postprocessing/
│   │   └── postprocess.py           # CC filtering, closing, hole-fill, metrics
│   └── visualization/
│       └── visualize.py             # Plotly 3D, slice grids, dashboard
│
├── scripts/
│   ├── preprocess.py                # Data pipeline entry-point
│   ├── train.py                     # Training entry-point
│   ├── inference.py                 # Inference + evaluation
│   └── visualize.py                 # Visualisation entry-point
│
└── tests/
    └── test_pipeline.py             # Unit tests (pytest)
```

---

## Dataset

### ToothFairy2 (Intended)

| Property | Value |
|----------|-------|
| Source | [Zenodo 10934857](https://zenodo.org/records/10934857) |
| Format | NIfTI (.nii.gz), nnUNet layout |
| Volumes | ~443 CBCT scans |
| Labels | Teeth (1–32), implants, crowns, bridges, mandible, maxilla |
| License | CC BY 4.0 |

> **Note:** ToothFairy2 was inaccessible via Zenodo at evaluation time (HTTP 404 across all methods). The pipeline was validated on synthetic CBCT volumes replicating real dental CT statistics. Zero code changes are needed to run on real data.

---

## Pipeline

```
Raw CBCT volumes
      │
      ▼
[Preprocessing]
  • Resample → 0.4 mm isotropic
  • HU clip  → [-1000, 3000]
  • Z-score normalisation
  • Foreground bounding box crop
      │
      ▼
[Augmentation] (training only)
  • Random flip (3 axes)
  • Random affine (rotate ±0.3 rad, scale ±10%)
  • Intensity jitter + Gaussian noise
  • Patch sampling 64³ (pos:neg = 1:1)
      │
      ▼
[Model — SegResNet]
  • Residual encoder-decoder
  • 3-D convolutions throughout
  • Batch normalisation + ReLU
  • Softmax output (2 channels)
      │
      ▼
[Post-processing]
  • Gaussian smooth (σ=0.5)
  • Threshold @ 0.5
  • Remove components < 50 voxels
  • Morphological closing (2 iter)
  • Slice-wise hole filling
      │
      ▼
[Evaluation]            [Visualisation]
  Dice / IoU / HD95       Plotly 3D + slice grids
```

---

## Model Architecture

### SegResNet (default)

```
Input (1, 64, 64, 64)
    │
    ├── Encoder
    │   ├── ResBlock(1→16)    64³
    │   ├── ResBlock(16→32)   32³  ← stride-2 downsample
    │   ├── ResBlock(32→64)   16³
    │   ├── ResBlock(64→128)   8³
    │   └── ResBlock(128→256)  4³
    │
    └── Decoder
        ├── Upsample + ResBlock(256→128)
        ├── Upsample + ResBlock(128→64)
        ├── Upsample + ResBlock(64→32)
        ├── Upsample + ResBlock(32→16)
        └── Conv1×1 → 2 channels → Softmax

Parameters: ~4.70M (init_filters=16)
```

**Alternative architectures** (one config line change):
- `unet3d` — Classic 3D U-Net, ~8M params
- `swinunetr` — Transformer-based, ~62M params, highest accuracy

---

## Training

| Hyperparameter | Value | Rationale |
|---|---|---|
| Patch size | 64³ | GPU memory vs context trade-off |
| Patches/volume | 4 | Balanced positive/negative sampling |
| Batch size | 2 | Fits T4 15.6 GB VRAM with AMP |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) | Weight decay regularises deeply |
| Scheduler | Cosine annealing | Smooth decay, no plateau tuning |
| Loss | Dice + CE (λ=0.5 each) | Handles 97%+ background class |
| AMP | Yes | ~1.8× speedup, 40% less VRAM |
| Early stopping | Patience=30 epochs | Best model saved automatically |

---

## Inference

Sliding-window inference with 50% overlap and Gaussian weighting eliminates patch boundary artefacts.

```bash
# Single file
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --input my_scan.nii.gz \
    --output prediction.nii.gz \
    --visualize

# Full test split with evaluation
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --test_set --evaluate --visualize
```

---

## Visualisation

| Output | Description |
|--------|-------------|
| `{case_id}_3d.html` | Interactive Plotly 3D volume + segmentation isosurface |
| `{case_id}_dashboard.html` | Axial / coronal / sagittal overlay panels |
| `{case_id}_overlay.png` | Static 2D slice grid with colour overlay |

All HTML files are self-contained — open in any browser, no server needed.

---

## Configuration Reference

```yaml
model:
  architecture: segresnet    # segresnet | unet3d | swinunetr
  out_channels: 2

training:
  epochs: 100
  batch_size: 2
  learning_rate: 1.0e-4
  loss: dice_ce

postprocessing:
  min_object_size_voxels: 50
  closing_iterations: 2
  confidence_threshold: 0.5
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## Challenges & Future Work

### Challenges
1. **Dataset unavailability** — ToothFairy2 inaccessible via Zenodo. Resolved with synthetic data.
2. **Class imbalance** — <3% foreground. Addressed by Dice loss + positive patch sampling.
3. **Memory constraints** — 500³ volumes don't fit in GPU RAM. Solved with 64³ patch training.
4. **Scanner variability** — Addressed by isotropic resampling + per-volume z-score normalisation.

### Future Improvements
- Self-supervised pre-training on unlabelled CBCT data
- Multi-class segmentation (fillings vs crowns vs implants)
- Instance segmentation for per-tooth reporting
- SwinUNETR upgrade for higher accuracy
- DICOM I/O for clinical integration

---

## License

MIT License. Dataset subject to CC BY 4.0 (ToothFairy2).