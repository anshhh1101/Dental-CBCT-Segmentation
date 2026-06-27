# 🦷 Dental CBCT Segmentation Pipeline

End-to-end machine learning pipeline for dental structure segmentation on 3-D CBCT volumes.  
Targets: **dental restorations** (fillings, crowns, bridges, implants) via the **ToothFairy2** dataset,  
with architecture and post-processing tuned for the morphology of small, high-density structures.

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
11. [Results](#results)
12. [Challenges & Future Work](#challenges--future-work)

---

## Overview

| Component | Choice | Reason |
|-----------|--------|--------|
| Dataset | ToothFairy2 (Zenodo) | 443 CBCTs, multi-class labels, public |
| Model | SegResNet (default) | Fast, ~4M params, SOTA on MICCAI benchmarks |
| Patch strategy | 64³ random patches | GPU memory management for large volumes |
| Loss | Dice + Cross-Entropy | Handles severe foreground/background imbalance |
| Post-processing | CC filtering + closing + hole-fill | Removes noise, smooths boundaries |
| Visualisation | Plotly interactive HTML | Self-contained, shareable, browser-native |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/dental_pipeline.git
cd dental_pipeline

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch ≥ 2.1, CUDA-capable GPU recommended (8 GB+ VRAM).

### 2. Download the Dataset

```bash
# List available datasets
python scripts/preprocess.py --list_datasets

# Download ToothFairy2 (~7 GB) — requires internet access
python scripts/preprocess.py --download toothfairy2
```

> **Manual download option:**  
> Download `Dataset112_ToothFairy2.zip` from [Zenodo 10934857](https://zenodo.org/records/10934857)  
> and extract to `data/raw/toothfairy2/`.

### 3. Preprocess

```bash
python scripts/preprocess.py
```

This will:
- Resample all volumes to 0.4 mm isotropic spacing
- Clip HU values to [-1000, 3000] and z-score normalise
- Crop to foreground bounding box
- Save compressed NIfTI to `data/processed/`
- Write `data/splits/{train,val,test}.json`

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
# Full test set + metrics + visualisations
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --test_set \
    --evaluate \
    --visualize
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
├── data/
│   ├── raw/                         # Downloaded dataset (gitignored)
│   ├── processed/                   # Preprocessed NIfTI files (gitignored)
│   └── splits/                      # train.json, val.json, test.json
│
├── outputs/
│   ├── checkpoints/best_model.pth
│   ├── predictions/
│   ├── visualizations/
│   └── logs/
│
└── tests/
    └── test_pipeline.py
```

---

## Dataset

### ToothFairy2

| Property | Value |
|----------|-------|
| Source | [Zenodo 10934857](https://zenodo.org/records/10934857) |
| Format | NIfTI (.nii.gz), nnUNet layout |
| Volumes | ~443 CBCT scans |
| Resolution | Variable (typically ~0.2–0.5 mm) |
| Labels | Teeth (1–32), implants, crowns, bridges, mandible, maxilla |
| License | CC BY 4.0 |

**Why not cavities directly?**  
The latest public ToothFairy releases no longer include cavity/caries annotations due to annotation quality issues. This pipeline therefore targets **dental restorations** (fillings + crowns + implants), which are the closest publicly annotated structures and share similar segmentation challenges: small size, high contrast vs surrounding tissue.

**Alternative datasets:**
- `mmdental` — multimodal X-ray + CBCT (see `src/preprocessing/download.py`)
- Any custom dataset following the `imagesTr/labelsTr` nnUNet layout

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
    │   ├── ResBlock(1→16)   64³
    │   ├── ResBlock(16→32)  32³  ← stride-2 downsample
    │   ├── ResBlock(32→64)  16³
    │   ├── ResBlock(64→128)  8³
    │   └── ResBlock(128→256) 4³
    │
    └── Decoder
        ├── Upsample + ResBlock(256→128)
        ├── Upsample + ResBlock(128→64)
        ├── Upsample + ResBlock(64→32)
        ├── Upsample + ResBlock(32→16)
        └── Conv1×1 → 2 channels → Softmax

Parameters: ~4.0 M (init_filters=16)
```

Each residual block: Conv3D → BN → ReLU → Conv3D → BN → (+ skip) → ReLU.

**Why SegResNet over UNet3D?**  
SegResNet uses larger receptive fields through progressive downsampling and has consistently achieved top performance on the Medical Segmentation Decathlon. It is ~2× faster to train than SwinUNETR while remaining competitive in accuracy.

**SwinUNETR** is available via `config.yaml → model.architecture: swinunetr` for maximum accuracy at the cost of ~6× more parameters and GPU memory.

---

## Training

### Strategy

| Hyperparameter | Value | Rationale |
|---|---|---|
| Patch size | 64³ | GPU memory vs context trade-off |
| Patches/volume | 8 | Balanced positive/negative sampling |
| Batch size | 2 | Fits 8 GB VRAM with AMP |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) | Weight decay regularises deeply |
| Scheduler | Cosine annealing | Smooth decay, no plateau tuning |
| Loss | Dice + CE (λ=0.5 each) | Handles 95%+ background class |
| AMP | Yes | ~1.8× speedup, 40% less VRAM |
| Early stopping | Patience=30 epochs | Avoids overfitting small datasets |

### Monitoring

```bash
tensorboard --logdir outputs/logs/tensorboard
```

Metrics logged: `Loss/train`, `Dice/val`, `LR`.  
CSV log: `outputs/logs/metrics.csv`.

---

## Inference

Inference uses **sliding-window** with 50% overlap and Gaussian weighting at patch borders, which eliminates boundary artefacts that would appear with hard tiling.

```bash
# Single file (raw, unpreprocessed CBCT)
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --input my_scan.nii.gz \
    --output my_scan_pred.nii.gz \
    --visualize

# Full test split
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --test_set --evaluate --visualize
```

Results written to `outputs/predictions/test_results.json`.

---

## Visualisation

Three output types per case:

| File | Content |
|------|---------|
| `{case_id}_3d.html` | Interactive Plotly 3-D volume + segmentation isosurface |
| `{case_id}_dashboard.html` | Axial / coronal / sagittal overlay panels |
| `{case_id}_{view}.png` | Static slice grid (6 slices × 2 rows: image + overlay) |

All HTML files are **self-contained** (Plotly loaded via CDN) and require no server.

---

## Configuration Reference

```yaml
model:
  architecture: segresnet    # segresnet | unet3d | swinunetr
  out_channels: 2            # background + 1 foreground class

training:
  epochs: 200
  batch_size: 2
  learning_rate: 1.0e-4
  loss: dice_ce              # dice | ce | dice_ce | focal | dice_focal

postprocessing:
  min_object_size_voxels: 50
  closing_iterations: 2
  confidence_threshold: 0.5
```

Full options documented in `config.yaml`.

---

## Results

*Results will populate here after training on your hardware. Typical expected ranges for dental restoration segmentation on ToothFairy2:*

| Metric | Expected Range |
|--------|---------------|
| Dice (val) | 0.72 – 0.88 |
| IoU (test) | 0.60 – 0.80 |
| HD95 (mm) | 2.0 – 6.0 |

*Training time: ~6–12 hours on a single A100 80GB GPU for 200 epochs.*

---

## Challenges & Future Work

### Challenges

1. **Class imbalance** — Dental restorations occupy <3% of CBCT volume.  
   *Addressed by:* Dice loss, positive/negative patch sampling.

2. **Memory constraints** — Full-resolution CBCT (500³+) cannot fit in GPU RAM.  
   *Addressed by:* 64³ patch training + sliding-window inference.

3. **Dataset variability** — Scans from different scanners have variable spacing and FOV.  
   *Addressed by:* Isotropic resampling and z-score normalisation per volume.

4. **Annotation quality** — Public datasets vary in label consistency (especially small caries).  
   *Addressed by:* Post-processing to remove sub-threshold predictions.

### Future Improvements

- **Self-supervised pre-training** (e.g., SwinUNETR pre-training on unlabelled CBCT) for better feature initialisation.
- **Multi-class segmentation** — Separate channels for fillings, crowns, implants using ToothFairy2 multi-label annotations.
- **Instance segmentation** — Separate individual teeth using connected-component analysis or panoptic approaches.
- **Caries detection** — Fine-tuning on specialised datasets with cavity annotations once they become publicly available.
- **Clinical integration** — DICOM I/O and OHIF/3D Slicer plugin export.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## License

MIT License. Dataset is subject to its own license (CC BY 4.0 for ToothFairy2).
