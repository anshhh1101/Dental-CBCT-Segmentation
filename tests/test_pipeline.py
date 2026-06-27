"""
tests/test_pipeline.py — Unit tests for core pipeline components.

Run with:  pytest tests/ -v --tb=short
"""

import numpy as np
import pytest


# ─────────────────────────────────────────────
# Metric tests
# ─────────────────────────────────────────────

class TestMetrics:
    def test_dice_perfect(self):
        from src.utils.helpers import dice_score
        a = np.ones((10, 10, 10), dtype=bool)
        assert dice_score(a, a) == pytest.approx(1.0, abs=1e-4)

    def test_dice_no_overlap(self):
        from src.utils.helpers import dice_score
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.ones((10, 10, 10), dtype=bool)
        a[:5] = True
        b[:5] = False
        score = dice_score(a, b)
        assert score < 0.01

    def test_iou_perfect(self):
        from src.utils.helpers import iou_score
        a = np.ones((5, 5, 5), dtype=bool)
        assert iou_score(a, a) == pytest.approx(1.0, abs=1e-4)

    def test_iou_half_overlap(self):
        from src.utils.helpers import iou_score
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.zeros((10, 10, 10), dtype=bool)
        a[:, :, :5] = True
        b[:, :, 3:8] = True
        iou = iou_score(a, b)
        assert 0.0 < iou < 1.0


# ─────────────────────────────────────────────
# Post-processing tests
# ─────────────────────────────────────────────

class TestPostprocessing:
    def test_threshold(self):
        from src.postprocessing.postprocess import threshold
        prob = np.array([[[0.3, 0.6, 0.9]]])
        mask = threshold(prob, 0.5)
        expected = np.array([[[0, 1, 1]]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)

    def test_remove_small_objects_removes_noise(self):
        from src.postprocessing.postprocess import remove_small_objects
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[0, 0, 0] = 1          # tiny blob (1 voxel)
        mask[5:15, 5:15, 5:15] = 1 # large region (1000 voxels)
        cleaned = remove_small_objects(mask, min_size=10)
        assert cleaned[0, 0, 0] == 0
        assert cleaned[10, 10, 10] == 1

    def test_fill_holes(self):
        from src.postprocessing.postprocess import fill_holes
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[3:7, 3:7, 3:7] = 1
        mask[4:6, 4:6, 4:6] = 0   # punch a hole
        filled = fill_holes(mask)
        # Hole should be filled in some slices (not all: slice-by-slice fill)
        assert filled.sum() >= mask.sum()

    def test_postprocess_pipeline(self):
        from src.postprocessing.postprocess import postprocess_prediction
        cfg = {
            "postprocessing": {
                "confidence_threshold": 0.5,
                "remove_small_objects": True,
                "min_object_size_voxels": 5,
                "fill_holes": True,
                "closing_iterations": 1,
                "smooth_predictions": False,
            }
        }
        prob = np.random.rand(32, 32, 32).astype(np.float32)
        prob[10:20, 10:20, 10:20] = 0.9   # strong foreground region
        raw, refined = postprocess_prediction(prob, cfg)
        assert raw.dtype == np.uint8
        assert refined.dtype == np.uint8
        assert refined[15, 15, 15] == 1


# ─────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────

class TestModels:
    BASE_CFG = {
        "model": {
            "in_channels": 1,
            "out_channels": 2,
            "init_filters": 8,
            "dropout_prob": 0.0,
            "channels": [8, 16, 32],
            "strides": [2, 2],
        },
        "dataset": {"patch_size": [32, 32, 32]},
    }

    @pytest.mark.parametrize("arch", ["segresnet", "unet3d"])
    def test_forward_pass(self, arch):
        import torch
        cfg = {**self.BASE_CFG, "model": {**self.BASE_CFG["model"], "architecture": arch}}
        from src.models.architectures import build_model
        model = build_model(cfg)
        model.eval()
        x = torch.zeros(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 32, 32, 32), f"Unexpected output shape: {out.shape}"

    def test_loss_dice_ce(self):
        import torch
        cfg = {
            "training": {"loss": "dice_ce", "dice_ce_lambda_dice": 0.5, "dice_ce_lambda_ce": 0.5},
            "model": {"out_channels": 2},
        }
        from src.models.losses import build_loss
        loss_fn = build_loss(cfg)
        logits = torch.randn(2, 2, 8, 8, 8)
        labels = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()
        loss = loss_fn(logits, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)


# ─────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────

class TestConfig:
    def test_load_config(self, tmp_path):
        import yaml
        from src.utils.helpers import load_config
        cfg_path = tmp_path / "test_cfg.yaml"
        cfg_path.write_text("model:\n  out_channels: 2\n")
        cfg = load_config(str(cfg_path))
        assert cfg["model"]["out_channels"] == 2
