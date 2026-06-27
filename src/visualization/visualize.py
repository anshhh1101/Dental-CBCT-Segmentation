"""
visualization/visualize.py — 3-D interactive visualisation of CBCT + segmentation.

Generates:
  1. Interactive HTML viewer using Plotly (volume rendering + surface mesh)
  2. 2-D slice grids (axial / coronal / sagittal) as PNG
  3. Overlay comparison plots

The HTML output is self-contained and viewable in any modern browser
without any server or additional dependencies.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.utils.helpers import get_logger

logger = get_logger("visualize")


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_arrays(
    image_path: str,
    label_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple]:
    """Load image (and optional label) → numpy arrays + spacing."""
    img_sitk = sitk.ReadImage(str(image_path))
    spacing = img_sitk.GetSpacing()  # (x, y, z) in mm
    img_arr = sitk.GetArrayFromImage(img_sitk)  # (z, y, x)

    lbl_arr = None
    if label_path and Path(label_path).exists():
        lbl_sitk = sitk.ReadImage(str(label_path))
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.uint8)

    return img_arr, lbl_arr, spacing


# ─────────────────────────────────────────────
# 3D Plotly volume + surface rendering
# ─────────────────────────────────────────────

def _downsample(arr: np.ndarray, target_max: int = 128) -> np.ndarray:
    """Downsample along each axis to keep volume manageable for browser rendering."""
    factors = [max(1, s // target_max) for s in arr.shape]
    return arr[::factors[0], ::factors[1], ::factors[2]]


def build_3d_figure(
    img_arr: np.ndarray,
    lbl_arr: Optional[np.ndarray],
    spacing: Tuple,
    title: str = "Dental CBCT 3D View",
    max_vol_dim: int = 96,
) -> go.Figure:
    """
    Create an interactive Plotly figure with:
      - Volume rendering of CBCT scan (semi-transparent)
      - Isosurface / volume rendering of segmentation overlay
    """
    # Downsample for performance
    img_ds = _downsample(img_arr, max_vol_dim)
    z_size, y_size, x_size = img_ds.shape

    # Physical axes (mm)
    sx, sy, sz = spacing
    x_range = np.linspace(0, x_size * sx, x_size)
    y_range = np.linspace(0, y_size * sy, y_size)
    z_range = np.linspace(0, z_size * sz, z_size)
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")

    # Normalise image to [0, 1] for colourscale
    img_norm = img_ds.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    # Swap axes for plotly (needs x, y, z)
    img_plot = img_norm.transpose(2, 1, 0)

    traces = []

    # CBCT volume — rendered as volume trace with dental bone window
    traces.append(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=img_plot.flatten(),
        isomin=0.1,
        isomax=0.9,
        opacity=0.08,
        surface_count=15,
        colorscale="Gray",
        showscale=False,
        name="CBCT Volume",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    # Segmentation overlay
    if lbl_arr is not None:
        lbl_ds = _downsample(lbl_arr, max_vol_dim)
        lbl_plot = lbl_ds.transpose(2, 1, 0).astype(np.float32)

        # Isosurface of the segmentation
        traces.append(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=lbl_plot.flatten(),
            isomin=0.5,
            isomax=1.0,
            surface_count=2,
            colorscale=[[0, "rgba(255,80,80,0.0)"], [1, "rgba(255,80,80,0.85)"]],
            showscale=False,
            name="Segmentation",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

    layout = go.Layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            bgcolor="rgb(10, 10, 20)",
            xaxis=dict(color="white"),
            yaxis=dict(color="white"),
            zaxis=dict(color="white"),
        ),
        paper_bgcolor="rgb(10, 10, 20)",
        font_color="white",
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
    )

    return go.Figure(data=traces, layout=layout)


# ─────────────────────────────────────────────
# 2D slice grids
# ─────────────────────────────────────────────

def plot_slice_grid(
    img_arr: np.ndarray,
    lbl_arr: Optional[np.ndarray],
    out_path: str,
    n_slices: int = 6,
    view: str = "axial",
) -> None:
    """
    Generate a grid of 2-D slices (with optional segmentation overlay).
    view: 'axial' | 'coronal' | 'sagittal'
    """
    if view == "axial":
        indices = np.linspace(0, img_arr.shape[0] - 1, n_slices, dtype=int)
        get_slice = lambda i: (img_arr[i], lbl_arr[i] if lbl_arr is not None else None)
        axis_label = "Axial"
    elif view == "coronal":
        indices = np.linspace(0, img_arr.shape[1] - 1, n_slices, dtype=int)
        get_slice = lambda i: (img_arr[:, i, :], lbl_arr[:, i, :] if lbl_arr is not None else None)
        axis_label = "Coronal"
    else:  # sagittal
        indices = np.linspace(0, img_arr.shape[2] - 1, n_slices, dtype=int)
        get_slice = lambda i: (img_arr[:, :, i], lbl_arr[:, :, i] if lbl_arr is not None else None)
        axis_label = "Sagittal"

    cols = n_slices
    rows = 2 if lbl_arr is not None else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.patch.set_facecolor("#0a0a14")

    if cols == 1:
        axes = np.array(axes).reshape(rows, 1)
    elif rows == 1:
        axes = axes.reshape(1, -1)

    # Normalize image for display
    vmin = np.percentile(img_arr, 1)
    vmax = np.percentile(img_arr, 99)

    for j, idx in enumerate(indices):
        img_sl, lbl_sl = get_slice(idx)

        ax_img = axes[0, j]
        ax_img.imshow(img_sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        ax_img.axis("off")
        ax_img.set_title(f"{axis_label} {idx}", color="white", fontsize=8)

        if lbl_arr is not None and rows > 1:
            ax_ov = axes[1, j]
            ax_ov.imshow(img_sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            mask_rgba = np.zeros((*lbl_sl.shape, 4))
            mask_rgba[..., 0] = 1.0   # red channel
            mask_rgba[..., 3] = lbl_sl.astype(float) * 0.6  # alpha
            ax_ov.imshow(mask_rgba, aspect="equal")
            ax_ov.axis("off")
            if j == 0:
                ax_ov.set_ylabel("Overlay", color="white", fontsize=8)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Slice grid saved → {out_path}")


# ─────────────────────────────────────────────
# Multi-view dashboard
# ─────────────────────────────────────────────

def build_dashboard(
    img_arr: np.ndarray,
    lbl_arr: Optional[np.ndarray],
    case_id: str,
    metrics: Optional[Dict] = None,
) -> go.Figure:
    """
    Plotly dashboard with:
      - Axial / coronal / sagittal slice sliders
      - Metric annotations
    """
    z_mid = img_arr.shape[0] // 2
    y_mid = img_arr.shape[1] // 2
    x_mid = img_arr.shape[2] // 2

    vmin = np.percentile(img_arr, 1)
    vmax = np.percentile(img_arr, 99)

    def norm(sl):
        return np.clip((sl - vmin) / (vmax - vmin + 1e-8), 0, 1)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Axial", "Coronal", "Sagittal"],
    )

    def overlay_img(img_sl, lbl_sl):
        """Blend grayscale + red mask as RGB."""
        gray = (norm(img_sl) * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        if lbl_sl is not None:
            rgb[lbl_sl > 0, 0] = np.clip(rgb[lbl_sl > 0, 0] + 120, 0, 255)
            rgb[lbl_sl > 0, 1] = np.clip(rgb[lbl_sl > 0, 1] - 40, 0, 255)
            rgb[lbl_sl > 0, 2] = np.clip(rgb[lbl_sl > 0, 2] - 40, 0, 255)
        return rgb

    lbl_ax = lbl_arr[z_mid] if lbl_arr is not None else None
    lbl_co = lbl_arr[:, y_mid, :] if lbl_arr is not None else None
    lbl_sa = lbl_arr[:, :, x_mid] if lbl_arr is not None else None

    fig.add_trace(go.Image(z=overlay_img(img_arr[z_mid], lbl_ax)), row=1, col=1)
    fig.add_trace(go.Image(z=overlay_img(img_arr[:, y_mid, :], lbl_co)), row=1, col=2)
    fig.add_trace(go.Image(z=overlay_img(img_arr[:, :, x_mid], lbl_sa)), row=1, col=3)

    annotation_text = f"Case: {case_id}"
    if metrics:
        annotation_text += (
            f"  |  Dice={metrics.get('dice', 'N/A')}"
            f"  IoU={metrics.get('iou', 'N/A')}"
            f"  HD95={metrics.get('hd95', 'N/A')}mm"
        )

    fig.update_layout(
        title=annotation_text,
        paper_bgcolor="#0a0a14",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ─────────────────────────────────────────────
# Batch export
# ─────────────────────────────────────────────

def visualize_case(
    image_path: str,
    pred_path: Optional[str],
    gt_path: Optional[str],
    out_dir: str,
    case_id: str,
    metrics: Optional[Dict] = None,
    cfg: Optional[Dict] = None,
) -> None:
    """Generate and save all visualisations for a single case."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_arr, gt_arr, spacing = load_arrays(image_path, gt_path)

    # Use prediction if available, else GT
    seg_arr = None
    if pred_path and Path(pred_path).exists():
        _, seg_arr, _ = load_arrays(image_path, pred_path)
    elif gt_arr is not None:
        seg_arr = gt_arr

    # 1. Interactive 3-D HTML
    fig3d = build_3d_figure(img_arr, seg_arr, spacing, title=f"3D View — {case_id}")
    html_path = Path(out_dir) / f"{case_id}_3d.html"
    fig3d.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info(f"3D viewer → {html_path}")

    # 2. 2-D slice grids
    for view in ["axial", "coronal", "sagittal"]:
        plot_slice_grid(
            img_arr, seg_arr,
            out_path=str(Path(out_dir) / f"{case_id}_{view}.png"),
            n_slices=6,
            view=view,
        )

    # 3. Dashboard HTML
    dash = build_dashboard(img_arr, seg_arr, case_id, metrics)
    dash_path = Path(out_dir) / f"{case_id}_dashboard.html"
    dash.write_html(str(dash_path), include_plotlyjs="cdn")
    logger.info(f"Dashboard → {dash_path}")
