"""
src/explainers/cf_visualizer.py
-------------------------------
Publication-quality 8-panel counterfactual visualization.

Panel layout per row:
  1. Original + lesion contour
  2. Grad-CAM on Original (turbo, α=0.4)
  3. Counterfactual + lesion contour
  4. Grad-CAM on CF (turbo, α=0.4)
  5. Attention Difference (RdBu_r, zero-centered)
  6. Signed δ (RdBu_r, symmetric)
  7. 10× Amplified δ
  8. |δ| Heatmap (inferno, p99)

References
----------
Selvaraju et al. (2017). Grad-CAM. ICCV.
Singla et al. (2023). Counterfactual explanations. MedIA.
Wang et al. (2023). Counterfactual-based saliency. ICCV.
IEEE TMI figure guidelines (2024).
Nature Research figure specifications (2025).
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional

from src.abc.config_abc import IMAGE_MEAN, IMAGE_STD

# ─────────────────────────────────────────────
# Publication matplotlib settings
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "pdf.fonttype": 42,       # TrueType (Nature requirement)
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────
# Grad-CAM computation
# ─────────────────────────────────────────────

class QuickGradCAM:
    """
    Lightweight Grad-CAM for generating heatmaps on arbitrary inputs.
    
    Unlike the full GradCAMExperiment, this is designed for inline use
    during counterfactual generation — no file I/O, minimal overhead.
    
    Reference: Selvaraju et al. (2017). Grad-CAM. ICCV 2017.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        self._hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Hook into the last conv layer of EfficientNet-B4."""
        target_layer = self.model.get_feature_layer()
        
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self._hook_handles.append(target_layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(bwd_hook))
    
    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
    
    @torch.enable_grad()
    def generate(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.
        
        Parameters
        ----------
        image_tensor : torch.Tensor (1, 3, H, W) on device
        target_class : int
        
        Returns
        -------
        np.ndarray (H, W) in [0, 1]
        """
        self.model.eval()
        inp = image_tensor.clone().detach().requires_grad_(True)
        
        logits = self.model(inp)
        score = logits[0, target_class]
        
        self.model.zero_grad()
        score.backward()
        
        if self.activations is None or self.gradients is None:
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))
        
        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        
        # Upsample to input resolution
        cam = F.interpolate(cam, size=image_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        return cam


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def denorm(t: torch.Tensor) -> np.ndarray:
    """Denormalize tensor (3, H, W) → uint8 (H, W, 3)."""
    mean = np.array(IMAGE_MEAN)
    std = np.array(IMAGE_STD)
    img = t.permute(1, 2, 0).numpy()
    return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)


def draw_contour(image: np.ndarray, mask: Optional[np.ndarray],
                 color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draw lesion boundary contour on image.
    
    Standard: green 2px solid contour (ISBI dermoscopy convention).
    """
    if mask is None:
        return image.copy()
    
    overlay = image.copy()
    binary = (mask > 0.5).astype(np.uint8) * 255
    
    # Resize mask to image size if needed
    if binary.shape[:2] != image.shape[:2]:
        binary = cv2.resize(binary, (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def gradcam_overlay(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on image.
    
    Standard: α=0.4 heatmap, 0.6 image (Selvaraju et al., 2017).
    Colormap: turbo (improved jet, Google 2019).
    """
    # Resize heatmap if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    blended = (alpha * colored.astype(float) + (1 - alpha) * image.astype(float))
    return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Main 8-panel visualization
# ─────────────────────────────────────────────

def save_8panel_figure(
    source_class_idx: int,
    results: List[Dict],
    classifier,
    device: torch.device,
    save_path: Path,
    title: str,
    max_rows: int = 10,
):
    """
    Generate publication-quality 8-column panel figure.
    
    Columns:
      1. Original + green lesion contour
      2. Grad-CAM on Original (turbo overlay, α=0.4)
      3. Counterfactual + green lesion contour
      4. Grad-CAM on CF (turbo overlay, α=0.4)
      5. Attention Difference map (RdBu_r, zero-centered)
      6. Signed δ (RdBu_r, symmetric vmin/vmax)
      7. 10× Amplified δ overlay
      8. |δ| Heatmap (inferno, p99 scaling)
    
    Parameters
    ----------
    results : list of dict
        Each dict from ABCCounterfactualExplainer.generate().
    classifier : nn.Module
        HAM10000 classifier for Grad-CAM.
    device : torch.device
    save_path : Path
    title : str
    max_rows : int
        Maximum number of rows (examples) to show.
    """
    results = results[:max_rows]
    n = len(results)
    if n == 0:
        return
    
    # Initialize Grad-CAM
    gcam = QuickGradCAM(classifier, device)
    
    col_headers = [
        "Original",
        "Grad-CAM\n(source class)",
        "Counterfactual",
        "Grad-CAM\n(source class on CF)",
        "Attention\nDifference",
        "Signed δ\n(RdBu_r)",
        "δ Overlay(red↑ blue↓)",
        "|δ| Heatmap",
    ]
    
    ncols = 8
    fig_w = 7.16 * 2  # double-column width × 2 for readability
    row_h = 1.8
    fig_h = row_h * n + 0.8  # rows + header space
    
    fig, axes = plt.subplots(n, ncols, figsize=(fig_w, fig_h))
    if n == 1:
        axes = axes[np.newaxis, :]
    
    for row, res in enumerate(results):
        # ── Prepare images ────────────────────
        orig_t = res["cf_tensor"] - res["delta"]  # (3, H, W)
        cf_t = res["cf_tensor"]                    # (3, H, W)
        delta_np = res["delta"].permute(1, 2, 0).numpy()  # (H, W, 3)
        
        orig_np = denorm(orig_t)
        cf_np = denorm(cf_t)
        
        # Get mask if available
        mask = res.get("mask", None)
        
        # ── Grad-CAM computation ──────────────
        src_cls = source_class_idx
        
        orig_gpu = orig_t.unsqueeze(0).to(device)
        cf_gpu = cf_t.unsqueeze(0).to(device)
        
        cam_orig = gcam.generate(orig_gpu, src_cls)
        cam_cf = gcam.generate(cf_gpu, src_cls)
        
        # ── Attention difference ──────────────
        attn_diff = cam_orig - cam_cf  # positive = more attention on original
        
        # ── Delta computations ────────────────
        diff_gray = delta_np.mean(axis=2)  # signed mean
        abs_delta = np.abs(delta_np).max(axis=2)  # absolute max
        
        vmax_diff = max(abs(diff_gray.min()), abs(diff_gray.max()), 0.01)
        p99 = np.percentile(abs_delta[abs_delta > 0], 99) if (abs_delta > 0).any() else 0.01
        vmax_heat = max(p99, 0.005)
        
        # Dynamic amplified δ overlay on original
        delta_max = np.abs(delta_np).max()
        amp_factor = min(0.4 / max(delta_max, 1e-6), 50.0)  # dynamic, cap at 50×
        # Overlay: original + amplified signed delta (red=increase, blue=decrease)
        orig_float = orig_np.astype(np.float32) / 255.0
        delta_rgb = np.zeros_like(orig_float)
        delta_mean = delta_np.mean(axis=2)
        pos = np.clip(delta_mean * amp_factor, 0, 1)   # increases → red
        neg = np.clip(-delta_mean * amp_factor, 0, 1)   # decreases → blue
        delta_rgb[:,:,0] = pos        # red channel
        delta_rgb[:,:,2] = neg        # blue channel
        overlay_amp = np.clip(orig_float * 0.6 + delta_rgb * 0.8, 0, 1)
        overlay_amp_uint8 = (overlay_amp * 255).astype(np.uint8)
        
        # ── COL 1: Original + contour ─────────
        axes[row, 0].imshow(draw_contour(orig_np, mask))
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title(col_headers[0], fontsize=7, fontweight="bold")
        
        # ── COL 2: Grad-CAM on Original ──────
        axes[row, 1].imshow(gradcam_overlay(orig_np, cam_orig))
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title(col_headers[1], fontsize=7, fontweight="bold")
        
        # ── COL 3: Counterfactual + contour ──
        valid_str = "✓" if res["validity"] else "✗"
        cf_label = f"P={res['final_prob']:.2f} {valid_str}"
        axes[row, 2].imshow(draw_contour(cf_np, mask))
        axes[row, 2].text(
            0.02, 0.98, cf_label, transform=axes[row, 2].transAxes,
            fontsize=6, color="white", fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )
        axes[row, 2].axis("off")
        if row == 0:
            axes[row, 2].set_title(col_headers[2], fontsize=7, fontweight="bold")
        
        # ── COL 4: Grad-CAM on CF ────────────
        axes[row, 3].imshow(gradcam_overlay(cf_np, cam_cf))
        axes[row, 3].axis("off")
        if row == 0:
            axes[row, 3].set_title(col_headers[3], fontsize=7, fontweight="bold")
        
        # ── COL 5: Attention Difference ──────
        vmax_attn = max(abs(attn_diff.min()), abs(attn_diff.max()), 0.01)
        norm_attn = mcolors.TwoSlopeNorm(vmin=-vmax_attn, vcenter=0, vmax=vmax_attn)
        im_attn = axes[row, 4].imshow(attn_diff, cmap="RdBu_r", norm=norm_attn)
        axes[row, 4].axis("off")
        if row == 0:
            axes[row, 4].set_title(col_headers[4], fontsize=7, fontweight="bold")
        if row == n - 1:  # colorbar only on last row
            cb = plt.colorbar(im_attn, ax=axes[row, 4], fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=5)
        
        # ── COL 6: Signed δ ─────────────────
        im_diff = axes[row, 5].imshow(diff_gray, cmap="RdBu_r",
                                       vmin=-vmax_diff, vmax=vmax_diff)
        # ABC scores overlay
        abc_text = f"ΔA={res['delta_A']:.3f}\nΔB={res['delta_B']:.3f}\nΔC={res['delta_C']:.3f}"
        axes[row, 5].text(
            0.02, 0.02, abc_text, transform=axes[row, 5].transAxes,
            fontsize=5, color="white", fontweight="bold",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
        )
        axes[row, 5].axis("off")
        if row == 0:
            axes[row, 5].set_title(col_headers[5], fontsize=7, fontweight="bold")
        if row == n - 1:
            cb = plt.colorbar(im_diff, ax=axes[row, 5], fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=5)
        
        # ── COL 7: δ overlay on original ─────
        axes[row, 6].imshow(overlay_amp_uint8)
        # L1 overlay
        l1_text = f"L1={res['proximity_l1']:.3f}"
        axes[row, 6].text(
            0.02, 0.98, l1_text, transform=axes[row, 6].transAxes,
            fontsize=5, color="white", fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
        )
        axes[row, 6].axis("off")
        if row == 0:
            axes[row, 6].set_title(col_headers[6], fontsize=7, fontweight="bold")
        
        # ── COL 8: |δ| Heatmap ──────────────
        im_heat = axes[row, 7].imshow(abs_delta, cmap="inferno",
                                       vmin=0, vmax=vmax_heat)
        sp_text = f"Sp={res['sparsity']:.2f}"
        axes[row, 7].text(
            0.02, 0.98, sp_text, transform=axes[row, 7].transAxes,
            fontsize=5, color="white", fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
        )
        axes[row, 7].axis("off")
        if row == 0:
            axes[row, 7].set_title(col_headers[7], fontsize=7, fontweight="bold")
        if row == n - 1:
            cb = plt.colorbar(im_heat, ax=axes[row, 7], fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=5)
    
    # Cleanup Grad-CAM hooks
    gcam.remove_hooks()
    
    plt.suptitle(title, fontsize=9, fontweight="bold", y=1.0)
    plt.subplots_adjust(wspace=0.03, hspace=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  [Vis] 8-panel saved: {save_path.name} ({n} rows)")
