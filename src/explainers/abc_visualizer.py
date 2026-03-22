"""
src/explainers/abc_visualizer.py
---------------------------------
Clinical ABC visualization for counterfactual explanations.

For each image pair (original → counterfactual), shows:
  Row 1: Original  | Asymmetry | Border | Color Map | Scores
  Row 2: CF        | Asymmetry | Border | Color Map | Scores

Asymmetry: Principal axis + reflected mask overlay (red = asymmetric region)
Border:    Contour colored by local curvature (green=smooth, red=irregular)
Color:     Lesion pixels colored by detected dermoscopic color category

References
----------
Stolz, W. et al. (1994). ABCD rule of dermatoscopy.
Argenziano, G. et al. (1998). Dermoscopic colors.
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from src.abc.config_abc import (
    IMAGE_MEAN, IMAGE_STD, DERMOSCOPIC_COLORS, IP_COLOR_THRESHOLD,
)


# ─────────────────────────────────────────────
# Color definitions for visualization
# ─────────────────────────────────────────────
DERM_COLOR_RGB = {
    "black":       (30,  30,  30),
    "dark_brown":  (101, 67,  33),
    "light_brown": (181, 137, 82),
    "red":         (200, 50,  50),
    "blue_gray":   (100, 130, 160),
    "white":       (240, 240, 240),
}

DERM_COLOR_DISPLAY = {
    "black":       "#1E1E1E",
    "dark_brown":  "#654321",
    "light_brown": "#B58952",
    "red":         "#C83232",
    "blue_gray":   "#6482A0",
    "white":       "#F0F0F0",
}


# ─────────────────────────────────────────────
# Helper: denormalize
# ─────────────────────────────────────────────
def _denorm(t):
    import torch
    mean = np.array(IMAGE_MEAN)
    std = np.array(IMAGE_STD)
    if isinstance(t, torch.Tensor):
        img = t.permute(1, 2, 0).cpu().numpy()
    else:
        img = t
    return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# A — Asymmetry Visualization
# ─────────────────────────────────────────────
def viz_asymmetry(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Visualize asymmetry by showing reflected mask overlap.
    
    Green  = symmetric (overlap between original and reflected mask)
    Red    = asymmetric (non-overlapping regions)
    
    Also draws the two principal axes of the lesion.
    """
    H, W = mask.shape[:2]
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    
    # Resize mask to image size
    if mask_u8.shape[:2] != image.shape[:2]:
        mask_u8 = cv2.resize(mask_u8, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    
    mask_bool = mask_u8 > 127
    canvas = image.copy()
    
    # Find contour and fit ellipse for principal axes
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return canvas
    
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return canvas
    
    # Fit ellipse for axes
    if len(cnt) >= 5:
        (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
    else:
        M = cv2.moments(mask_u8)
        if M["m00"] == 0:
            return canvas
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        angle = 0.0
        ma = mi = 100
    
    # Rotate mask to align principal axis
    ih, iw = image.shape[:2]
    centre = (iw / 2, ih / 2)
    rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(mask_u8, rot_mat, (iw, ih),
                              flags=cv2.INTER_NEAREST).astype(bool)
    
    # Reflect along both axes
    flip_h = np.flip(rotated, axis=0)
    flip_v = np.flip(rotated, axis=1)
    
    # Symmetric = overlap, Asymmetric = difference
    sym_h = np.logical_and(rotated, flip_h)
    asym_h = np.logical_xor(rotated, flip_h)
    sym_v = np.logical_and(rotated, flip_v)
    asym_v = np.logical_xor(rotated, flip_v)
    
    # Rotate back
    inv_rot = cv2.getRotationMatrix2D(centre, -angle, 1.0)
    sym_back = cv2.warpAffine(
        (sym_h & sym_v).astype(np.uint8) * 255,
        inv_rot, (iw, ih), flags=cv2.INTER_NEAREST
    ) > 127
    asym_back = cv2.warpAffine(
        (asym_h | asym_v).astype(np.uint8) * 255,
        inv_rot, (iw, ih), flags=cv2.INTER_NEAREST
    ) > 127
    
    # Overlay
    overlay = canvas.astype(np.float32)
    overlay[sym_back] = overlay[sym_back] * 0.6 + np.array([0, 180, 0], dtype=np.float32) * 0.4
    overlay[asym_back] = overlay[asym_back] * 0.6 + np.array([220, 40, 40], dtype=np.float32) * 0.4
    canvas = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Draw principal axes through lesion centroid
    # Use contour bounding to scale axis length
    x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
    ax_len = int(max(w_b, h_b) * 0.6)
    rad = np.deg2rad(angle)
    dx1, dy1 = int(ax_len * np.cos(rad)), int(ax_len * np.sin(rad))
    dx2, dy2 = int(ax_len * -np.sin(rad)), int(ax_len * np.cos(rad))
    # cx, cy = lesion centroid from fitEllipse
    c = (int(cx), int(cy))
    cv2.line(canvas, (c[0]-dx1, c[1]-dy1), (c[0]+dx1, c[1]+dy1), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (c[0]-dx2, c[1]-dy2), (c[0]+dx2, c[1]+dy2), (255, 255, 0), 2, cv2.LINE_AA)
    # Mark centroid
    cv2.circle(canvas, c, 3, (255, 255, 0), -1, cv2.LINE_AA)
    
    return canvas


# ─────────────────────────────────────────────
# B — Border Irregularity Visualization
# ─────────────────────────────────────────────
def viz_border(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Visualize border irregularity by coloring the contour by local curvature.
    
    Green = smooth (low curvature)
    Yellow = moderate
    Red = irregular (high curvature)
    """
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    
    if mask_u8.shape[:2] != image.shape[:2]:
        mask_u8 = cv2.resize(mask_u8, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    
    canvas = image.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return canvas
    
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 10:
        return canvas
    
    pts = cnt.squeeze()  # (N, 2)
    N = len(pts)
    
    # Compute local curvature (angle change over sliding window)
    window = max(5, N // 20)
    curvature = np.zeros(N)
    
    for i in range(N):
        p_prev = pts[(i - window) % N]
        p_curr = pts[i]
        p_next = pts[(i + window) % N]
        
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        dot = np.dot(v1, v2)
        cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
        norm = max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8)
        
        curvature[i] = cross / norm  # sin of angle ≈ curvature
    
    # Normalize curvature to [0, 1]
    c_max = max(np.percentile(curvature, 95), 1e-8)
    curvature_norm = np.clip(curvature / c_max, 0, 1)
    
    # Draw black outline first (glow effect for visibility)
    for i in range(N - 1):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), (0, 0, 0), 6, cv2.LINE_AA)
    cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]), (0, 0, 0), 6, cv2.LINE_AA)
    # Draw colored contour on top (curvature gradient)
    for i in range(N - 1):
        c = curvature_norm[i]
        r = int(min(255, c * 2 * 255))
        g = int(min(255, (1 - c) * 2 * 255))
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), (r, g, 0), 3, cv2.LINE_AA)
    c = curvature_norm[-1]
    r = int(min(255, c * 2 * 255))
    g = int(min(255, (1 - c) * 2 * 255))
    cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]), (r, g, 0), 3, cv2.LINE_AA)
    
    return canvas


# ─────────────────────────────────────────────
# C — Color Variegation Visualization
# ─────────────────────────────────────────────
def viz_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Visualize detected dermoscopic colors on the lesion.
    
    Each lesion pixel is assigned its dominant dermoscopic color category
    and colored accordingly. Non-lesion pixels are shown dimmed.
    """
    mask_bool = (mask > 0.5)
    
    if mask_bool.shape[:2] != image.shape[:2]:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8),
                               (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Create color-coded output
    canvas = (image.astype(np.float32) * 0.3).astype(np.uint8)  # dimmed background
    
    # For each pixel in lesion, find dominant dermoscopic color
    color_map = np.zeros_like(image)
    detected_colors = set()
    
    for color_name, ranges in DERMOSCOPIC_COLORS.items():
        h_lo, h_hi = ranges["h"]
        s_lo, s_hi = ranges["s"]
        v_lo, v_hi = ranges["v"]
        
        h = hsv[:, :, 0].astype(int)
        s = hsv[:, :, 1].astype(int)
        v = hsv[:, :, 2].astype(int)
        
        in_range = ((h >= h_lo) & (h <= h_hi) &
                    (s >= s_lo) & (s <= s_hi) &
                    (v >= v_lo) & (v <= v_hi) &
                    mask_bool)
        
        # Handle h_wrap for red
        if "h_wrap" in ranges:
            hw_lo, hw_hi = ranges["h_wrap"]
            in_wrap = ((h >= hw_lo) & (h <= hw_hi) &
                       (s >= s_lo) & (s <= s_hi) &
                       (v >= v_lo) & (v <= v_hi) &
                       mask_bool)
            in_range = in_range | in_wrap
        
        frac = in_range.sum() / max(mask_bool.sum(), 1)
        if frac >= IP_COLOR_THRESHOLD:
            detected_colors.add(color_name)
            rgb = DERM_COLOR_RGB[color_name]
            color_map[in_range] = rgb
    
    # Blend: lesion pixels get color overlay, background stays dimmed
    lesion_colored = mask_bool[:, :, np.newaxis]
    canvas = np.where(lesion_colored,
                      (image.astype(float) * 0.4 + color_map.astype(float) * 0.6).astype(np.uint8),
                      canvas)
    
    # Uncolored lesion pixels (no detected dermoscopic color) show original
    no_color = mask_bool & (color_map.sum(axis=2) == 0)
    canvas[no_color] = image[no_color]
    
    return canvas, detected_colors


# ─────────────────────────────────────────────
# Score bar visualization
# ─────────────────────────────────────────────
def draw_score_bars(ax, a_score, b_score, c_score, label=""):
    """Draw horizontal ABC score bars."""
    scores = [a_score, b_score, c_score]
    names = ["A (Asymmetry)", "B (Border)", "C (Color)"]
    colors = ["#E24B4A", "#EF9F27", "#1D9E75"]
    
    ax.barh(range(3), scores, color=colors, height=0.6, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Score", fontsize=6)
    ax.tick_params(axis='x', labelsize=5)
    ax.set_title(label, fontsize=7, fontweight="bold")
    
    for i, (s, n) in enumerate(zip(scores, names)):
        ax.text(min(s + 0.02, 0.95), i, f"{s:.3f}", va="center", fontsize=6)


# ─────────────────────────────────────────────
# Main ABC panel figure
# ─────────────────────────────────────────────
def save_abc_panel(
    results: List[Dict],
    abc_regressor,
    device,
    save_path: Path,
    title: str,
    max_rows: int = 5,
):
    """
    Generate ABC clinical visualization panel.
    
    For each example, shows 2 rows:
      Row A: Original  | Asymmetry | Border | Color Map | ABC Scores
      Row B: CF        | Asymmetry | Border | Color Map | ABC Scores
    
    Parameters
    ----------
    results : list of dict from counterfactual generation
    abc_regressor : trained ABC model
    device : torch.device
    save_path : Path
    title : str
    max_rows : int
        Number of example pairs to show
    """
    import torch
    
    results = results[:max_rows]
    n = len(results)
    if n == 0:
        return
    
    ncols = 5  # Image | Asymmetry | Border | Color | Scores
    nrows = n * 2  # 2 rows per example (orig + CF)
    
    fig_w = 14
    fig_h = 2.2 * nrows + 0.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                              gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.6]})
    if nrows == 2:
        axes = axes.reshape(2, ncols)
    
    col_headers = ["Lesion Image", "A: Asymmetry", "B: Border", "C: Color Map", "ABC Scores"]
    
    abc_regressor.eval()
    
    for idx, res in enumerate(results):
        orig_t = res["cf_tensor"] - res["delta"]
        cf_t = res["cf_tensor"]
        mask = res.get("mask", None)
        
        orig_np = _denorm(orig_t)
        cf_np = _denorm(cf_t)
        
        # Get mask as binary numpy
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_np = (mask > 0.5).astype(np.uint8)
            else:
                mask_np = np.ones(orig_np.shape[:2], dtype=np.uint8)
        else:
            mask_np = np.ones(orig_np.shape[:2], dtype=np.uint8)
        
        # Compute ABC scores via regressor
        with torch.no_grad():
            orig_abc = abc_regressor(orig_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            cf_abc = abc_regressor(cf_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        
        row_orig = idx * 2
        row_cf = idx * 2 + 1
        
        for row_idx, img_np, abc_scores, label in [
            (row_orig, orig_np, orig_abc, "Original"),
            (row_cf, cf_np, cf_abc, "CF"),
        ]:
            # Col 0: Image + contour
            contoured = img_np.copy()
            contour_mask = (mask_np * 255).astype(np.uint8)
            if contour_mask.shape[:2] != img_np.shape[:2]:
                contour_mask = cv2.resize(contour_mask, (img_np.shape[1], img_np.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contoured, cnts, -1, (0, 255, 0), 2)
            
            axes[row_idx, 0].imshow(contoured)
            axes[row_idx, 0].set_ylabel(label, fontsize=8, fontweight="bold", rotation=0,
                                         labelpad=40, va="center")
            axes[row_idx, 0].axis("off")
            
            # Col 1: Asymmetry
            asym_viz = viz_asymmetry(img_np, mask_np)
            axes[row_idx, 1].imshow(asym_viz)
            axes[row_idx, 1].axis("off")
            axes[row_idx, 1].text(0.5, 0.02, f"A={abc_scores[0]:.3f}",
                                   transform=axes[row_idx, 1].transAxes,
                                   fontsize=7, ha="center", va="bottom",
                                   color="white", fontweight="bold",
                                   bbox=dict(facecolor="black", alpha=0.6, pad=2))
            
            # Col 2: Border
            border_viz = viz_border(img_np, mask_np)
            axes[row_idx, 2].imshow(border_viz)
            axes[row_idx, 2].axis("off")
            axes[row_idx, 2].text(0.5, 0.02, f"B={abc_scores[1]:.3f}",
                                   transform=axes[row_idx, 2].transAxes,
                                   fontsize=7, ha="center", va="bottom",
                                   color="white", fontweight="bold",
                                   bbox=dict(facecolor="black", alpha=0.6, pad=2))
            
            # Col 3: Color Map
            color_viz, detected = viz_color(img_np, mask_np)
            axes[row_idx, 3].imshow(color_viz)
            axes[row_idx, 3].axis("off")
            color_str = ", ".join(sorted(detected)) if detected else "none"
            axes[row_idx, 3].text(0.5, 0.02, f"C={abc_scores[2]:.3f}\n{color_str}",
                                   transform=axes[row_idx, 3].transAxes,
                                   fontsize=5, ha="center", va="bottom",
                                   color="white", fontweight="bold",
                                   bbox=dict(facecolor="black", alpha=0.6, pad=2))
            
            # Col 4: Score bars
            draw_score_bars(axes[row_idx, 4], abc_scores[0], abc_scores[1], abc_scores[2],
                           label=f"{label}")
        
        # Draw separator line between examples
        if idx < n - 1:
            for c in range(ncols):
                axes[row_cf, c].axhline(y=0, color="gray", linewidth=0.5)
        
        # Add column headers on first pair only
        if idx == 0:
            for c, header in enumerate(col_headers):
                axes[0, c].set_title(header, fontsize=8, fontweight="bold")
    
    # Add color legend
    legend_patches = [mpatches.Patch(color=DERM_COLOR_DISPLAY[c], label=c.replace("_", " "))
                      for c in DERM_COLOR_DISPLAY]
    legend_patches.append(mpatches.Patch(color="green", label="Symmetric"))
    legend_patches.append(mpatches.Patch(color="red", label="Asymmetric"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=6,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(title, fontsize=10, fontweight="bold", y=1.0)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  [ABC-Vis] Panel saved: {save_path.name} ({n} examples)")
