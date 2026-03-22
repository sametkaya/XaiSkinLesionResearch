"""
src/explainers/abc_visualizer.py
---------------------------------
Clinical ABC visualization for counterfactual explanations.

For each image pair (original -> counterfactual), shows:
  Row 1: Original  | Asymmetry | Border | Color Map | Scores
  Row 2: CF        | Asymmetry | Border | Color Map | Scores

Asymmetry: Yellow bounding box + cross through center (NO overlay)
Border:    Lesion contour polygon colored by curvature + reference circle
Color:     Lesion pixels colored by detected dermoscopic color category
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.abc.config_abc import (
    IMAGE_MEAN, IMAGE_STD, DERMOSCOPIC_COLORS, IP_COLOR_THRESHOLD,
)

DERM_COLOR_RGB = {
    "black":       (30,  30,  30),
    "dark_brown":  (101, 67,  33),
    "light_brown": (181, 137, 82),
    "red":         (200, 50,  50),
    "blue_gray":   (100, 130, 160),
    "white":       (240, 240, 240),
}
DERM_COLOR_DISPLAY = {
    "black": "#1E1E1E", "dark_brown": "#654321", "light_brown": "#B58952",
    "red": "#C83232", "blue_gray": "#6482A0", "white": "#F0F0F0",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def _denorm(t):
    import torch
    mean = np.array(IMAGE_MEAN)
    std = np.array(IMAGE_STD)
    if isinstance(t, torch.Tensor):
        img = t.permute(1, 2, 0).cpu().numpy()
    else:
        img = t
    return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)


def _otsu_segment(image: np.ndarray) -> np.ndarray:
    """Otsu segmentation fallback for when no mask is available."""
    green = image[:, :, 1]
    blur = cv2.GaussianBlur(green, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        mask = (labels == best).astype(np.uint8) * 255
    return mask


def _get_mask_u8(mask, image: np.ndarray) -> np.ndarray:
    """Get binary mask. Uses Otsu if mask is None or covers >80% of image."""
    target_shape = image.shape
    if mask is not None:
        mask_u8 = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
        if mask_u8.shape[:2] != target_shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (target_shape[1], target_shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        # Check if mask covers >80% = probably all-ones (bad mask)
        coverage = mask_u8.sum() / 255 / (mask_u8.shape[0] * mask_u8.shape[1])
        if coverage < 0.80:
            return mask_u8
    # Fallback: Otsu segmentation
    return _otsu_segment(image)


# ----- A: Asymmetry -----
def viz_asymmetry(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Bounding box around the mole + cross through center.
    NO green/red overlay. Just geometric reference.
    """
    mask_u8 = _get_mask_u8(mask, image)
    canvas = image.copy()

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return canvas
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return canvas

    x, y, w, h = cv2.boundingRect(cnt)
    cx, cy = x + w // 2, y + h // 2

    # Yellow bounding rectangle
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 2, cv2.LINE_AA)

    # Cross from center to box edges
    cv2.line(canvas, (cx, y), (cx, y + h), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (x, cy), (x + w, cy), (255, 255, 0), 2, cv2.LINE_AA)

    # Center dot
    cv2.circle(canvas, (cx, cy), 4, (255, 255, 0), -1, cv2.LINE_AA)

    # Thin green lesion contour
    cv2.drawContours(canvas, [cnt], -1, (0, 200, 0), 1, cv2.LINE_AA)

    return canvas


# ----- B: Border -----
def viz_border(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Lesion contour polygon colored by local curvature.
    Green = smooth, Red = irregular.
    Blue dashed circle = reference perfect border.
    Background outside lesion is dimmed.
    """
    mask_u8 = _get_mask_u8(mask, image)
    mask_bool = mask_u8 > 127

    canvas = image.copy()
    canvas[~mask_bool] = (canvas[~mask_bool].astype(float) * 0.4).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return canvas
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 10:
        return canvas

    # Reference circle
    area = cv2.contourArea(cnt)
    radius = int(np.sqrt(area / np.pi))
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        for a in range(0, 360, 8):
            a1, a2 = np.deg2rad(a), np.deg2rad(a + 4)
            p1 = (int(cx + radius * np.cos(a1)), int(cy + radius * np.sin(a1)))
            p2 = (int(cx + radius * np.cos(a2)), int(cy + radius * np.sin(a2)))
            cv2.line(canvas, p1, p2, (100, 150, 255), 1, cv2.LINE_AA)

    pts = cnt.squeeze()
    N = len(pts)

    window = max(5, N // 20)
    curvature = np.zeros(N)
    for i in range(N):
        pp = pts[(i - window) % N].astype(float)
        pc = pts[i].astype(float)
        pn = pts[(i + window) % N].astype(float)
        v1, v2 = pp - pc, pn - pc
        cross_val = abs(v1[0] * v2[1] - v1[1] * v2[0])
        norm_val = max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8)
        curvature[i] = cross_val / norm_val

    c_max = max(np.percentile(curvature, 95), 1e-8)
    curv_n = np.clip(curvature / c_max, 0, 1)

    # Black outline
    for i in range(N - 1):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]), (0, 0, 0), 5, cv2.LINE_AA)

    # Curvature-colored contour
    for i in range(N - 1):
        c = curv_n[i]
        r = int(min(255, c * 2 * 255))
        g = int(min(255, (1 - c) * 2 * 255))
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (r, g, 0), 3, cv2.LINE_AA)
    c = curv_n[-1]
    cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]),
             (int(min(255, c * 2 * 255)), int(min(255, (1 - c) * 2 * 255)), 0),
             3, cv2.LINE_AA)

    return canvas


# ----- C: Color -----
def viz_color(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Set[str]]:
    mask_bool = _get_mask_u8(mask, image) > 127
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    canvas = (image.astype(np.float32) * 0.3).astype(np.uint8)
    color_map = np.zeros_like(image)
    detected = set()

    for name, ranges in DERMOSCOPIC_COLORS.items():
        h = hsv[:, :, 0].astype(int)
        s = hsv[:, :, 1].astype(int)
        v = hsv[:, :, 2].astype(int)
        in_r = ((h >= ranges["h"][0]) & (h <= ranges["h"][1]) &
                (s >= ranges["s"][0]) & (s <= ranges["s"][1]) &
                (v >= ranges["v"][0]) & (v <= ranges["v"][1]) & mask_bool)
        if "h_wrap" in ranges:
            hw = ranges["h_wrap"]
            in_r = in_r | ((h >= hw[0]) & (h <= hw[1]) &
                           (s >= ranges["s"][0]) & (s <= ranges["s"][1]) &
                           (v >= ranges["v"][0]) & (v <= ranges["v"][1]) & mask_bool)
        if in_r.sum() / max(mask_bool.sum(), 1) >= IP_COLOR_THRESHOLD:
            detected.add(name)
            color_map[in_r] = DERM_COLOR_RGB[name]

    lesion_m = mask_bool[:, :, np.newaxis]
    canvas = np.where(lesion_m,
                      (image.astype(float) * 0.4 + color_map.astype(float) * 0.6).astype(np.uint8),
                      canvas)
    no_color = mask_bool & (color_map.sum(axis=2) == 0)
    canvas[no_color] = image[no_color]
    return canvas, detected


# ----- Score bars -----
def draw_score_bars(ax, a, b, c, label=""):
    colors = ["#E24B4A", "#EF9F27", "#1D9E75"]
    ax.barh(range(3), [a, b, c], color=colors, height=0.6, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["A (Asymmetry)", "B (Border)", "C (Color)"], fontsize=6)
    ax.set_xlabel("Score", fontsize=6)
    ax.tick_params(axis='x', labelsize=5)
    ax.set_title(label, fontsize=7, fontweight="bold")
    for i, s in enumerate([a, b, c]):
        ax.text(min(s + 0.02, 0.95), i, f"{s:.3f}", va="center", fontsize=6)


# ----- Main panel -----
def save_abc_panel(
    results: List[Dict],
    abc_regressor,
    device,
    save_path: Path,
    title: str,
    max_rows: int = 5,
):
    import torch

    results = results[:max_rows]
    n = len(results)
    if n == 0:
        return

    ncols = 5
    nrows = n * 2
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
        mask_np = np.asarray(mask) if mask is not None else np.ones(orig_np.shape[:2], dtype=np.uint8)

        with torch.no_grad():
            orig_abc = abc_regressor(orig_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            cf_abc = abc_regressor(cf_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        row_orig = idx * 2
        row_cf = idx * 2 + 1

        for row_idx, img_np, abc_scores, label in [
            (row_orig, orig_np, orig_abc, "Original"),
            (row_cf, cf_np, cf_abc, "CF"),
        ]:
            m8 = _get_mask_u8(mask_np, img_np)

            # Col 0: Image + contour
            contoured = img_np.copy()
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contoured, cnts, -1, (0, 255, 0), 2)
            axes[row_idx, 0].imshow(contoured)
            axes[row_idx, 0].set_ylabel(label, fontsize=8, fontweight="bold",
                                         rotation=0, labelpad=40, va="center")
            axes[row_idx, 0].axis("off")

            # Col 1: Asymmetry
            axes[row_idx, 1].imshow(viz_asymmetry(img_np, mask_np))
            axes[row_idx, 1].axis("off")
            axes[row_idx, 1].text(0.5, 0.02, f"A={abc_scores[0]:.3f}",
                                   transform=axes[row_idx, 1].transAxes,
                                   fontsize=7, ha="center", va="bottom",
                                   color="white", fontweight="bold",
                                   bbox=dict(facecolor="black", alpha=0.6, pad=2))

            # Col 2: Border
            axes[row_idx, 2].imshow(viz_border(img_np, mask_np))
            axes[row_idx, 2].axis("off")
            axes[row_idx, 2].text(0.5, 0.02, f"B={abc_scores[1]:.3f}",
                                   transform=axes[row_idx, 2].transAxes,
                                   fontsize=7, ha="center", va="bottom",
                                   color="white", fontweight="bold",
                                   bbox=dict(facecolor="black", alpha=0.6, pad=2))

            # Col 3: Color
            color_viz, detected = viz_color(img_np, mask_np)
            axes[row_idx, 3].imshow(color_viz)
            axes[row_idx, 3].axis("off")
            color_str = ", ".join(sorted(detected)) if detected else "none"
            axes[row_idx, 3].text(0.5, 0.02, f"C={abc_scores[2]:.3f}\n{color_str}",
                                   transform=axes[row_idx, 3].transAxes,
                                   fontsize=5, ha="center", va="bottom",
                                   color="white", fontweight="bold",
                                   bbox=dict(facecolor="black", alpha=0.6, pad=2))

            # Col 4: Scores
            draw_score_bars(axes[row_idx, 4], abc_scores[0], abc_scores[1],
                           abc_scores[2], label=label)

        if idx == 0:
            for c, header in enumerate(col_headers):
                axes[0, c].set_title(header, fontsize=8, fontweight="bold")

    legend_patches = [mpatches.Patch(color=DERM_COLOR_DISPLAY[c], label=c.replace("_", " "))
                      for c in DERM_COLOR_DISPLAY]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=6,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(title, fontsize=10, fontweight="bold", y=1.0)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  [ABC-Vis] Panel saved: {save_path.name} ({n} examples)")