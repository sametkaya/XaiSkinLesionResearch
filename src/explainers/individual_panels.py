"""
src/explainers/individual_panels.py
------------------------------------
Generate one comprehensive XAI panel per image.

Layout (2 rows × 4 columns):
  Row 1: Original+Contour | Grad-CAM | A:Asymmetry | B:Border
  Row 2: Counterfactual   | CF Grad-CAM | C:Color Map | ABC Scores + CF Explanation

Each file is named: {src}_{tgt}_{image_id}_{mode}.png
At least 20 examples per class pair.

References:
  Selvaraju et al. (2017). Grad-CAM. ICCV.
  Stolz et al. (1994). ABCD rule of dermatoscopy.
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.abc.config_abc import (
    IMAGE_MEAN, IMAGE_STD, DERMOSCOPIC_COLORS, IP_COLOR_THRESHOLD,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _denorm(t):
    mean = np.array(IMAGE_MEAN)
    std = np.array(IMAGE_STD)
    if isinstance(t, torch.Tensor):
        img = t.permute(1, 2, 0).cpu().numpy()
    else:
        img = t
    return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)


def _otsu_segment(image: np.ndarray) -> np.ndarray:
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


def _get_mask(mask, image: np.ndarray) -> np.ndarray:
    if mask is not None:
        m = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
        if m.shape[:2] != image.shape[:2]:
            m = cv2.resize(m, (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        coverage = m.sum() / 255 / (m.shape[0] * m.shape[1])
        if coverage < 0.80:
            return m
    return _otsu_segment(image)


# ─────────────────────────────────────────────
# Grad-CAM (inline, lightweight)
# ─────────────────────────────────────────────

class _GradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.act = None
        self.grad = None
        self._hooks = []
        layer = model.get_feature_layer()
        self._hooks.append(layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'act', o.detach())))
        self._hooks.append(layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grad', go[0].detach())))

    @torch.enable_grad()
    def generate(self, img_t: torch.Tensor, cls: int) -> np.ndarray:
        self.model.eval()
        inp = img_t.clone().detach().requires_grad_(True)
        logits = self.model(inp)
        self.model.zero_grad()
        logits[0, cls].backward()
        if self.act is None or self.grad is None:
            return np.zeros((img_t.shape[2], img_t.shape[3]))
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=img_t.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        mx = cam.max()
        return cam / mx if mx > 0 else cam

    def remove(self):
        for h in self._hooks:
            h.remove()


def _overlay_cam(image: np.ndarray, heatmap: np.ndarray, alpha=0.4) -> np.ndarray:
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.clip(alpha * colored + (1 - alpha) * image, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# ABC visualizations (same as abc_visualizer)
# ─────────────────────────────────────────────

def _viz_asymmetry(image, mask_u8):
    canvas = image.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return canvas
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return canvas
    x, y, w, h = cv2.boundingRect(cnt)
    cx, cy = x + w // 2, y + h // 2
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, y), (cx, y + h), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (x, cy), (x + w, cy), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 4, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.drawContours(canvas, [cnt], -1, (0, 200, 0), 1, cv2.LINE_AA)
    return canvas


def _viz_border(image, mask_u8):
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
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        for a in range(0, 360, 8):
            a1, a2 = np.deg2rad(a), np.deg2rad(a + 4)
            cv2.line(canvas,
                     (int(cx + radius * np.cos(a1)), int(cy + radius * np.sin(a1))),
                     (int(cx + radius * np.cos(a2)), int(cy + radius * np.sin(a2))),
                     (100, 150, 255), 1, cv2.LINE_AA)
    pts = cnt.squeeze()
    N = len(pts)
    window = max(5, N // 20)
    curv = np.zeros(N)
    for i in range(N):
        pp, pc, pn = pts[(i-window)%N].astype(float), pts[i].astype(float), pts[(i+window)%N].astype(float)
        v1, v2 = pp - pc, pn - pc
        curv[i] = abs(v1[0]*v2[1]-v1[1]*v2[0]) / max(np.linalg.norm(v1)*np.linalg.norm(v2), 1e-8)
    cm = max(np.percentile(curv, 95), 1e-8)
    cn = np.clip(curv / cm, 0, 1)
    for i in range(N - 1):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]), (0, 0, 0), 5, cv2.LINE_AA)
    for i in range(N - 1):
        c = cn[i]
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]),
                 (int(min(255, c*2*255)), int(min(255, (1-c)*2*255)), 0), 3, cv2.LINE_AA)
    c = cn[-1]
    cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]),
             (int(min(255, c*2*255)), int(min(255, (1-c)*2*255)), 0), 3, cv2.LINE_AA)
    return canvas


DERM_RGB = {
    "black": (30,30,30), "dark_brown": (101,67,33), "light_brown": (181,137,82),
    "red": (200,50,50), "blue_gray": (100,130,160), "white": (240,240,240),
}

def _viz_color(image, mask_u8):
    mask_bool = mask_u8 > 127
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    canvas = (image.astype(float) * 0.3).astype(np.uint8)
    cmap = np.zeros_like(image)
    detected = set()
    for name, rng in DERMOSCOPIC_COLORS.items():
        h, s, v = hsv[:,:,0].astype(int), hsv[:,:,1].astype(int), hsv[:,:,2].astype(int)
        in_r = ((h>=rng["h"][0])&(h<=rng["h"][1])&(s>=rng["s"][0])&(s<=rng["s"][1])&
                (v>=rng["v"][0])&(v<=rng["v"][1])&mask_bool)
        if "h_wrap" in rng:
            hw = rng["h_wrap"]
            in_r = in_r | ((h>=hw[0])&(h<=hw[1])&(s>=rng["s"][0])&(s<=rng["s"][1])&
                           (v>=rng["v"][0])&(v<=rng["v"][1])&mask_bool)
        if in_r.sum() / max(mask_bool.sum(), 1) >= IP_COLOR_THRESHOLD:
            detected.add(name)
            cmap[in_r] = DERM_RGB[name]
    lm = mask_bool[:,:,np.newaxis]
    canvas = np.where(lm, (image.astype(float)*0.4 + cmap.astype(float)*0.6).astype(np.uint8), canvas)
    canvas[mask_bool & (cmap.sum(axis=2)==0)] = image[mask_bool & (cmap.sum(axis=2)==0)]
    return canvas, detected


# ─────────────────────────────────────────────
# Main: generate individual panels
# ─────────────────────────────────────────────

def generate_individual_panels(
    results: List[Dict],
    classifier,
    abc_regressor,
    device: torch.device,
    output_dir: Path,
    src_name: str,
    tgt_name: str,
    mode: str = "ABC",
    class_labels: List[str] = None,
):
    """
    Generate one comprehensive panel per image.

    Layout (2 rows × 4 columns):
      Row 1: Original+Contour | Grad-CAM(src) | A:Asymmetry | B:Border
      Row 2: Counterfactual   | Grad-CAM(CF)  | C:Color Map | Scores+Text

    Each file: {src}_to_{tgt}_{image_idx:03d}_{mode}.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gcam = _GradCAM(classifier, device)
    abc_regressor.eval()

    if class_labels is None:
        from src import config
        class_labels = config.CLASS_LABELS

    src_idx = class_labels.index(src_name)
    tgt_idx = class_labels.index(tgt_name)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 200,
    })

    for i, res in enumerate(results):
        orig_t = res["cf_tensor"] - res["delta"]
        cf_t = res["cf_tensor"]
        mask_raw = res.get("mask", None)

        orig_np = _denorm(orig_t)
        cf_np = _denorm(cf_t)
        mask_u8 = _get_mask(mask_raw, orig_np)

        # Grad-CAM
        orig_gpu = orig_t.unsqueeze(0).to(device)
        cf_gpu = cf_t.unsqueeze(0).to(device)
        cam_orig = gcam.generate(orig_gpu, src_idx)
        cam_cf = gcam.generate(cf_gpu, src_idx)

        # ABC scores
        with torch.no_grad():
            abc_orig = abc_regressor(orig_gpu).squeeze().cpu().numpy()
            abc_cf = abc_regressor(cf_gpu).squeeze().cpu().numpy()

        # ── Build figure ──────────────────────
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # ── Row 1: Original ───────────────────
        # (0,0) Original + contour
        img_c = orig_np.copy()
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_c, cnts, -1, (0, 255, 0), 2)
        axes[0, 0].imshow(img_c)
        axes[0, 0].set_title(f"Original ({src_name})\nP({src_name})={res.get('src_prob', '?')}", fontsize=8, fontweight="bold")
        axes[0, 0].axis("off")

        # (0,1) Grad-CAM on original
        axes[0, 1].imshow(_overlay_cam(orig_np, cam_orig))
        axes[0, 1].set_title(f"Grad-CAM ({src_name})", fontsize=8, fontweight="bold")
        axes[0, 1].axis("off")

        # (0,2) Asymmetry
        axes[0, 2].imshow(_viz_asymmetry(orig_np, mask_u8))
        axes[0, 2].set_title(f"A: Asymmetry = {abc_orig[0]:.3f}", fontsize=8, fontweight="bold")
        axes[0, 2].axis("off")

        # (0,3) Border
        axes[0, 3].imshow(_viz_border(orig_np, mask_u8))
        axes[0, 3].set_title(f"B: Border = {abc_orig[1]:.3f}", fontsize=8, fontweight="bold")
        axes[0, 3].axis("off")

        # ── Row 2: Counterfactual ─────────────
        # (1,0) Counterfactual + contour
        cf_c = cf_np.copy()
        cv2.drawContours(cf_c, cnts, -1, (0, 255, 0), 2)
        axes[1, 0].imshow(cf_c)
        valid = "✓" if res["validity"] else "✗"
        axes[1, 0].set_title(f"Counterfactual ({tgt_name}) {valid}\nP({tgt_name})={res['final_prob']:.3f}", fontsize=8, fontweight="bold")
        axes[1, 0].axis("off")

        # (1,1) Grad-CAM on CF
        axes[1, 1].imshow(_overlay_cam(cf_np, cam_cf))
        axes[1, 1].set_title(f"Grad-CAM on CF ({src_name})", fontsize=8, fontweight="bold")
        axes[1, 1].axis("off")

        # (1,2) Color Map
        color_viz, detected = _viz_color(orig_np, mask_u8)
        axes[1, 2].imshow(color_viz)
        color_str = ", ".join(sorted(detected)) if detected else "none"
        axes[1, 2].set_title(f"C: Color = {abc_orig[2]:.3f}\n{color_str}", fontsize=7, fontweight="bold")
        axes[1, 2].axis("off")

        # (1,3) Scores + CF explanation text
        ax_score = axes[1, 3]
        ax_score.clear()

        # Score comparison bars
        y_pos = np.arange(3)
        bar_h = 0.35
        labels_abc = ["A (Asymmetry)", "B (Border)", "C (Color)"]
        orig_scores = [abc_orig[0], abc_orig[1], abc_orig[2]]
        cf_scores = [abc_cf[0], abc_cf[1], abc_cf[2]]

        bars1 = ax_score.barh(y_pos - bar_h/2, orig_scores, bar_h,
                               label="Original", color=["#E24B4A", "#EF9F27", "#1D9E75"], alpha=0.7)
        bars2 = ax_score.barh(y_pos + bar_h/2, cf_scores, bar_h,
                               label="CF", color=["#E24B4A", "#EF9F27", "#1D9E75"], alpha=0.35,
                               edgecolor=["#E24B4A", "#EF9F27", "#1D9E75"], linewidth=1.5)

        ax_score.set_xlim(0, max(max(orig_scores), max(cf_scores)) * 1.4 + 0.01)
        ax_score.set_yticks(y_pos)
        ax_score.set_yticklabels(labels_abc, fontsize=7)
        ax_score.tick_params(axis='x', labelsize=6)

        for j, (o, c) in enumerate(zip(orig_scores, cf_scores)):
            delta = c - o
            sign = "+" if delta >= 0 else ""
            ax_score.text(max(o, c) + 0.003, j,
                         f"{o:.3f}→{c:.3f} ({sign}{delta:.3f})",
                         va="center", fontsize=6)

        ax_score.legend(fontsize=6, loc="lower right")
        ax_score.set_title("ABC Scores: Original vs CF", fontsize=8, fontweight="bold")

        # Counterfactual explanation text
        delta_A = abs(abc_cf[0] - abc_orig[0])
        delta_B = abs(abc_cf[1] - abc_orig[1])
        delta_C = abs(abc_cf[2] - abc_orig[2])

        explanation_lines = [
            f"Transition: {src_name} → {tgt_name} (mode={mode})",
            f"L1={res['proximity_l1']:.4f}  Sparsity={res['sparsity']:.3f}",
            f"ΔA={delta_A:.4f}  ΔB={delta_B:.4f}  ΔC={delta_C:.4f}",
        ]

        # Add clinical interpretation
        max_delta = max(delta_A, delta_B, delta_C)
        if max_delta == delta_A:
            explanation_lines.append("→ Asymmetry changed most")
        elif max_delta == delta_B:
            explanation_lines.append("→ Border regularity changed most")
        else:
            explanation_lines.append("→ Color variegation changed most")

        fig.text(0.75, 0.02, "\n".join(explanation_lines),
                fontsize=7, family="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.8, pad=5),
                va="bottom", ha="center")

        # Title
        fig.suptitle(
            f"{src_name} → {tgt_name} | mode={mode} | Example {i+1}/{len(results)}",
            fontsize=11, fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        fname = f"{src_name}_to_{tgt_name}_{i+1:03d}_{mode}.png"
        fig.savefig(output_dir / fname, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    gcam.remove()
    print(f"  [Individual] {len(results)} panels saved to {output_dir.name}/")
