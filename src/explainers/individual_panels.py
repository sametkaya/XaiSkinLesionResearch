"""
src/explainers/individual_panels.py
------------------------------------
Generate one comprehensive XAI panel per lesion image.

Layout (2 rows x 4 columns):
  Row 1: Original+Contour | A:Asymmetry | B:Border   | C:Color Map
  Row 2: Grad-CAM         | ABC Profile | Ablation   | CF Narrative

Requires all 4 ablation mode results for the same image.
Each file: {src}_to_{tgt}_{idx:03d}.png
"""

import cv2
import textwrap
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 200,
})


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


def _otsu_segment(image):
    green = image[:, :, 1]
    blur = cv2.GaussianBlur(green, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        mask = (labels == best).astype(np.uint8) * 255
    return mask


def _get_mask(mask, image):
    if mask is not None:
        m = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
        if m.shape[:2] != image.shape[:2]:
            m = cv2.resize(m, (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
            m = (m > 127).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if n_labels > 2:
            best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            m = (labels == best).astype(np.uint8) * 255
        coverage = m.sum() / 255 / (m.shape[0] * m.shape[1])
        if coverage < 0.80:
            return m
    return _otsu_segment(image)


# ─────────────────────────────────────────────
# Grad-CAM (inline)
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
    def generate(self, img_t, cls):
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


def _overlay_cam(image, heatmap, alpha=0.4):
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.clip(alpha * colored + (1 - alpha) * image, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# ABC Visualizations
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
# Main function
# ─────────────────────────────────────────────

def generate_individual_panels(
    all_records: Dict[str, List[Dict]],
    classifier,
    abc_regressor,
    device: torch.device,
    output_dir: Path,
    src_name: str,
    tgt_name: str,
    n_images: int = 20,
    class_labels: List[str] = None,
):
    """
    Generate one comprehensive panel per lesion.

    Parameters
    ----------
    all_records : dict
        Keys: mode names ('baseline', 'A_only', 'AB', 'ABC')
        Values: list of result dicts (same image order across modes)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gcam = _GradCAM(classifier, device)
    abc_regressor.eval()

    if class_labels is None:
        from src import config
        class_labels = config.CLASS_LABELS

    src_idx = class_labels.index(src_name)

    modes = ["baseline", "A_only", "AB", "ABC"]
    n = min(n_images, min(len(all_records.get(m, [])) for m in modes))
    if n == 0:
        print(f"  [Individual] No records for {src_name}->{tgt_name}, skipping")
        gcam.remove()
        return

    for i in range(n):
        mode_results = {m: all_records[m][i] for m in modes if i < len(all_records[m])}
        if len(mode_results) < 4:
            continue

        res = mode_results["ABC"]
        orig_t = res["cf_tensor"] - res["delta"]
        orig_np = _denorm(orig_t)
        mask_u8 = _get_mask(res.get("mask", None), orig_np)

        # Grad-CAM
        orig_gpu = orig_t.unsqueeze(0).to(device)
        cam_orig = gcam.generate(orig_gpu, src_idx)

        # ABC scores
        with torch.no_grad():
            abc_orig = abc_regressor(orig_gpu).squeeze().cpu().numpy()

        # ── Build 2x4 figure ─────────────────
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # ═══ ROW 1 ═══════════════════════════
        # (0,0) Original + contour
        img_c = orig_np.copy()
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_c, cnts, -1, (0, 255, 0), 2)
        axes[0, 0].imshow(img_c)
        axes[0, 0].set_title(
            f"Original ({src_name})\nP({src_name})={res.get('src_prob', '?')}",
            fontsize=9, fontweight="bold")
        axes[0, 0].axis("off")

        # (0,1) Asymmetry
        axes[0, 1].imshow(_viz_asymmetry(orig_np, mask_u8))
        axes[0, 1].set_title(f"A: Asymmetry = {abc_orig[0]:.3f}", fontsize=9, fontweight="bold")
        axes[0, 1].axis("off")

        # (0,2) Border
        axes[0, 2].imshow(_viz_border(orig_np, mask_u8))
        axes[0, 2].set_title(f"B: Border = {abc_orig[1]:.3f}", fontsize=9, fontweight="bold")
        axes[0, 2].axis("off")

        # (0,3) Color Map
        color_viz, detected = _viz_color(orig_np, mask_u8)
        axes[0, 3].imshow(color_viz)
        color_str = ", ".join(sorted(detected)) if detected else "none"
        axes[0, 3].set_title(f"C: Color = {abc_orig[2]:.3f}\n{color_str}",
                              fontsize=8, fontweight="bold")
        axes[0, 3].axis("off")

        # ═══ ROW 2 ═══════════════════════════
        # (1,0) Grad-CAM
        axes[1, 0].imshow(_overlay_cam(orig_np, cam_orig))
        axes[1, 0].set_title(f"Grad-CAM ({src_name})", fontsize=9, fontweight="bold")
        axes[1, 0].axis("off")

        # (1,1) ABC Profile
        ax_abc = axes[1, 1]
        colors_abc = ["#E24B4A", "#EF9F27", "#1D9E75"]
        labels_abc = ["A (Asymmetry)", "B (Border)", "C (Color)"]
        scores = [abc_orig[0], abc_orig[1], abc_orig[2]]
        ax_abc.barh(range(3), scores, color=colors_abc, height=0.6, alpha=0.85)
        ax_abc.set_xlim(0, max(max(scores) * 1.3, 0.05))
        ax_abc.set_yticks(range(3))
        ax_abc.set_yticklabels(labels_abc, fontsize=8)
        ax_abc.tick_params(axis='x', labelsize=7)
        for j, s in enumerate(scores):
            ax_abc.text(s + 0.002, j, f"{s:.3f}", va="center", fontsize=8, fontweight="bold")
        ax_abc.set_title("ABC Profile", fontsize=9, fontweight="bold")
        ax_abc.grid(axis='x', alpha=0.3)

        # (1,2) Ablation comparison
        ax_abl = axes[1, 2]
        mode_display = ["Baseline", "+A", "+A+B", "+A+B+C"]
        x_pos = np.arange(4)
        bar_w = 0.25
        delta_A = [mode_results[m].get("delta_A", 0) for m in modes]
        delta_B = [mode_results[m].get("delta_B", 0) for m in modes]
        delta_C = [mode_results[m].get("delta_C", 0) for m in modes]
        ax_abl.bar(x_pos - bar_w, delta_A, bar_w, label=u"\u0394A", color="#E24B4A", alpha=0.85)
        ax_abl.bar(x_pos,         delta_B, bar_w, label=u"\u0394B", color="#EF9F27", alpha=0.85)
        ax_abl.bar(x_pos + bar_w, delta_C, bar_w, label=u"\u0394C", color="#1D9E75", alpha=0.85)
        ax_abl.set_xticks(x_pos)
        ax_abl.set_xticklabels(mode_display, fontsize=8)
        ax_abl.set_ylabel(u"\u0394 (change)", fontsize=8)
        ax_abl.legend(fontsize=7, loc="upper right")
        ax_abl.set_title("Ablation: ABC Preservation", fontsize=9, fontweight="bold")
        ax_abl.grid(axis='y', alpha=0.3)
        for mi, m in enumerate(modes):
            r = mode_results[m]
            v = u"\u2713" if r["validity"] else u"\u2717"
            ax_abl.text(mi, ax_abl.get_ylim()[1] * 0.92,
                       f"{v} L1={r['proximity_l1']:.3f}",
                       ha="center", fontsize=6, color="gray")

        # (1,3) CF Narrative
        ax_txt = axes[1, 3]
        ax_txt.axis("off")
        bl_res = mode_results["baseline"]
        abc_res = mode_results["ABC"]
        lines = [
            f"Transition: {src_name} -> {tgt_name}",
            f"P({tgt_name}) = {abc_res['final_prob']:.3f}  "
            f"{'Valid' if abc_res['validity'] else 'Invalid'}",
            "",
            "Counterfactual Explanation:",
            f"  L1 = {bl_res['proximity_l1']:.4f}",
            "",
            "ABC Preservation Effect:",
        ]
        bl_total = bl_res["delta_A"] + bl_res["delta_B"] + bl_res["delta_C"]
        abc_total = abc_res["delta_A"] + abc_res["delta_B"] + abc_res["delta_C"]
        if bl_total > 0:
            reduction = (1 - abc_total / bl_total) * 100
            lines.append(f"  Total delta reduced {reduction:.0f}%")
        deltas = {"A": bl_res["delta_A"], "B": bl_res["delta_B"], "C": bl_res["delta_C"]}
        max_c = max(deltas, key=deltas.get)
        cnames = {"A": "Asymmetry", "B": "Border", "C": "Color"}
        lines.append(f"\nMost affected: {cnames[max_c]}")
        lines.append(f"  (delta_{max_c}={deltas[max_c]:.4f})")
        if "text_en" in abc_res and abc_res["text_en"]:
            wrapped = textwrap.fill(abc_res["text_en"][:250], width=42)
            lines.append(f"\n{wrapped}")
        ax_txt.text(0.05, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
                   fontsize=7, family="monospace", verticalalignment="top",
                   bbox=dict(facecolor="lightyellow", alpha=0.8, pad=8,
                            edgecolor="gray", linewidth=0.5))
        ax_txt.set_title("CF Explanation", fontsize=9, fontweight="bold")

        # ── Save ─────────────────────────────
        fig.suptitle(f"{src_name} -> {tgt_name} | Example {i+1}/{n}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"{src_name}_to_{tgt_name}_{i+1:03d}.png"
        fig.savefig(output_dir / fname, dpi=200, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    gcam.remove()
    print(f"  [Individual] {n} panels saved to {output_dir.name}/")