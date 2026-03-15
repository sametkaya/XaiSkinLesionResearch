"""
explainers/shap_explainer.py
-----------------------------
SHAP (SHapley Additive exPlanations) for dermoscopic image classification.

Uses GradientExplainer (DeepSHAP variant) which combines the speed of
gradient-based methods with the theoretical guarantees of SHAP values.
GradientExplainer uses a background distribution of training images to
estimate feature attributions.

For image classification, SHAP values indicate the contribution of each
pixel (grouped into superpixels for interpretability) to the predicted
class probability — positive values support the prediction, negative
values oppose it.

Two visualisation modes:
  1. Raw SHAP heatmap: per-pixel signed attribution values
  2. Superpixel SHAP: QuickSHIFT segments coloured by mean SHAP value

References:
    Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
    model predictions. NeurIPS 2017.
    https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

    Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for
    deep networks. ICML 2017.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from torch.utils.data import DataLoader
from skimage.segmentation import quickshift

from src import config
from src.model import SkinLesionClassifier
from src.explainers.gradcam import denormalize
from src.utils.result_manager import ResultManager


# ─────────────────────────────────────────────
# GradientSHAP (DeepSHAP variant)
# ─────────────────────────────────────────────

class GradientSHAPExplainer:
    """
    GradientSHAP explainer using integrated gradients over a
    background distribution.

    Computes E_z[∇_x f(αx + (1-α)z) · (x - z)] where z is drawn
    from the background set and α ~ Uniform(0,1).

    Parameters
    ----------
    model : SkinLesionClassifier
    device : torch.device
    n_background : int
        Number of background images for the baseline distribution.
    n_samples : int
        Number of interpolation steps per explanation.
    """

    def __init__(
        self,
        model: SkinLesionClassifier,
        device: torch.device,
        n_background: int = 50,
        n_samples: int = 25,
    ):
        self.model        = model
        self.device       = device
        self.n_background = n_background
        self.n_samples    = n_samples
        self.background   = None   # set via set_background()

        self.model.eval()

    def set_background(self, loader: DataLoader) -> None:
        """
        Build background distribution from a random subset of the
        training/validation set.

        Parameters
        ----------
        loader : DataLoader
        """
        bg_tensors = []
        with torch.no_grad():
            for images, _ in loader:
                bg_tensors.append(images)
                if sum(t.shape[0] for t in bg_tensors) >= self.n_background:
                    break
        bg = torch.cat(bg_tensors, dim=0)[:self.n_background]
        self.background = bg.to(self.device)
        print(f"[SHAP] Background set: {self.background.shape[0]} images")

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute GradientSHAP attribution map for a single image.

        Parameters
        ----------
        image_tensor : torch.Tensor  (3, H, W) normalised
        target_class : int or None

        Returns
        -------
        shap_values : np.ndarray  (H, W) — signed attribution per pixel
        target_class : int
        confidence   : float
        """
        if self.background is None:
            raise RuntimeError(
                "[SHAP] Background not set. Call set_background() first."
            )

        inp = image_tensor.unsqueeze(0).to(self.device)

        # Get predicted class
        with torch.no_grad():
            logits = self.model(inp)
            probs  = torch.softmax(logits, dim=1)
            if target_class is None:
                target_class = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, target_class].item())

        # GradientSHAP: average over n_samples interpolations
        total_grads = torch.zeros_like(inp)

        for _ in range(self.n_samples):
            # Random background baseline
            idx  = np.random.randint(0, self.background.shape[0])
            base = self.background[idx:idx+1]                   # (1,3,H,W)

            # Random interpolation factor α ~ Uniform(0,1)
            alpha = float(np.random.uniform(0, 1))

            # Interpolated input
            x_interp = (alpha * inp + (1 - alpha) * base).detach().requires_grad_(True)

            with torch.enable_grad():
                out  = self.model(x_interp)
                score = out[0, target_class]
                score.backward()

            # Gradient × (input - baseline)
            grad    = x_interp.grad.detach()
            delta   = (inp - base).detach()
            total_grads += grad * delta

        # Average over samples
        shap_map = (total_grads / self.n_samples).squeeze(0)  # (3, H, W)

        # Collapse to single channel: sum of absolute values across RGB
        shap_abs  = shap_map.abs().sum(dim=0).cpu().numpy()   # (H, W)
        shap_sign = shap_map.sum(dim=0).cpu().numpy()          # (H, W) signed

        # Normalise to [-1, 1] for visualisation
        max_abs = np.abs(shap_sign).max() + 1e-8
        shap_norm = shap_sign / max_abs

        return shap_norm, target_class, confidence


# ─────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────

def overlay_shap(
    image: np.ndarray,
    shap_values: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay signed SHAP values on the original image.

    Positive values (support prediction) → red
    Negative values (oppose prediction) → blue

    Parameters
    ----------
    image     : (H, W, 3) uint8
    shap_values: (H, W) float in [-1, 1]
    alpha     : float — overlay opacity

    Returns
    -------
    np.ndarray  (H, W, 3) uint8
    """
    H, W = image.shape[:2]

    # Red channel for positive, blue for negative
    pos_map = np.clip(shap_values, 0, 1)     # (H, W)
    neg_map = np.clip(-shap_values, 0, 1)    # (H, W)

    overlay = image.copy().astype(float)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pos_map * 255 * alpha, 0, 255)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + neg_map * 255 * alpha, 0, 255)

    return overlay.astype(np.uint8)


def superpixel_shap(
    image: np.ndarray,
    shap_values: np.ndarray,
    kernel_size: int = 4,
    max_dist: int = 200,
) -> np.ndarray:
    """
    Segment image with QuickSHIFT and colour each segment by mean SHAP.

    Parameters
    ----------
    image      : (H, W, 3) uint8
    shap_values: (H, W) float in [-1, 1]

    Returns
    -------
    np.ndarray  (H, W, 3) uint8
    """
    segments = quickshift(
        image.astype(float) / 255.0,
        kernel_size=kernel_size,
        max_dist=max_dist,
        ratio=0.2,
    )

    result = image.copy().astype(float)
    cmap   = plt.cm.RdBu_r

    for seg_id in np.unique(segments):
        mask      = segments == seg_id
        mean_shap = float(shap_values[mask].mean())
        colour_rgb= np.array(cmap((mean_shap + 1) / 2)[:3]) * 255
        result[mask] = 0.4 * result[mask] + 0.6 * colour_rgb

    return result.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────

class SHAPExperiment:
    """
    Run SHAP explanations on a sample of test images and save outputs.

    Produces four-panel figures:
      [Original] | [SHAP Heatmap] | [SHAP Overlay] | [Superpixel SHAP]

    Parameters
    ----------
    model       : SkinLesionClassifier
    test_loader : DataLoader
    val_loader  : DataLoader  (used for background distribution)
    device      : torch.device
    result_dir  : Path
    """

    def __init__(
        self,
        model: SkinLesionClassifier,
        test_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        result_dir: Path,
    ):
        self.model       = model
        self.test_loader = test_loader
        self.device      = device
        self.result_dir  = result_dir
        self.class_names = config.CLASS_LABELS

        self.explainer = GradientSHAPExplainer(
            model=model,
            device=device,
            n_background=50,
            n_samples=25,
        )
        self.explainer.set_background(val_loader)

    def _collect_samples(self, n_per_class: int = 3) -> list:
        """Collect correctly classified samples, n per class."""
        self.model.eval()
        buckets = {i: [] for i in range(config.NUM_CLASSES)}

        with torch.no_grad():
            for images, labels in self.test_loader:
                imgs = images.to(self.device)
                preds = self.model(imgs).argmax(dim=1).cpu()
                for img, lbl, prd in zip(images.cpu(), labels, preds):
                    if int(lbl) == int(prd) and len(buckets[int(lbl)]) < n_per_class:
                        buckets[int(lbl)].append((img, int(lbl)))
                if all(len(v) >= n_per_class for v in buckets.values()):
                    break

        samples = []
        for items in buckets.values():
            samples.extend(items)
        return samples

    def run(self) -> dict:
        """
        Execute SHAP experiment and save visualisations.

        Returns
        -------
        dict  experiment statistics
        """
        start   = time.time()
        per_dir = self.result_dir / "per_class"
        per_dir.mkdir(exist_ok=True)

        samples = self._collect_samples(n_per_class=3)
        stats   = {
            "total_samples"   : len(samples),
            "samples_per_class": 3,
            "n_background"    : self.explainer.n_background,
            "n_interpolations": self.explainer.n_samples,
        }

        all_confs = []
        for idx, (img_tensor, true_label) in enumerate(samples):
            img_np = denormalize(img_tensor)              # (H, W, 3)

            shap_vals, pred_cls, conf = self.explainer.explain(img_tensor)
            all_confs.append(conf)

            overlay   = overlay_shap(img_np, shap_vals)
            superpix  = superpixel_shap(img_np, shap_vals)

            # Absolute SHAP magnitude heatmap (for colourbar display)
            shap_abs  = np.abs(shap_vals)
            shap_abs_norm = (shap_abs - shap_abs.min()) / (shap_abs.max() - shap_abs.min() + 1e-8)

            class_name = self.class_names[true_label]
            fig, axes  = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(img_np)
            axes[0].set_title(
                f"Original\nTrue: {class_name}", fontsize=9
            )
            axes[0].axis("off")

            im = axes[1].imshow(shap_vals, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[1].set_title(
                f"SHAP Values\nPred: {self.class_names[pred_cls]} ({conf:.2f})",
                fontsize=9,
            )
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            axes[2].imshow(overlay)
            axes[2].set_title("SHAP Overlay\nRed=support, Blue=oppose", fontsize=9)
            axes[2].axis("off")

            axes[3].imshow(superpix)
            axes[3].set_title("Superpixel SHAP\n(QuickSHIFT segments)", fontsize=9)
            axes[3].axis("off")

            plt.suptitle(
                f"GradientSHAP — {config.CLASS_NAMES[class_name]}",
                fontsize=10, fontweight="bold",
            )
            plt.tight_layout()
            save_path = per_dir / f"sample_{idx:03d}_{class_name}.png"
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close()

        elapsed = time.time() - start
        stats["mean_confidence"] = round(float(np.mean(all_confs)), 4)
        stats["elapsed_seconds"] = round(elapsed, 2)

        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="GradientSHAP Experiment",
            conditions={
                "model"          : config.MODEL_NAME,
                "method"         : "GradientSHAP (DeepSHAP variant)",
                "background_n"   : self.explainer.n_background,
                "interpolations" : self.explainer.n_samples,
                "samples_per_class": 3,
                "segmentation"   : "QuickSHIFT",
            },
            statistics=stats,
        )

        print(
            f"[SHAP] {len(samples)} panels saved to {per_dir}  "
            f"(avg {elapsed/len(samples):.1f}s/image)"
        )
        return stats
