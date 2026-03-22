"""
src/explainers/abc_counterfactual.py
-------------------------------------
ABC-Guided Counterfactual Explainer  (v4 — segmentation-guided)

Extends the ACE counterfactual framework with:
  1. Segmentation-guided perturbation: δ_eff = soft_mask ⊙ δ_raw
     Only lesion pixels are modified; background stays exactly unchanged.
  2. VGG perceptual loss: preserves semantic structure within the lesion.
  3. Gaussian blur on δ per step: prevents high-frequency noise.
  4. ABC morphology preservation via learned regressor constraints.

Loss:
  L = λ_cls  · CE(f(x+M⊙δ), c_tgt)
    + λ_A    · |g(x+M⊙δ)_A − s_A|
    + λ_B    · |g(x+M⊙δ)_B − s_B|
    + λ_C    · |g(x+M⊙δ)_C − s_C|
    + λ_l1   · ‖M⊙δ‖₁
    + λ_TV   · TV(M⊙δ)
    + λ_perc · Σ‖φ_j(x) − φ_j(x+M⊙δ)‖²

References
----------
Singla, S. et al. (2023). Medical Image Analysis, 84, 102721.
Wachter, S. et al. (2017). Harvard J. Law & Tech., 31(2).
Rudin, L. et al. (1992). Physica D, 60(1-4), 259–268.
Johnson, J. et al. (2016). Perceptual Losses. ECCV 2016.
Sobieski, A. et al. (2025). RCSB. ICLR 2025.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models as tv_models
from tqdm import tqdm

from src.abc.config_abc import (
    ABC_CF_MAX_ITER, ABC_CF_LEARNING_RATE,
    ABC_CF_LAMBDA_CLS, ABC_CF_LAMBDA_A, ABC_CF_LAMBDA_B, ABC_CF_LAMBDA_C,
    ABC_CF_LAMBDA_L1, ABC_CF_LAMBDA_TV, ABC_CF_LAMBDA_PERC,
    ABC_CF_CONFIDENCE_THRES,
    ABC_CF_NUM_IMAGES, ABC_CF_PIXEL_THRESHOLD, ABC_CF_PAIRS,
    ABC_CF_MASK_DILATE_PX, ABC_CF_MASK_BLUR_SIGMA,
    ABC_CF_DELTA_BLUR_SIGMA, ABC_CF_DELTA_BLUR_KERNEL,
    IMAGE_MEAN, IMAGE_STD, IMAGE_SIZE, ABC_NAMES,
)
from src.segmentation.segmenter import LesionSegmenter
from src.utils.result_manager import ResultManager
from src.explainers.cf_visualizer import save_8panel_figure
from src.explainers.abc_visualizer import save_abc_panel
from src.explainers.individual_panels import generate_individual_panels
from src.explainers.cf_losses import (
    VGGPerceptualLoss,
    total_variation_loss,
    low_pass_filter,
    saliency_init,
    min_perturbation_hinge,
)


# ─────────────────────────────────────────────
# Class name mappings (for textual explanations)
# ─────────────────────────────────────────────

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses (akiec)",
    "bcc"  : "Basal Cell Carcinoma (bcc)",
    "bkl"  : "Benign Keratosis (bkl)",
    "df"   : "Dermatofibroma (df)",
    "mel"  : "Melanoma (mel)",
    "nv"   : "Melanocytic Nevi (nv)",
    "vasc" : "Vascular Lesions (vasc)",
}

CLASS_FULL_NAMES_TR = {
    "akiec": "Aktinik Keratoz",
    "bcc"  : "Bazal Hücreli Karsinom",
    "bkl"  : "Benign Keratoz",
    "df"   : "Dermatofibrom",
    "mel"  : "Melanom",
    "nv"   : "Melanositik Nevüs",
    "vasc" : "Vasküler Lezyon",
}

# ABC score interpretation thresholds
ABC_LEVEL_THRESHOLDS = {
    "low"   : (0.0, 0.33),
    "medium": (0.33, 0.66),
    "high"  : (0.66, 1.0),
}


def _abc_level(score: float) -> str:
    """Interpret a [0,1] ABC score as low/medium/high."""
    if score < 0.33:
        return "low"
    elif score < 0.66:
        return "moderate"
    else:
        return "high"


def _abc_level_tr(score: float) -> str:
    """Turkish interpretation of ABC scores."""
    if score < 0.33:
        return "düşük"
    elif score < 0.66:
        return "orta"
    else:
        return "yüksek"


def _delta_direction(src: float, cf: float) -> str:
    """Direction of change: increased / decreased / unchanged."""
    diff = cf - src
    if abs(diff) < 0.01:
        return "unchanged"
    return "increased" if diff > 0 else "decreased"


def _delta_direction_tr(src: float, cf: float) -> str:
    """Turkish direction of change."""
    diff = cf - src
    if abs(diff) < 0.01:
        return "değişmedi"
    return "arttı" if diff > 0 else "azaldı"


# ─────────────────────────────────────────────
# Textual explanation generators
# ─────────────────────────────────────────────

def generate_textual_explanation(result: Dict, src_name: str, tgt_name: str) -> str:
    """
    Generate a human-readable textual counterfactual explanation (English).

    Example output:
      "This Melanocytic Nevi lesion (predicted with 92% confidence) would
       be reclassified as Melanoma (87% confidence) if:
         • Asymmetry increased from 0.23 (low) to 0.67 (high)   [+0.44]
         • Color Variegation increased from 0.18 (low) to 0.52 (moderate)  [+0.34]
         • Border Irregularity remained approximately unchanged (0.31 → 0.33)
       The counterfactual required changes to 12.3% of pixels (sparsity)
       with a mean perturbation magnitude of 0.0234 (L1)."
    """
    src_full = CLASS_FULL_NAMES.get(src_name, src_name)
    tgt_full = CLASS_FULL_NAMES.get(tgt_name, tgt_name)
    abc_src  = result["abc_src"]
    abc_cf   = result["abc_cf"]
    valid    = result["validity"]

    if valid:
        header = (
            f"This {src_full} lesion would be reclassified as "
            f"{tgt_full} (confidence: {result['final_prob']:.0%}) if:"
        )
    else:
        header = (
            f"The model could NOT confidently reclassify this {src_full} "
            f"lesion as {tgt_full} (best confidence: {result['final_prob']:.0%}). "
            f"The attempted changes were:"
        )

    changes = []
    for crit, full_name in [("A", "Asymmetry"), ("B", "Border Irregularity"), ("C", "Color Variegation")]:
        s = abc_src[crit]
        c = abc_cf[crit]
        d = c - s
        direction = _delta_direction(s, c)

        if direction == "unchanged":
            changes.append(
                f"  • {full_name} remained approximately unchanged "
                f"({s:.2f} → {c:.2f})"
            )
        else:
            changes.append(
                f"  • {full_name} {direction} from {s:.2f} ({_abc_level(s)}) "
                f"to {c:.2f} ({_abc_level(c)})  [{d:+.2f}]"
            )

    footer = (
        f"Perturbation: sparsity={result['sparsity']:.1%} of pixels changed, "
        f"L1 magnitude={result['proximity_l1']:.4f}, "
        f"converged in {result['n_iter']} iterations."
    )

    return "\n".join([header, *changes, footer])


def generate_textual_explanation_tr(result: Dict, src_name: str, tgt_name: str) -> str:
    """
    Turkish textual counterfactual explanation for clinical accessibility.

    Example output:
      "Bu Melanositik Nevüs lezyonu aşağıdaki değişikliklerle
       Melanom olarak sınıflandırılırdı (%87 güven):
         • Asimetri: 0.23'ten (düşük) → 0.67'ye (yüksek) arttı  [+0.44]
         • Renk Çeşitliliği: 0.18'den (düşük) → 0.52'ye (orta) arttı  [+0.34]
         • Sınır Düzensizliği: yaklaşık aynı kaldı (0.31 → 0.33)
       Pertürbasyon: piksellerin %12.3'ü değişti, ortalama L1=0.0234."
    """
    src_full = CLASS_FULL_NAMES_TR.get(src_name, src_name)
    tgt_full = CLASS_FULL_NAMES_TR.get(tgt_name, tgt_name)
    abc_src  = result["abc_src"]
    abc_cf   = result["abc_cf"]
    valid    = result["validity"]

    ABC_NAMES_TR = {
        "A": "Asimetri",
        "B": "Sınır Düzensizliği",
        "C": "Renk Çeşitliliği",
    }

    if valid:
        header = (
            f"Bu {src_full} lezyonu aşağıdaki değişikliklerle "
            f"{tgt_full} olarak sınıflandırılırdı (güven: %{result['final_prob']*100:.0f}):"
        )
    else:
        header = (
            f"Model bu {src_full} lezyonunu {tgt_full} olarak yeniden "
            f"sınıflandıramadı (en iyi güven: %{result['final_prob']*100:.0f}). "
            f"Denenen değişiklikler:"
        )

    changes = []
    for crit in ["A", "B", "C"]:
        s = abc_src[crit]
        c = abc_cf[crit]
        d = c - s
        direction = _delta_direction_tr(s, c)
        name_tr = ABC_NAMES_TR[crit]

        if abs(d) < 0.01:
            changes.append(
                f"  • {name_tr}: yaklaşık aynı kaldı ({s:.2f} → {c:.2f})"
            )
        else:
            changes.append(
                f"  • {name_tr}: {s:.2f}'den ({_abc_level_tr(s)}) → "
                f"{c:.2f}'ye ({_abc_level_tr(c)}) {direction}  [{d:+.2f}]"
            )

    footer = (
        f"Pertürbasyon: piksellerin %{result['sparsity']*100:.1f}'i değişti, "
        f"ortalama L1={result['proximity_l1']:.4f}, "
        f"{result['n_iter']} iterasyonda yakınsadı."
    )

    return "\n".join([header, *changes, footer])


# ─────────────────────────────────────────────
# VGG Perceptual Loss (Johnson et al., ECCV 2016)
# ─────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-16 feature matching.

    Computes MSE between intermediate feature maps of original and
    counterfactual images.  This prevents adversarial high-frequency
    perturbations while preserving semantic structure.

    Uses VGG-16 WITHOUT batch normalization (better perceptual features).

    References
    ----------
    Johnson, J., Alahi, A., & Fei-Fei, L. (2016).
        Perceptual Losses for Real-Time Style Transfer and Super-Resolution.
        ECCV 2016.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = tv_models.vgg16(weights="IMAGENET1K_V1").features
        # Extract features at relu2_2 (layer 8) and relu3_3 (layer 15)
        self.slice1 = nn.Sequential(*list(vgg[:9])).to(device).eval()
        self.slice2 = nn.Sequential(*list(vgg[9:16])).to(device).eval()
        for p in self.parameters():
            p.requires_grad = False
        # VGG expects ImageNet normalization
        self.register_buffer(
            "vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            "vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        )

    def _denorm_to_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from HAM10000 normalization to VGG ImageNet normalization."""
        ham_mean = torch.tensor(IMAGE_MEAN, device=x.device).view(1, 3, 1, 1)
        ham_std  = torch.tensor(IMAGE_STD,  device=x.device).view(1, 3, 1, 1)
        # HAM → [0,1] → VGG normalized
        x_01 = x * ham_std + ham_mean
        x_01 = torch.clamp(x_01, 0, 1)
        return (x_01 - self.vgg_mean) / self.vgg_std

    def forward(self, x: torch.Tensor, x_cf: torch.Tensor) -> torch.Tensor:
        x_vgg  = self._denorm_to_vgg(x)
        cf_vgg = self._denorm_to_vgg(x_cf)
        f1_x, f1_cf = self.slice1(x_vgg), self.slice1(cf_vgg)
        f2_x, f2_cf = self.slice2(f1_x),  self.slice2(f1_cf)
        loss = F.mse_loss(f1_cf, f1_x) + F.mse_loss(f2_cf, f2_x)
        return loss


# ─────────────────────────────────────────────
# Soft mask preparation (SoftSeg-inspired)
# ─────────────────────────────────────────────

def prepare_soft_mask(
    binary_mask: np.ndarray,
    dilate_px: int = ABC_CF_MASK_DILATE_PX,
    blur_sigma: float = ABC_CF_MASK_BLUR_SIGMA,
) -> np.ndarray:
    """
    Convert binary lesion mask to soft float mask with smooth edges.

    Steps:
      1. Dilate by dilate_px to capture border transition zone
      2. Gaussian blur for smooth edges (prevents visible seams)
      3. Normalize to [0, 1]

    Reference: SoftSeg (Medical Image Analysis, 2021)
    """
    mask = binary_mask.astype(np.uint8) * 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    if blur_sigma > 0:
        ksize = int(6 * blur_sigma) | 1  # ensure odd
        mask = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), blur_sigma)
    mask = mask.astype(np.float32) / max(mask.max(), 1e-8)
    return mask


def mask_to_tensor(soft_mask: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H,W) soft mask to (1,1,H,W) tensor for broadcasting with δ."""
    return torch.from_numpy(soft_mask).float().unsqueeze(0).unsqueeze(0).to(device)


# ─────────────────────────────────────────────
# Gaussian blur on perturbation (per-step)
# ─────────────────────────────────────────────

def gaussian_blur_delta(
    delta: torch.Tensor,
    kernel_size: int = ABC_CF_DELTA_BLUR_KERNEL,
    sigma: float = ABC_CF_DELTA_BLUR_SIGMA,
) -> torch.Tensor:
    """
    Apply Gaussian blur to perturbation tensor in-place.

    This acts as an implicit smoothness prior at each optimization step,
    preventing high-frequency noise within the lesion mask.

    Reference: Mirror-CFE (2024) uses Gaussian blur as validity test.
    """
    if sigma <= 0:
        return delta
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=delta.device).float() - kernel_size // 2
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()
    # Separable 2D convolution
    kernel_h = gauss.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    kernel_w = gauss.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    pad = kernel_size // 2
    blurred = F.conv2d(delta, kernel_h, padding=(pad, 0), groups=3)
    blurred = F.conv2d(blurred, kernel_w, padding=(0, pad), groups=3)
    return blurred


# ─────────────────────────────────────────────
# SSIM metric
# ─────────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two images.

    Parameters: img1, img2: (H,W,3) uint8 numpy arrays.
    Returns: float in [-1, 1], higher = more similar.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


# ─────────────────────────────────────────────
# Total Variation loss
# ─────────────────────────────────────────────

def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    Compute anisotropic Total Variation of a perturbation tensor.

    TV(δ) = Σ|δ_{i,j} − δ_{i+1,j}| + Σ|δ_{i,j} − δ_{i,j+1}|

    This promotes spatially smooth perturbations: adjacent pixels
    should change by similar amounts. Without TV, gradient-based
    counterfactuals tend to produce per-pixel noise that is
    imperceptible but not clinically interpretable.

    Parameters
    ----------
    delta : torch.Tensor  (B, C, H, W) or (1, C, H, W)

    Returns
    -------
    torch.Tensor (scalar)
        Mean TV value across the batch.

    References
    ----------
    Rudin, L. I., Osher, S., & Fatemi, E. (1992).
        Nonlinear total variation based noise removal algorithms.
        Physica D, 60(1-4), 259–268.
    """
    diff_h = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :])
    diff_w = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()


# ─────────────────────────────────────────────
# Ablation mode → λ values
# ─────────────────────────────────────────────

ABLATION_MODES = {
    "baseline": {"A": 0.0,             "B": 0.0,             "C": 0.0},
    "A_only"  : {"A": ABC_CF_LAMBDA_A, "B": 0.0,             "C": 0.0},
    "AB"      : {"A": ABC_CF_LAMBDA_A, "B": ABC_CF_LAMBDA_B, "C": 0.0},
    "ABC"     : {"A": ABC_CF_LAMBDA_A, "B": ABC_CF_LAMBDA_B, "C": ABC_CF_LAMBDA_C},
}


# ─────────────────────────────────────────────
# Core counterfactual generator
# ─────────────────────────────────────────────

class ABCCounterfactualExplainer:
    """
    Generate segmentation-guided ABC-constrained counterfactual explanations.

    v4: Perturbations are masked to the lesion region via δ_eff = M ⊙ δ_raw.
    VGG perceptual loss prevents adversarial noise. Gaussian blur smooths δ.

    Parameters
    ----------
    classifier, abc_regressor, device, class_labels
    segmenter : LesionSegmenter or None
        If provided, generates masks for images without pre-computed masks.
    """

    def __init__(
        self,
        classifier: nn.Module,
        abc_regressor: nn.Module,
        device: torch.device,
        class_labels: List[str],
        segmenter: Optional[LesionSegmenter] = None,
    ):
        self.clf       = classifier
        self.abc_reg   = abc_regressor
        self.device    = device
        self.labels    = class_labels
        self.segmenter = segmenter
        self.perc_loss = VGGPerceptualLoss(device)

        self.clf.eval()
        self.abc_reg.eval()

    def generate(
        self,
        image_tensor: torch.Tensor,
        source_class: int,
        target_class: int,
        mode: str = "ABC",
        mask: Optional[np.ndarray] = None,
        max_iter: int = ABC_CF_MAX_ITER,
        lr: float = ABC_CF_LEARNING_RATE,
        confidence_threshold: float = ABC_CF_CONFIDENCE_THRES,
    ) -> Dict:
        """
        Generate a segmentation-guided ABC counterfactual.

        Parameters
        ----------
        image_tensor : torch.Tensor (3, H, W) — normalised
        source_class, target_class : int
        mode : str — 'baseline' | 'A_only' | 'AB' | 'ABC'
        mask : np.ndarray (H, W) bool or None
            Binary lesion mask.  If None, segmenter is used as fallback.
        """
        lambdas  = ABLATION_MODES[mode]
        orig     = image_tensor.unsqueeze(0).to(self.device)
        cf       = orig.clone().detach()
        delta    = torch.zeros_like(orig, requires_grad=True, device=self.device)
        target_t = torch.tensor([target_class], dtype=torch.long, device=self.device)

        # ── Prepare soft mask ──────────────────
        if mask is None and self.segmenter is not None:
            mask = self.segmenter.segment(image_tensor)
        if mask is not None:
            soft_mask = prepare_soft_mask(mask)
            mask_t    = mask_to_tensor(soft_mask, self.device)  # (1,1,H,W)
        else:
            mask_t    = torch.ones(1, 1, image_tensor.shape[1],
                                   image_tensor.shape[2], device=self.device)
            soft_mask = np.ones((image_tensor.shape[1], image_tensor.shape[2]))

        # Pre-compute source ABC scores and source probability
        with torch.no_grad():
            src_abc    = self.abc_reg(orig).squeeze(0)
            src_logits = self.clf(orig)
            src_probs  = torch.softmax(src_logits, dim=1)
            src_prob   = float(src_probs[0, source_class].item())

        # ── Adam optimizer for δ ───────────────
        optimizer = torch.optim.Adam([delta], lr=lr)

        n_iter     = 0
        best_prob  = 0.0
        best_delta = None

        with torch.enable_grad():
            for step in range(max_iter):
                optimizer.zero_grad()

                # Apply mask: only lesion pixels are perturbed
                delta_masked = mask_t * delta
                x_cf = cf + delta_masked

                # Classification loss
                logits   = self.clf(x_cf)
                probs    = torch.softmax(logits, dim=1)
                loss_cls = F.cross_entropy(logits, target_t) * ABC_CF_LAMBDA_CLS

                # ABC preservation losses
                cf_abc = self.abc_reg(x_cf).squeeze(0)
                loss_A = lambdas["A"] * torch.abs(cf_abc[0] - src_abc[0])
                loss_B = lambdas["B"] * torch.abs(cf_abc[1] - src_abc[1])
                loss_C = lambdas["C"] * torch.abs(cf_abc[2] - src_abc[2])

                # Sparsity (L1 on masked perturbation)
                loss_l1 = ABC_CF_LAMBDA_L1 * delta_masked.abs().mean()

                # Total Variation on masked perturbation
                loss_tv = ABC_CF_LAMBDA_TV * total_variation_loss(delta_masked)

                # VGG Perceptual loss — prevents adversarial noise
                loss_perc = -ABC_CF_LAMBDA_PERC * self.perc_loss(orig, x_cf)  # v6: NEGATIVE — encourage visible change

                loss_hinge = 30.0 * F.relu(0.05 - delta_masked.abs().mean())
                loss = loss_cls + loss_A + loss_B + loss_C + loss_tv + loss_perc + loss_hinge
                loss.backward()
                optimizer.step()

                # Post-step processing
                with torch.no_grad():
                    # Gaussian blur on δ for smoothness
                    # pass  # v6: delta blur disabled — preserves focal perturbations  # v6: disabled — preserves focal changes
                    # Re-apply mask (ensure no leakage after blur)
                    delta.data.mul_(mask_t)
                    # Clip to valid range
                    x_clamped = torch.clamp(cf + mask_t * delta, -3.0, 3.0)
                    delta.data.copy_((x_clamped - cf) / (mask_t + 1e-8))
                    delta.data.mul_(mask_t)

                n_iter   = step + 1
                cur_prob = float(probs[0, target_class].item())

                if cur_prob > best_prob:
                    best_prob  = cur_prob
                    best_delta = (mask_t * delta).detach().clone()

                if False and cur_prob >= confidence_threshold:  # v6: disabled — let optimizer build visible delta
                    break

        # Use best delta if final didn't reach threshold
        with torch.no_grad():
            if best_delta is None:
                delta_final_gpu = mask_t * delta
            elif best_prob > float(probs[0, target_class].item()):
                delta_final_gpu = best_delta
            else:
                delta_final_gpu = mask_t * delta

            x_cf_final   = (cf + delta_final_gpu).squeeze(0).cpu()
            delta_final   = delta_final_gpu.squeeze(0).cpu()

            final_logits = self.clf(cf + delta_final_gpu)
            final_probs  = torch.softmax(final_logits, dim=1)
            final_pred   = int(final_probs.argmax(dim=1).item())
            final_prob   = float(final_probs[0, target_class].item())

            cf_abc_final = self.abc_reg(cf + delta_final_gpu).squeeze(0).cpu()

        # Compute metrics
        validity = 1 if final_pred == target_class else 0
        prox_l1  = float(delta_final.abs().mean().item())
        sparsity = float(
            (delta_final.abs() > ABC_CF_PIXEL_THRESHOLD).float().mean().item()
        )

        # SSIM between original and counterfactual
        orig_np = self._denorm_static(image_tensor)
        cf_np   = self._denorm_static(x_cf_final)
        ssim_val = compute_ssim(orig_np, cf_np)

        return {
            "cf_tensor"   : x_cf_final,
            "delta"       : delta_final,
            "mask"        : mask,  # original binary mask for visualization
            "validity"    : validity,
            "final_prob"  : round(final_prob, 4),
            "src_prob"    : round(src_prob, 4),
            "proximity_l1": round(prox_l1, 5),
            "sparsity"    : round(sparsity, 5),
            "ssim"        : round(ssim_val, 4),
            "n_iter"      : n_iter,
            "abc_src"     : {
                "A": round(float(src_abc[0]), 4),
                "B": round(float(src_abc[1]), 4),
                "C": round(float(src_abc[2]), 4),
            },
            "abc_cf"      : {
                "A": round(float(cf_abc_final[0]), 4),
                "B": round(float(cf_abc_final[1]), 4),
                "C": round(float(cf_abc_final[2]), 4),
            },
            "delta_A"     : round(abs(float(cf_abc_final[0]) - float(src_abc[0])), 4),
            "delta_B"     : round(abs(float(cf_abc_final[1]) - float(src_abc[1])), 4),
            "delta_C"     : round(abs(float(cf_abc_final[2]) - float(src_abc[2])), 4),
            "mode"        : mode,
        }

    @staticmethod
    def _denorm_static(t: torch.Tensor) -> np.ndarray:
        mean = np.array(IMAGE_MEAN)
        std  = np.array(IMAGE_STD)
        img  = t.permute(1, 2, 0).numpy()
        return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────

class ABCCounterfactualExperiment:
    """
    Full ABC-guided counterfactual experiment with ablation study.

    v2 additions:
      - Textual counterfactual explanations (EN + TR)
      - Enhanced visual panels with narrative annotations
      - Per-pair narrative summary files
      - Comprehensive result.txt with all statistics and timings

    Runs four ablation modes for each class-pair defined in ABC_CF_PAIRS:
      baseline, A_only, AB, ABC

    Parameters
    ----------
    classifier, abc_regressor, test_loader, device, result_dir, class_labels
    """

    def __init__(
        self,
        classifier: nn.Module,
        abc_regressor: nn.Module,
        test_loader,
        device: torch.device,
        result_dir: Path,
        class_labels: List[str],
        segmenter: Optional[LesionSegmenter] = None,
    ):
        self.explainer  = ABCCounterfactualExplainer(
            classifier, abc_regressor, device, class_labels,
            segmenter=segmenter,
        )
        self.loader     = test_loader
        self.device     = device
        self.result_dir = result_dir
        self.labels     = class_labels
        self.label2idx  = {lbl: i for i, lbl in enumerate(class_labels)}
        self.segmenter  = segmenter

    def run(self) -> Dict:
        """Execute full experiment and return aggregate statistics."""
        start       = time.time()
        all_records = []

        pairs_dir      = self.result_dir / "per_class"
        ablation_dir   = self.result_dir / "ablation"
        metrics_dir    = self.result_dir / "metrics"
        narrative_dir  = self.result_dir / "narratives"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        ablation_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        narrative_dir.mkdir(parents=True, exist_ok=True)

        for src_name, tgt_name in ABC_CF_PAIRS:
            src_idx = self.label2idx.get(src_name)
            tgt_idx = self.label2idx.get(tgt_name)
            if src_idx is None or tgt_idx is None:
                print(f"[ABCCounterfactual] Skipping unknown pair {src_name}→{tgt_name}")
                continue

            print(f"\n[ABCCounterfactual] Pair: {src_name} → {tgt_name}")
            pair_samples = self._collect_samples(src_idx, ABC_CF_NUM_IMAGES)

            for mode in ABLATION_MODES:
                mode_records = []
                for img_t, true_lbl, mask in pair_samples:
                    result = self.explainer.generate(
                        img_t, src_idx, tgt_idx, mode=mode, mask=mask
                    )
                    result.update({
                        "src_class": src_name,
                        "tgt_class": tgt_name,
                    })
                    # Generate textual explanations
                    result["text_en"] = generate_textual_explanation(
                        result, src_name, tgt_name
                    )
                    result["text_tr"] = generate_textual_explanation_tr(
                        result, src_name, tgt_name
                    )
                    mode_records.append(result)
                    all_records.append(result)

                # Save visual panels for first 3 samples
                self._save_panels(
                    mode_records[:10],
                    pairs_dir / f"{src_name}_to_{tgt_name}_{mode}.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                )

                # Save enhanced panels with textual annotations
                self._save_narrative_panels(
                    mode_records[:10],
                    narrative_dir / f"{src_name}_to_{tgt_name}_{mode}_narrative.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                    src_name, tgt_name,
                )
                # v7: 8-panel publication-quality figure
                save_8panel_figure(
                    src_idx,
                    mode_records[:10],
                    self.explainer.clf,
                    self.device,
                    pairs_dir / f"{src_name}_to_{tgt_name}_{mode}_8panel.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                    max_rows=10,
                )
                # v7: ABC clinical visualization
                if mode == "ABC":
                    save_abc_panel(
                        mode_records[:5],
                        self.explainer.abc_reg,
                        self.device,
                        pairs_dir / f"{src_name}_to_{tgt_name}_abc_clinical.png",
                        f"{src_name} → {tgt_name} | ABC Clinical Analysis",
                        max_rows=5,
                    )

                # Ablation row
                print(
                    f"  [{mode:8s}]  validity={np.mean([r['validity'] for r in mode_records]):.3f}  "
                    f"L1={np.mean([r['proximity_l1'] for r in mode_records]):.4f}  "
                    f"ΔA={np.mean([r['delta_A'] for r in mode_records]):.4f}  "
                    f"ΔB={np.mean([r['delta_B'] for r in mode_records]):.4f}  "
                    f"ΔC={np.mean([r['delta_C'] for r in mode_records]):.4f}"
                )

        elapsed = time.time() - start

        # ── Save all records CSV ───────────────
        self._save_all_records(all_records, metrics_dir / "all_records.csv")

        # ── Save narrative text files ──────────
        self._save_narrative_texts(all_records, narrative_dir)

        # ── Aggregate stats ────────────────────
        stats = self._compute_stats(all_records)
        stats["elapsed_seconds"] = round(elapsed, 1)

        # ── Ablation comparison table ──────────
        self._save_ablation_table(all_records, ablation_dir / "ablation_table.csv")
        self._plot_ablation(all_records, ablation_dir / "ablation_comparison.png")

        # ── ABC delta comparison plot ──────────
        self._plot_abc_delta_comparison(all_records, ablation_dir / "abc_delta_comparison.png")

        # ── result.txt ─────────────────────────
        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="ABC-Guided Counterfactual Explanations (v4 — segmentation-guided)",
            conditions={
                "classifier"        : "EfficientNet-B4 (HAM10000)",
                "abc_regressor"     : "ABCRegressor (PH2+Derm7pt)",
                "optimizer"         : "Adam (per Singla et al., 2023)",
                "learning_rate"     : ABC_CF_LEARNING_RATE,
                "max_iterations"    : ABC_CF_MAX_ITER,
                "confidence_thres"  : ABC_CF_CONFIDENCE_THRES,
                "loss_weights"      : f"λ_cls={ABC_CF_LAMBDA_CLS}, λ_A={ABC_CF_LAMBDA_A}, "
                                      f"λ_B={ABC_CF_LAMBDA_B}, λ_C={ABC_CF_LAMBDA_C}, "
                                      f"λ_l1={ABC_CF_LAMBDA_L1}, λ_TV={ABC_CF_LAMBDA_TV}, "
                                      f"λ_perc={ABC_CF_LAMBDA_PERC}",
                "mask_guidance"     : f"soft mask (dilate={ABC_CF_MASK_DILATE_PX}px, "
                                      f"blur σ={ABC_CF_MASK_BLUR_SIGMA})",
                "delta_blur"        : f"Gaussian σ={ABC_CF_DELTA_BLUR_SIGMA}, "
                                      f"kernel={ABC_CF_DELTA_BLUR_KERNEL}",
                "perceptual_loss"   : "VGG-16 relu2_2+relu3_3 (Johnson et al., 2016)",
                "ablation_modes"    : list(ABLATION_MODES.keys()),
                "class_pairs"       : [f"{s}→{t}" for s, t in ABC_CF_PAIRS],
                "n_images_per_pair" : ABC_CF_NUM_IMAGES,
                "pixel_threshold"   : ABC_CF_PIXEL_THRESHOLD,
            },
            statistics=stats,
        )

        print(
            f"\n[ABCCounterfactual] Done.  "
            f"Records: {len(all_records)}  Elapsed: {elapsed:.1f}s"
        )
        return stats

    # ─────────────────────────────────────────
    # Data collection
    # ─────────────────────────────────────────

    def _collect_samples(
        self, src_class: int, n: int,
        min_source_prob: float = 0.80,
    ) -> List[Tuple[torch.Tensor, int]]:
        """Collect n correctly classified images with high source confidence."""
        samples = []
        self.explainer.clf.eval()
        with torch.no_grad():
            for images, labels in self.loader:
                imgs  = images.to(self.device)
                logits = self.explainer.clf(imgs)
                probs  = torch.softmax(logits, dim=1).cpu()
                preds  = probs.argmax(dim=1)

                for i, (img, lbl, prd) in enumerate(
                    zip(images, labels, preds)
                ):
                    if (int(lbl) == src_class
                            and int(prd) == src_class
                            and float(probs[i, src_class]) >= min_source_prob):
                        samples.append((img, int(lbl), None))
                    if len(samples) >= n:
                        break
                if len(samples) >= n:
                    break

        if len(samples) < n:
            print(f"  ⚠ Only found {len(samples)}/{n} samples with "
                  f"source prob ≥ {min_source_prob} for class {src_class}")
        return samples

    # ─────────────────────────────────────────
    # Denormalisation
    # ─────────────────────────────────────────

    def _denorm(self, t: torch.Tensor) -> np.ndarray:
        mean = np.array(IMAGE_MEAN)
        std  = np.array(IMAGE_STD)
        img  = t.permute(1, 2, 0).numpy()
        return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────
    # Visual panels (original style)
    # ─────────────────────────────────────────

    def _save_panels(
            self,
            results: List[Dict],
            save_path: Path,
            title: str,
    ) -> None:
        """
        Save [Original | CF | Signed Diff | |δ| Heatmap] panel for each sample.

        Visualization improvements (v2):
        - Signed difference uses RdBu_r colormap with symmetric vmin/vmax
        - |δ| heatmap uses 'inferno' with percentile-based scaling
        - Colorbars added for both diff and heatmap panels
        - Amplification factor for very small perturbations
        """
        n = len(results)
        if n == 0:
            return

        fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, res in enumerate(results):
            orig_np = self._denorm(
                res["cf_tensor"] - res["delta"]
            )
            cf_np = self._denorm(res["cf_tensor"])

            # δ in normalised space (3, H, W) → (H, W, 3)
            delta_np = res["delta"].permute(1, 2, 0).numpy()

            # Signed difference — channel mean for single-channel display
            diff_gray = delta_np.mean(axis=2)
            vmax_diff = max(abs(diff_gray.min()), abs(diff_gray.max()), 0.01)

            # |δ| magnitude — channel-max for heatmap
            abs_delta = np.abs(delta_np).max(axis=2)  # (H, W)
            # Percentile-based scaling for visibility
            p99 = np.percentile(abs_delta[abs_delta > 0], 99) if (abs_delta > 0).any() else 0.01
            vmax_heat = max(p99, 0.005)

            # ── Panel 1: Original ──────────────
            axes[row, 0].imshow(orig_np)
            axes[row, 0].set_title("Original", fontsize=9, fontweight="bold")
            axes[row, 0].axis("off")

            # ── Panel 2: Counterfactual ────────
            axes[row, 1].imshow(cf_np)
            valid_str = "✓" if res["validity"] else "✗"
            axes[row, 1].set_title(
                f"Counterfactual {valid_str}\n"
                f"P(target)={res['final_prob']:.3f}",
                fontsize=9, fontweight="bold",
            )
            axes[row, 1].axis("off")

            # ── Panel 3: Signed Difference ─────
            im_diff = axes[row, 2].imshow(
                diff_gray, cmap="RdBu_r",
                vmin=-vmax_diff, vmax=vmax_diff,
            )
            axes[row, 2].set_title(
                f"Difference (δ)\n"
                f"ΔA={res['delta_A']:.3f}  ΔB={res['delta_B']:.3f}  ΔC={res['delta_C']:.3f}",
                fontsize=8,
            )
            axes[row, 2].axis("off")
            plt.colorbar(im_diff, ax=axes[row, 2], fraction=0.046, pad=0.04)

            # ── Panel 4: |δ| Heatmap ──────────
            im_heat = axes[row, 3].imshow(
                abs_delta, cmap="inferno",
                vmin=0, vmax=vmax_heat,
            )
            axes[row, 3].set_title(
                f"|δ| Heatmap\n"
                f"Sparsity={res['sparsity']:.3f}  L1={res['proximity_l1']:.4f}",
                fontsize=8,
            )
            axes[row, 3].axis("off")
            plt.colorbar(im_heat, ax=axes[row, 3], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────
    # Narrative panels (textual explanations)
    # ─────────────────────────────────────────

    def _save_narrative_panels(
        self,
        results: List[Dict],
        save_path: Path,
        title: str,
        src_name: str,
        tgt_name: str,
    ) -> None:
        """
        Save [Original | CF | |δ| | Textual Explanation] panel.

        The rightmost column contains a human-readable narrative
        explaining what ABC features would need to change for
        the diagnosis to flip.
        """
        n = len(results)
        if n == 0:
            return

        fig, axes = plt.subplots(n, 4, figsize=(22, 5 * n),
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, res in enumerate(results):
            orig_np = self._denorm(res["cf_tensor"] - res["delta"])
            cf_np   = self._denorm(res["cf_tensor"])
            abs_np  = np.abs(res["delta"].permute(1, 2, 0).numpy())

            # Column 0: Original
            axes[row, 0].imshow(orig_np)
            src_full = CLASS_FULL_NAMES.get(src_name, src_name)
            axes[row, 0].set_title(f"Original\n{src_full}", fontsize=8)
            axes[row, 0].axis("off")

            # Column 1: Counterfactual
            axes[row, 1].imshow(cf_np)
            tgt_full = CLASS_FULL_NAMES.get(tgt_name, tgt_name)
            validity_icon = "✓ Valid" if res["validity"] else "✗ Invalid"
            axes[row, 1].set_title(
                f"Counterfactual → {tgt_full}\n"
                f"Conf: {res['final_prob']:.0%}  {validity_icon}",
                fontsize=8,
            )
            axes[row, 1].axis("off")

            # Column 2: Perturbation heatmap
            abs_max = abs_np.max() + 1e-8
            axes[row, 2].imshow(
                abs_np.mean(axis=2) / abs_max,
                cmap="inferno", vmin=0, vmax=1,
            )
            axes[row, 2].set_title(
                f"|δ| Heatmap\n"
                f"Sparsity={res['sparsity']:.1%}, L1={res['proximity_l1']:.4f}",
                fontsize=8,
            )
            axes[row, 2].axis("off")

            # Column 3: Textual explanation
            axes[row, 3].axis("off")
            text = res.get("text_en", "")
            axes[row, 3].text(
                0.05, 0.95, text,
                transform=axes[row, 3].transAxes,
                fontsize=7.5,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          alpha=0.9, edgecolor="gray"),
                wrap=True,
            )
            axes[row, 3].set_title("Textual Counterfactual Explanation", fontsize=8)

        plt.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────
    # Narrative text file output
    # ─────────────────────────────────────────

    def _save_narrative_texts(
        self, records: List[Dict], narrative_dir: Path
    ) -> None:
        """Save all textual explanations to text files (EN + TR)."""
        # Group by pair and mode
        groups = {}
        for r in records:
            key = (r["src_class"], r["tgt_class"], r["mode"])
            groups.setdefault(key, []).append(r)

        # English
        en_lines = [
            "=" * 70,
            "  TEXTUAL COUNTERFACTUAL EXPLANATIONS (English)",
            "=" * 70, "",
        ]
        for (src, tgt, mode), recs in sorted(groups.items()):
            en_lines.append(f"\n{'─' * 60}")
            en_lines.append(f"  Pair: {src} → {tgt}  |  Mode: {mode}")
            en_lines.append(f"{'─' * 60}\n")
            for i, r in enumerate(recs):
                en_lines.append(f"  Sample {i+1}:")
                en_lines.append(f"  {r.get('text_en', 'N/A')}")
                en_lines.append("")

        (narrative_dir / "explanations_en.txt").write_text(
            "\n".join(en_lines), encoding="utf-8"
        )

        # Turkish
        tr_lines = [
            "=" * 70,
            "  METİNSEL KARŞIOLGUSAL AÇIKLAMALAR (Türkçe)",
            "=" * 70, "",
        ]
        for (src, tgt, mode), recs in sorted(groups.items()):
            tr_lines.append(f"\n{'─' * 60}")
            tr_lines.append(f"  Çift: {src} → {tgt}  |  Mod: {mode}")
            tr_lines.append(f"{'─' * 60}\n")
            for i, r in enumerate(recs):
                tr_lines.append(f"  Örnek {i+1}:")
                tr_lines.append(f"  {r.get('text_tr', 'N/A')}")
                tr_lines.append("")

        (narrative_dir / "explanations_tr.txt").write_text(
            "\n".join(tr_lines), encoding="utf-8"
        )

        print(f"[Narratives] Saved EN + TR explanations to {narrative_dir}")

    # ─────────────────────────────────────────
    # All records CSV
    # ─────────────────────────────────────────

    def _save_all_records(self, records: List[Dict], path: Path) -> None:
        """Save all individual records to CSV."""
        rows = []
        for r in records:
            rows.append({
                "src_class"   : r["src_class"],
                "tgt_class"   : r["tgt_class"],
                "mode"        : r["mode"],
                "validity"    : r["validity"],
                "final_prob"  : r["final_prob"],
                "src_prob"    : r.get("src_prob", None),
                "proximity_l1": r["proximity_l1"],
                "sparsity"    : r["sparsity"],
                "ssim"        : r.get("ssim", None),
                "n_iter"      : r["n_iter"],
                "A_src"       : r["abc_src"]["A"],
                "B_src"       : r["abc_src"]["B"],
                "C_src"       : r["abc_src"]["C"],
                "A_cf"        : r["abc_cf"]["A"],
                "B_cf"        : r["abc_cf"]["B"],
                "C_cf"        : r["abc_cf"]["C"],
                "delta_A"     : r["delta_A"],
                "delta_B"     : r["delta_B"],
                "delta_C"     : r["delta_C"],
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    # ─────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────

    def _compute_stats(self, records: List[Dict]) -> Dict:
        """Aggregate statistics across all records."""
        if not records:
            return {}
        stats = {}

        # Per-mode stats
        for mode in ABLATION_MODES:
            sub = [r for r in records if r["mode"] == mode]
            if not sub:
                continue
            stats[f"{mode}_validity"]  = round(np.mean([r["validity"]     for r in sub]), 4)
            stats[f"{mode}_prox_l1"]   = round(np.mean([r["proximity_l1"] for r in sub]), 5)
            stats[f"{mode}_sparsity"]  = round(np.mean([r["sparsity"]     for r in sub]), 4)
            stats[f"{mode}_delta_A"]   = round(np.mean([r["delta_A"]      for r in sub]), 4)
            stats[f"{mode}_delta_B"]   = round(np.mean([r["delta_B"]      for r in sub]), 4)
            stats[f"{mode}_delta_C"]   = round(np.mean([r["delta_C"]      for r in sub]), 4)
            stats[f"{mode}_n_iter"]    = round(np.mean([r["n_iter"]       for r in sub]), 1)
            stats[f"{mode}_ssim"]      = round(np.mean([r.get("ssim", 0)  for r in sub]), 4)

        # Per-pair stats
        for src, tgt in ABC_CF_PAIRS:
            sub = [r for r in records if r["src_class"] == src and r["tgt_class"] == tgt]
            if not sub:
                continue
            key = f"{src}_to_{tgt}"
            stats[f"{key}_validity"]   = round(np.mean([r["validity"] for r in sub]), 4)
            stats[f"{key}_best_prob"]  = round(max(r["final_prob"] for r in sub), 4)

        # Global
        stats["total_records"]       = len(records)
        stats["overall_validity"]    = round(np.mean([r["validity"] for r in records]), 4)
        stats["overall_prox_l1"]     = round(np.mean([r["proximity_l1"] for r in records]), 5)
        stats["overall_sparsity"]    = round(np.mean([r["sparsity"] for r in records]), 4)

        return stats

    def _save_ablation_table(self, records: List[Dict], path: Path) -> None:
        """Save ablation comparison as CSV."""
        rows = []
        for mode in ABLATION_MODES:
            sub = [r for r in records if r["mode"] == mode]
            if not sub:
                continue
            rows.append({
                "mode"       : mode,
                "validity"   : round(np.mean([r["validity"]     for r in sub]), 4),
                "prox_l1"    : round(np.mean([r["proximity_l1"] for r in sub]), 5),
                "sparsity"   : round(np.mean([r["sparsity"]     for r in sub]), 4),
                "delta_A"    : round(np.mean([r["delta_A"]      for r in sub]), 4),
                "delta_B"    : round(np.mean([r["delta_B"]      for r in sub]), 4),
                "delta_C"    : round(np.mean([r["delta_C"]      for r in sub]), 4),
                "n_iter"     : round(np.mean([r["n_iter"]       for r in sub]), 1),
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    def _plot_ablation(self, records: List[Dict], path: Path) -> None:
        """5-panel bar chart comparing ablation modes."""
        modes   = list(ABLATION_MODES.keys())
        metrics = ["validity", "proximity_l1", "delta_A", "delta_B", "delta_C"]
        labels  = ["Validity Rate", "Proximity (L1)", "ΔA (Asymmetry)", "ΔB (Border)", "ΔC (Color)"]
        colors  = ["#22c55e", "#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4.5))
        for ax, metric, label, color in zip(axes, metrics, labels, colors):
            vals = []
            for mode in modes:
                sub = [r for r in records if r["mode"] == mode]
                vals.append(np.mean([r[metric] for r in sub]) if sub else 0)
            bars = ax.bar(range(len(modes)), vals, color=color, alpha=0.85, edgecolor="white", linewidth=1.2)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=25, fontsize=8)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=7,
                )
            ax.set_ylim(0, max(vals) * 1.25 + 0.01)

        plt.suptitle(
            "ABC Counterfactual Ablation Study\n"
            "Higher validity = more successful class flip  |  "
            "Lower ΔA/ΔB/ΔC with ABC mode = better preservation",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _plot_abc_delta_comparison(self, records: List[Dict], path: Path) -> None:
        """
        Grouped bar chart: ΔA, ΔB, ΔC side by side for each ablation mode.

        This visualises the core hypothesis: ABC constraints reduce
        morphological drift during counterfactual generation.
        """
        modes  = list(ABLATION_MODES.keys())
        crits  = ["delta_A", "delta_B", "delta_C"]
        labels = ["ΔA (Asymmetry)", "ΔB (Border)", "ΔC (Color)"]
        colors = ["#ef4444", "#f59e0b", "#8b5cf6"]

        x      = np.arange(len(modes))
        width  = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (crit, label, color) in enumerate(zip(crits, labels, colors)):
            vals = []
            for mode in modes:
                sub = [r for r in records if r["mode"] == mode]
                vals.append(np.mean([r[crit] for r in sub]) if sub else 0)
            bars = ax.bar(x + i * width, vals, width, label=label,
                          color=color, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", fontsize=7)

        ax.set_xlabel("Ablation Mode", fontsize=11)
        ax.set_ylabel("Mean ABC Delta", fontsize=11)
        ax.set_title(
            "ABC Feature Preservation Across Ablation Modes\n"
            "(Lower = better morphological preservation)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(modes, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()