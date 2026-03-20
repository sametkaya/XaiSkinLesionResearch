"""
src/explainers/cf_losses.py
----------------------------
Counterfactual-specific loss functions and utilities.

Based on state-of-the-art research (2022-2025):
  - Negative LPIPS loss (Jeanneret et al., ACCV 2022 / CVPR 2023)
  - Total Variation regularization (Mahendran & Vedaldi, CVPR 2015)
  - Low-pass frequency filtering (Guo et al., UAI 2019)
  - Saliency-based δ initialization (Singla et al., MedIA 2023)

References
----------
Zhang, R., et al. (2018). The Unreasonable Effectiveness of Deep Features
    as a Perceptual Metric. CVPR 2018.
Jeanneret, G., et al. (2022). Diffusion Models for Counterfactual
    Explanations. ACCV 2022.
Guo, C., et al. (2019). Low Frequency Adversarial Perturbation. UAI 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# LPIPS Perceptual Loss (lightweight VGG-based)
# ─────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Lightweight VGG-16 perceptual loss using relu2_2 and relu3_3 features.
    
    Does NOT require the `lpips` package — uses raw VGG features directly.
    This avoids dependency issues while providing perceptual distance.
    
    Parameters
    ----------
    device : torch.device
    layers : list of str
        VGG layers to extract features from.
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        
        # relu2_2 = features[8], relu3_3 = features[15]
        self.blocks = nn.ModuleList([
            vgg[:9],    # → relu2_2
            vgg[9:16],  # → relu3_3
        ])
        
        # VGG expects ImageNet normalization
        self.register_buffer("vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("vgg_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from HAM10000 normalization to VGG normalization."""
        # First denormalize from HAM10000 stats
        ham_mean = torch.tensor([0.7630, 0.5456, 0.5701]).view(1, 3, 1, 1).to(x.device)
        ham_std  = torch.tensor([0.1409, 0.1526, 0.1694]).view(1, 3, 1, 1).to(x.device)
        x_01 = x * ham_std + ham_mean  # → [0, 1]
        x_01 = x_01.clamp(0, 1)
        # Then normalize to VGG stats
        return (x_01 - self.vgg_mean) / self.vgg_std
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual distance between x and y.
        
        Returns
        -------
        torch.Tensor
            Scalar perceptual distance (lower = more similar).
        """
        x_vgg = self._normalize_for_vgg(x)
        y_vgg = self._normalize_for_vgg(y)
        
        loss = 0.0
        feat_x, feat_y = x_vgg, y_vgg
        for block in self.blocks:
            feat_x = block(feat_x)
            feat_y = block(feat_y)
            # Normalize features before computing distance
            loss += F.mse_loss(
                F.instance_norm(feat_x),
                F.instance_norm(feat_y),
            )
        return loss


# ─────────────────────────────────────────────
# Total Variation Loss
# ─────────────────────────────────────────────

def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic total variation of the perturbation δ.
    
    Encourages spatially smooth perturbations — visible blobs
    rather than invisible pixel noise.
    
    Parameters
    ----------
    delta : torch.Tensor (B, C, H, W)
    
    Returns
    -------
    torch.Tensor
        Scalar TV loss.
    """
    diff_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    diff_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


# ─────────────────────────────────────────────
# Low-Pass Frequency Filter
# ─────────────────────────────────────────────

def low_pass_filter(delta: torch.Tensor, cutoff_ratio: float = 0.25) -> torch.Tensor:
    """
    Apply a circular low-pass filter in Fourier domain.
    
    Removes high-frequency components from δ that are imperceptible
    to humans but effective at fooling CNNs. Forces the optimizer
    to find low-frequency (visible) perturbations.
    
    Reference: Guo et al. (2019). Low Frequency Adversarial Perturbation.
    
    Parameters
    ----------
    delta : torch.Tensor (B, C, H, W)
    cutoff_ratio : float
        Fraction of frequency spectrum to keep (0.25 = keep lowest 25%).
    
    Returns
    -------
    torch.Tensor
        Filtered δ with same shape.
    """
    B, C, H, W = delta.shape
    
    # 2D FFT
    delta_fft = torch.fft.fft2(delta)
    delta_fft = torch.fft.fftshift(delta_fft)
    
    # Create circular mask
    cy, cx = H // 2, W // 2
    radius = int(min(H, W) * cutoff_ratio / 2)
    
    y_grid = torch.arange(H, device=delta.device).float() - cy
    x_grid = torch.arange(W, device=delta.device).float() - cx
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    mask = (xx**2 + yy**2 <= radius**2).float()
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Apply mask
    delta_fft = delta_fft * mask
    
    # Inverse FFT
    delta_fft = torch.fft.ifftshift(delta_fft)
    delta_filtered = torch.fft.ifft2(delta_fft).real
    
    return delta_filtered


# ─────────────────────────────────────────────
# Saliency-Based δ Initialization
# ─────────────────────────────────────────────

@torch.enable_grad()
def saliency_init(
    classifier: nn.Module,
    image: torch.Tensor,
    target_class: int,
    scale: float = 0.1,
) -> torch.Tensor:
    """
    Initialize δ from the gradient of target class logit w.r.t. input.
    
    Instead of starting from zero (which biases toward trivial solutions),
    start from the direction that increases target class probability.
    
    Parameters
    ----------
    classifier : nn.Module
    image : torch.Tensor (1, C, H, W) — requires_grad will be set
    target_class : int
    scale : float
        Scale factor for the saliency initialization.
    
    Returns
    -------
    torch.Tensor (1, C, H, W)
        Initial δ scaled to `scale` of the saliency maximum.
    """
    x = image.clone().detach().requires_grad_(True)
    logits = classifier(x)
    target_logit = logits[0, target_class]
    target_logit.backward()
    
    grad = x.grad.detach()
    
    # Scale to desired magnitude
    grad_max = grad.abs().max()
    if grad_max > 0:
        delta_init = grad / grad_max * scale
    else:
        delta_init = torch.zeros_like(grad)
    
    return delta_init.detach()


# ─────────────────────────────────────────────
# Minimum Perturbation Hinge Loss
# ─────────────────────────────────────────────

def min_perturbation_hinge(delta: torch.Tensor, tau: float = 0.03) -> torch.Tensor:
    """
    Hinge loss that penalizes perturbations BELOW a minimum threshold.
    
    Ensures the counterfactual is visually different from the original.
    L = max(0, τ − mean(|δ|))
    
    Parameters
    ----------
    delta : torch.Tensor
    tau : float
        Minimum required average pixel change.
    
    Returns
    -------
    torch.Tensor
        Scalar hinge loss (0 if perturbation is large enough).
    """
    return F.relu(tau - delta.abs().mean())
