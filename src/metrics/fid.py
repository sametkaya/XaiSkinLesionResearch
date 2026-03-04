"""
metrics/fid.py
--------------
Fréchet Inception Distance (FID) for evaluating counterfactual image quality.

FID measures the distributional distance between a set of real images and
a set of generated (counterfactual) images in the feature space of a
pre-trained Inception-v3 network.  Lower FID indicates more realistic,
higher-quality generated images.

This implementation computes FID between the original test images and their
counterfactual counterparts to quantify how much the distribution shifts
during counterfactual generation.

Reference:
    Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S.
    (2017). GANs trained by a two time-scale update rule converge to a local
    Nash equilibrium. NeurIPS 2017.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms as T


class FIDCalculator:
    """
    Computes Fréchet Inception Distance using InceptionV3 feature embeddings.

    Features are extracted from the pool_3 (2048-d) layer of InceptionV3.

    Parameters
    ----------
    device : torch.device
    batch_size : int
    """

    def __init__(self, device: torch.device, batch_size: int = 64):
        self.device     = device
        self.batch_size = batch_size

        # Load InceptionV3 and extract features (up to pool_3)
        inception        = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        inception.aux_logits = False
        # Replace final fully connected layer with identity
        inception.fc     = nn.Identity()
        inception        = inception.to(device).eval()
        self.inception   = inception

        self.transform = T.Compose([
            T.Resize((299, 299)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _extract_features(self, tensors: List[torch.Tensor]) -> np.ndarray:
        """
        Extract 2048-d InceptionV3 pool_3 features from a list of image tensors.

        Parameters
        ----------
        tensors : List[torch.Tensor]
            List of image tensors (3, H, W) already in [0, 1] normalised form.

        Returns
        -------
        np.ndarray
            Feature matrix of shape (N, 2048).
        """
        all_feats = []
        for i in range(0, len(tensors), self.batch_size):
            batch    = torch.stack(tensors[i : i + self.batch_size]).to(self.device)
            # Ensure values in [0, 1]
            batch    = torch.clamp(batch, 0, 1)
            batch    = self.transform(batch)
            feats    = self.inception(batch)         # (B, 2048)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    @staticmethod
    def _frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """
        Compute the Fréchet distance between two Gaussians.

        FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

        Parameters
        ----------
        mu1, mu2 : np.ndarray  – mean vectors (D,)
        sigma1, sigma2 : np.ndarray  – covariance matrices (D, D)

        Returns
        -------
        float
        """
        diff      = mu1 - mu2
        cov_sqrt, _ = sqrtm(sigma1 @ sigma2, disp=False)  # complex output possible

        # Numerical stability: discard small imaginary parts
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real

        fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * cov_sqrt))
        return fid

    def compute(
        self,
        real_tensors: List[torch.Tensor],
        fake_tensors: List[torch.Tensor],
    ) -> float:
        """
        Compute FID between real and generated/counterfactual images.

        Parameters
        ----------
        real_tensors : List[torch.Tensor]
            Real test images (3, H, W).
        fake_tensors : List[torch.Tensor]
            Counterfactual images (3, H, W).

        Returns
        -------
        float
            FID score (lower = more realistic counterfactuals).
        """
        if len(real_tensors) < 2 or len(fake_tensors) < 2:
            print("[FID] Too few samples to compute FID (need ≥ 2 each). Returning NaN.")
            return float("nan")

        real_feats = self._extract_features(real_tensors)
        fake_feats = self._extract_features(fake_tensors)

        mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
        mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

        fid = self._frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
        return round(fid, 4)
