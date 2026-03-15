"""
src/abc/ordinal_loss.py
-----------------------
Ordinal-aware loss functions for ABC score regression.

ABC scores are derived from ordinal clinical annotations (e.g., asymmetry 0/1/2,
pigment network absent/typical/atypical). Treating these as naive continuous
targets with MSE discards rank structure. Ordinal-aware losses respect this.

Implementations
---------------
1. CORAL (Consistent Rank Logits)
   Cao, Mirjalili & Raschka (2020). Rank consistent ordinal regression for
   neural networks with application to age estimation. Pattern Recognition
   Letters, 140, 11-16. https://doi.org/10.1016/j.patrec.2020.09.021

2. SORD (Soft Labels for Ordinal Regression)
   Díaz & Marathe (2019). Soft labels for ordinal regression.
   CVPR 2019. https://doi.org/10.1109/CVPR.2019.00491

3. OrdinalHuber (custom)
   Huber loss + ordinal rank regularisation. Penalises rank inversions
   more than small metric errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────
# SORD: Soft Labels for Ordinal Regression
# ─────────────────────────────────────────────

class SORDLoss(nn.Module):
    """
    SORD: converts ordinal integer labels to soft Gaussian distributions
    over K ordinal bins, then applies KL divergence.

    Each ABC score is discretised into NUM_BINS bins ∈ [0, 1].
    The ground-truth label y is placed at bin b = round(y * (K-1))
    and softened with a Gaussian of bandwidth σ.

    Parameters
    ----------
    num_bins : int
        Number of ordinal bins (default: 5 → {0.0, 0.25, 0.5, 0.75, 1.0}).
    sigma : float
        Gaussian bandwidth for label smoothing.

    References
    ----------
    Díaz & Marathe (2019), CVPR.
    """

    def __init__(self, num_bins: int = 5, sigma: float = 1.5):
        super().__init__()
        self.K     = num_bins
        self.sigma = sigma
        # Bin centres: [0/(K-1), 1/(K-1), ..., 1.0]
        self.register_buffer(
            "bins",
            torch.linspace(0.0, 1.0, num_bins)   # (K,)
        )

    def _soft_targets(self, y_cont: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous [0,1] targets to soft label distributions.

        Parameters
        ----------
        y_cont : (N,)  continuous scores in [0, 1]

        Returns
        -------
        (N, K) soft probability distributions
        """
        # Nearest bin index for each target
        y_idx = (y_cont * (self.K - 1)).round().long().clamp(0, self.K - 1)
        # Gaussian weights
        dist  = (torch.arange(self.K, device=y_cont.device).float()
                 - y_idx.float().unsqueeze(1)) ** 2           # (N, K)
        soft  = torch.exp(-dist / (2 * self.sigma ** 2))
        return soft / soft.sum(dim=1, keepdim=True)            # (N, K) normalised

    def forward(
        self,
        logits: torch.Tensor,    # (N, 3, K) raw logits per criterion per bin
        targets: torch.Tensor,   # (N, 3)   continuous targets in [0,1]
    ) -> torch.Tensor:
        N, C, K = logits.shape
        total_loss = 0.0
        for c in range(C):
            soft = self._soft_targets(targets[:, c])          # (N, K)
            log_p = F.log_softmax(logits[:, c, :], dim=1)     # (N, K)
            total_loss = total_loss + F.kl_div(
                log_p, soft, reduction="batchmean"
            )
        return total_loss / C

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to expected continuous scores.

        Parameters
        ----------
        logits : (N, 3, K)

        Returns
        -------
        (N, 3) continuous scores in [0, 1]
        """
        probs = F.softmax(logits, dim=2)                       # (N, 3, K)
        bins  = self.bins.to(logits.device)                    # (K,)
        return (probs * bins.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (N, 3)


# ─────────────────────────────────────────────
# OrdinalHuber: Huber + rank regularisation
# ─────────────────────────────────────────────

class OrdinalHuberLoss(nn.Module):
    """
    Smooth L1 (Huber) loss with an additional ordinal rank-consistency
    regularisation term.

    L = Huber(pred, target) + λ_rank * RankPenalty(pred, target)

    RankPenalty penalises cases where the model assigns a higher score
    to a sample with a lower ground-truth rank.

    Parameters
    ----------
    beta : float
        Huber transition point (default: 0.1).
    lambda_rank : float
        Weight of the rank consistency penalty (default: 0.1).
    """

    def __init__(self, beta: float = 0.1, lambda_rank: float = 0.1):
        super().__init__()
        self.beta        = beta
        self.lambda_rank = lambda_rank
        self.huber       = nn.SmoothL1Loss(beta=beta)

    def forward(
        self,
        pred: torch.Tensor,     # (N, 3)
        target: torch.Tensor,   # (N, 3)
    ) -> torch.Tensor:
        huber_loss = self.huber(pred, target)

        # Rank penalty: for random pairs (i, j) in the batch
        rank_loss = torch.tensor(0.0, device=pred.device)
        if pred.shape[0] > 1 and self.lambda_rank > 0:
            # For each criterion, penalise rank violations
            for c in range(pred.shape[1]):
                p = pred[:, c]
                t = target[:, c]
                # Pairwise differences
                dp = p.unsqueeze(0) - p.unsqueeze(1)        # (N, N)
                dt = t.unsqueeze(0) - t.unsqueeze(1)        # (N, N)
                # Penalise: dt > 0 but dp < 0 (rank violation)
                violation = F.relu(-dp * dt.sign())
                rank_loss = rank_loss + violation.mean()
            rank_loss = rank_loss / pred.shape[1]

        return huber_loss + self.lambda_rank * rank_loss


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_loss(loss_type: str = "ordinal_huber", **kwargs):
    """
    Build the ABC regression loss.

    Parameters
    ----------
    loss_type : str
        'sord'          — SORD soft-label KL loss (best for ordinal labels)
        'ordinal_huber' — Huber + rank penalty (good baseline)
        'huber'         — Plain Huber (original)

    Returns
    -------
    nn.Module
    """
    if loss_type == "sord":
        return SORDLoss(
            num_bins=kwargs.get("num_bins", 5),
            sigma=kwargs.get("sigma", 1.5),
        )
    elif loss_type == "ordinal_huber":
        return OrdinalHuberLoss(
            beta=kwargs.get("beta", 0.1),
            lambda_rank=kwargs.get("lambda_rank", 0.1),
        )
    elif loss_type == "huber":
        return nn.SmoothL1Loss(beta=kwargs.get("beta", 0.1))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
