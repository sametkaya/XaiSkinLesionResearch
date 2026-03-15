"""
src/abc/abc_model.py
--------------------
ABC Regressor model.

Architecture
------------
EfficientNet-B0 backbone (pretrained on ImageNet, optionally fine-tuned
from a HAM10000 classifier checkpoint) followed by a multi-output
regression head predicting A, B, C scores ∈ [0, 1].

Training strategy:
  Phase 1 (ABC_FREEZE_EPOCHS): Backbone frozen — only regression head trained.
  Phase 2 (remaining epochs):  Full network fine-tuned end-to-end.

This two-phase strategy is especially important given the small size of
PH2+Derm7pt (~2,200 images combined) relative to the backbone's capacity.
Freezing prevents catastrophic forgetting of dermoscopy-relevant features
already encoded in the HAM10000 pretrained weights.

Loss function
-------------
We use Huber (Smooth L1) loss rather than MSE because:
  1. ABC scores are ordinal-derived and contain label noise from the
     PH2→ABC and Derm7pt→ABC normalisation mapping.
  2. Huber is less sensitive to outlier scores at the extremes (0 or 1).

References
----------
Tan, M., & Le, Q. (2019).
    EfficientNet: Rethinking model scaling for CNNs. ICML 2019.

Huber, P. J. (1964).
    Robust estimation of a location parameter.
    Annals of Mathematical Statistics, 35(1), 73–101.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from src.abc.config_abc import (
    IMAGE_SIZE, ABC_DROPOUT, NUM_ABC,
)


class ABCRegressor(nn.Module):
    """
    EfficientNet-B0 based multi-output regression model for ABC scoring.

    Predicts three dermoscopic scores simultaneously:
      - A (Asymmetry)          ∈ [0, 1]
      - B (Border Irregularity)∈ [0, 1]
      - C (Color Variegation)  ∈ [0, 1]

    Parameters
    ----------
    backbone_weights : str or None
        'IMAGENET1K_V1' for ImageNet pretraining, or None for random init.
    freeze_backbone : bool
        If True, freeze all backbone parameters during initialisation.
        Use set_backbone_trainable(True) to unfreeze later.
    dropout_rate : float
        Dropout before the regression head.
    num_outputs : int
        Number of regression targets (default: 3 for A, B, C).
    """

    def __init__(
        self,
        backbone_weights: Optional[str] = "IMAGENET1K_V1",
        freeze_backbone: bool = True,
        dropout_rate: float = ABC_DROPOUT,
        num_outputs: int = NUM_ABC,
        num_bins: int = 5,
    ):
        super().__init__()

        # EfficientNet-B0 backbone
        base = models.efficientnet_b0(weights=backbone_weights)

        # Remove the original classifier head, keep feature extractor
        # EfficientNet-B0 produces 1280-dim features after adaptive pooling
        self.features = base.features       # CNN feature extractor
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.num_bins = num_bins
        self.use_sord = (num_bins > 1)

        if self.use_sord:
            # SORD head: output K logits per criterion → (B, num_outputs, K)
            # Ordinal-aware: respects rank structure of ABC annotations
            # Reference: Díaz & Marathe (CVPR 2019)
            self.feat_head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(1280, 512),
                nn.GELU(),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, 256),
                nn.GELU(),
            )
            self.ordinal_head = nn.Linear(256, num_outputs * num_bins)
        else:
            # Standard continuous regression head with Sigmoid output
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(1280, 512),
                nn.GELU(),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, num_outputs),
                nn.Sigmoid(),
            )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone (feature extractor) parameters."""
        for param in self.features.parameters():
            param.requires_grad = False

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        """
        Set backbone parameters as trainable or frozen.

        Parameters
        ----------
        trainable : bool
            True = unfreeze backbone for full fine-tuning.
            False = re-freeze backbone.
        """
        for param in self.features.parameters():
            param.requires_grad = trainable
        status = "unfrozen (fine-tuning)" if trainable else "frozen"
        print(f"[ABCRegressor] Backbone {status}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, 3, H, W)

        Returns
        -------
        If use_sord: torch.Tensor (B, 3, K) — logits for ordinal bins
        Else:        torch.Tensor (B, 3)    — continuous scores in [0, 1]
        """
        feats  = self.features(x)
        pooled = self.pool(feats)

        if self.use_sord:
            feat  = self.feat_head(pooled)
            logits = self.ordinal_head(feat)               # (B, 3*K)
            return logits.view(x.shape[0], -1, self.num_bins)  # (B, 3, K)
        else:
            return self.head(pooled)

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_abc_regressor(
    device: torch.device,
    ham_checkpoint: Optional[Path] = None,
    freeze_backbone: bool = True,
) -> ABCRegressor:
    """
    Build ABCRegressor and optionally initialise backbone from a
    HAM10000 classifier checkpoint.

    Loading the HAM10000 backbone weights provides a warm-start with
    dermoscopy-specific features, which is critical for good ABC regression
    on the small PH2+Derm7pt training set.

    Parameters
    ----------
    device : torch.device
    ham_checkpoint : Path or None
        Path to best_model.pth from HAM10000 classifier training.
        Only the backbone (features) weights are transferred; the
        classification head is discarded.
    freeze_backbone : bool
        Freeze backbone at initialisation (default: True for Phase 1).

    Returns
    -------
    ABCRegressor on device
    """
    from src.abc.config_abc import ABC_LOSS_TYPE, ABC_ORDINAL_BINS
    use_sord = (ABC_LOSS_TYPE == "sord")
    model = ABCRegressor(
        backbone_weights="IMAGENET1K_V1",
        freeze_backbone=freeze_backbone,
        num_bins=ABC_ORDINAL_BINS if use_sord else 1,
    )

    if ham_checkpoint is not None and Path(ham_checkpoint).exists():
        state = torch.load(ham_checkpoint, map_location="cpu")

        # HAM checkpoint stores full SkinLesionClassifier state;
        # extract only feature_extractor (= EfficientNet backbone) weights
        backbone_state = {
            k.replace("feature_extractor.", ""): v
            for k, v in state["model_state_dict"].items()
            if k.startswith("feature_extractor.")
        }

        missing, unexpected = model.features.load_state_dict(
            backbone_state, strict=False
        )
        if missing:
            print(f"[ABCRegressor] Backbone transfer — missing keys: {len(missing)}")
        if unexpected:
            print(f"[ABCRegressor] Backbone transfer — unexpected keys: {len(unexpected)}")
        print(
            f"[ABCRegressor] Backbone initialised from HAM10000 checkpoint: "
            f"{ham_checkpoint}"
        )
    else:
        print("[ABCRegressor] Using ImageNet pretrained backbone (no HAM10000 checkpoint).")

    model = model.to(device)

    total     = model.get_num_total_params()
    trainable = model.get_num_trainable_params()
    print(
        f"[ABCRegressor] Params — Total: {total:,} | "
        f"Trainable: {trainable:,} | Device: {device}"
    )
    return model
