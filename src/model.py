"""
model.py
--------
CNN classifier for HAM10000 skin lesion classification.

Supported backbones:
  - ResNet-50    (He et al., 2016)
  - EfficientNet-B0 (Tan & Le, 2019)

Both are initialised with ImageNet pre-trained weights and fine-tuned
end-to-end on HAM10000.  The final classification head is replaced with a
linear layer matching NUM_CLASSES = 7.

References:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
    image recognition. CVPR 2016.

    Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for
    convolutional neural networks. ICML 2019.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

from src import config


class SkinLesionClassifier(nn.Module):
    """
    Transfer-learning classifier for skin lesion diagnosis.

    The backbone is pre-trained on ImageNet; its final fully-connected (or
    classifier) layer is replaced with a new linear layer that outputs
    logits for each of the 7 HAM10000 diagnostic categories.

    Parameters
    ----------
    backbone : str
        Model architecture.  Must be one of 'resnet50' or 'efficientnet_b0'.
    num_classes : int
        Number of output classes (default: 7).
    pretrained : bool
        If True, load ImageNet pre-trained weights.
    freeze_backbone : bool
        If True, freeze all backbone parameters and only train the new head.
        This is typically used for fast feature extraction experiments only.
    dropout_rate : float
        Dropout probability applied before the classification head to
        regularise training.
    """

    def __init__(
        self,
        backbone: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.PRETRAINED,
        freeze_backbone: bool = config.FREEZE_BACKBONE,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        weights_arg = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet50":
            base_model = models.resnet50(weights=weights_arg)
            in_features = base_model.fc.in_features          # 2048
            base_model.fc = nn.Identity()                    # Remove original head
            self.feature_extractor = base_model
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )

        elif backbone == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=weights_arg)
            in_features = base_model.classifier[1].in_features  # 1280
            base_model.classifier = nn.Identity()
            self.feature_extractor = base_model
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1) if False else nn.Identity(),  # already pooled
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.GELU(),
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(512, num_classes),
            )

        elif backbone == "efficientnet_b4":
            # EfficientNet-B4: best accuracy/efficiency tradeoff on HAM10000
            # Top-1 accuracy 87.91%, F1 87% (Ali et al., ResearchGate 2023)
            # Optimal input size: 380×380
            base_model = models.efficientnet_b4(weights=weights_arg)
            in_features = base_model.classifier[1].in_features  # 1792
            base_model.classifier = nn.Identity()
            self.feature_extractor = base_model
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.GELU(),
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(512, num_classes),
            )

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Choose from: 'resnet50', 'efficientnet_b0', 'efficientnet_b4'."
            )

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images with shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Logits with shape (B, num_classes).
        """
        features = self.feature_extractor(x)
        logits   = self.classifier(features)
        return logits

    def get_feature_layer(self) -> nn.Module:
        """
        Return the last convolutional block used by Grad-CAM.

        Returns
        -------
        nn.Module
            The target layer for gradient-weighted class activation mapping.
        """
        if self.backbone_name == "resnet50":
            return self.feature_extractor.layer4[-1]
        elif self.backbone_name in ("efficientnet_b0", "efficientnet_b4"):
            return self.feature_extractor.features[-1]
        else:
            raise ValueError(f"No target layer defined for '{self.backbone_name}'.")

    def get_num_trainable_params(self) -> int:
        """Return the count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """Return the total parameter count."""
        return sum(p.numel() for p in self.parameters())


def build_model(device: Optional[torch.device] = None) -> SkinLesionClassifier:
    """
    Convenience factory: instantiate and move model to device.

    Parameters
    ----------
    device : torch.device, optional
        Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    SkinLesionClassifier
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkinLesionClassifier()
    model = model.to(device)

    total     = model.get_num_total_params()
    trainable = model.get_num_trainable_params()
    print(
        f"[Model] {config.MODEL_NAME.upper()} — "
        f"Total params: {total:,} | Trainable: {trainable:,} | Device: {device}"
    )
    return model