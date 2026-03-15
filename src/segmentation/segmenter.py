"""
src/segmentations/segmenter.py
------------------------------
Lesion segmentations for dermoscopic images.

Two segmentations strategies are implemented:

1. **U-Net (Deep Learning)**
   A lightweight encoder-decoder network trained on ISIC 2018 Task 1
   ground-truth masks.  When pre-trained weights are available, this
   produces binary segmentations masks at IMAGE_SIZE × IMAGE_SIZE.

2. **Otsu Thresholding (Fallback)**
   A classical image-processing segmentations using multi-level Otsu
   thresholding on the green channel (highest contrast for skin lesions).
   Used when no mask file is available and no pre-trained U-Net weights
   are provided.

The segmenter is used in the ABC pipeline to:
  - Extract lesion shape features for A (Asymmetry) IP scoring
  - Extract lesion boundary for B (Border) IP scoring
  - Restrict color analysis to the lesion region for C (Color) scoring

References
----------
Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-Net: Convolutional networks for biomedical image segmentations.
    MICCAI 2015. https://doi.org/10.1007/978-3-319-24574-4_28

Codella, N. C. F., et al. (2018).
    Skin lesion analysis toward melanoma detection: ISIC 2018 challenge.
    arXiv:1902.03368.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────
# Lightweight U-Net for lesion segmentations
# ─────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Two consecutive BN → Conv → ReLU layers."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class LesionUNet(nn.Module):
    """
    Lightweight U-Net for binary lesion segmentations.

    Encoder: 4 downsampling stages (32→64→128→256 channels)
    Decoder: symmetric upsampling with skip connections
    Output:  1-channel sigmoid probability map

    Parameters
    ----------
    in_channels : int
        Number of input image channels (3 for RGB).
    base_filters : int
        Number of filters in the first encoder block (default: 32).
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = _ConvBlock(in_channels, f)
        self.enc2 = _ConvBlock(f,  f * 2)
        self.enc3 = _ConvBlock(f * 2, f * 4)
        self.enc4 = _ConvBlock(f * 4, f * 8)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.2)

        # Bottleneck
        self.bottleneck = _ConvBlock(f * 8, f * 16)

        # Decoder
        self.up4   = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4  = _ConvBlock(f * 16, f * 8)
        self.up3   = nn.ConvTranspose2d(f * 8,  f * 4, 2, stride=2)
        self.dec3  = _ConvBlock(f * 8,  f * 4)
        self.up2   = nn.ConvTranspose2d(f * 4,  f * 2, 2, stride=2)
        self.dec2  = _ConvBlock(f * 4,  f * 2)
        self.up1   = nn.ConvTranspose2d(f * 2,  f,     2, stride=2)
        self.dec1  = _ConvBlock(f * 2,  f)

        self.out_conv = nn.Conv2d(f, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, 3, H, W)

        Returns
        -------
        torch.Tensor  (B, 1, H, W) — sigmoid probabilities
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.drop(self.pool(e3)))

        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


# ─────────────────────────────────────────────
# Classical fallback: Otsu thresholding
# ─────────────────────────────────────────────

def otsu_segmentation(image_np: np.ndarray) -> np.ndarray:
    """
    Segment a dermoscopic lesion using multi-level Otsu thresholding
    on the green channel.

    The green channel is chosen because it provides the highest contrast
    between skin and lesion in standard dermoscopy images.

    Post-processing:
      - Fill holes (morphological closing)
      - Keep only the largest connected component
      - Remove small artefacts (< SEGMENTATION_MIN_AREA pixels)

    Parameters
    ----------
    image_np : np.ndarray
        RGB image array (H, W, 3) in [0, 255] uint8.

    Returns
    -------
    np.ndarray
        Binary mask (H, W) bool.
    """
    # Green channel
    green = image_np[:, :, 1]

    # Normalise and apply Otsu
    blur  = cv2.GaussianBlur(green, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Keep largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        areas    = stats[1:, cv2.CC_STAT_AREA]
        best_lbl = int(np.argmax(areas)) + 1
        mask     = (labels == best_lbl).astype(np.uint8) * 255

    return mask.astype(bool)


# ─────────────────────────────────────────────
# Unified Segmenter class
# ─────────────────────────────────────────────

class LesionSegmenter:
    """
    Unified lesion segmenter supporting both deep learning (U-Net)
    and classical (Otsu) approaches.

    Parameters
    ----------
    model_weights : Path or None
        Path to saved U-Net weights (.pth).  If None or file missing,
        falls back to Otsu thresholding.
    device : torch.device
        Computation device.
    image_size : int
        Input/output spatial resolution.
    threshold : float
        Binary threshold on U-Net sigmoid output.
    """

    def __init__(
        self,
        model_weights: Optional[Path] = None,
        device: Optional[torch.device] = None,
        image_size: int = 224,
        threshold: float = 0.5,
    ):
        self.image_size = image_size
        self.threshold  = threshold
        self.device     = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_dl = False

        if model_weights is not None and Path(model_weights).exists():
            self.model = LesionUNet().to(self.device)
            state = torch.load(model_weights, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.use_dl = True
            print(f"[Segmenter] U-Net loaded from {model_weights}")
        else:
            print("[Segmenter] U-Net weights not found — using Otsu fallback")

    @torch.no_grad()
    def segment(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
    ) -> np.ndarray:
        """
        Produce a binary segmentations mask for a dermoscopic image.

        Parameters
        ----------
        image : np.ndarray (H,W,3) | torch.Tensor (3,H,W) | PIL Image

        Returns
        -------
        np.ndarray
            Binary mask (image_size, image_size) bool.
        """
        # Normalise input to numpy uint8 (H,W,3)
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB").resize(
                (self.image_size, self.image_size)
            ))
        elif isinstance(image, torch.Tensor):
            t = image.cpu()
            if t.max() <= 1.0:
                t = (t * 255).byte()
            img_np = t.permute(1, 2, 0).numpy().astype(np.uint8)
            if img_np.shape[:2] != (self.image_size, self.image_size):
                img_np = cv2.resize(img_np, (self.image_size, self.image_size))
        else:
            img_np = np.array(image, dtype=np.uint8)
            if img_np.shape[:2] != (self.image_size, self.image_size):
                img_np = cv2.resize(img_np, (self.image_size, self.image_size))

        if self.use_dl:
            return self._dl_segment(img_np)
        else:
            return otsu_segmentation(img_np)

    def _dl_segment(self, img_np: np.ndarray) -> np.ndarray:
        """Run U-Net inference."""
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.7630392, 0.5456477, 0.5700950),
                std =(0.1409286, 0.1526128, 0.1694007),
            ),
        ])
        inp  = tf(Image.fromarray(img_np)).unsqueeze(0).to(self.device)
        prob = self.model(inp).squeeze().cpu().numpy()
        return prob > self.threshold

    def segment_batch(
        self, images: torch.Tensor
    ) -> np.ndarray:
        """
        Segment a batch of images.

        Parameters
        ----------
        images : torch.Tensor  (B, 3, H, W) — normalised tensors

        Returns
        -------
        np.ndarray  (B, H, W) bool
        """
        masks = []
        for img_t in images:
            masks.append(self.segment(img_t))
        return np.stack(masks, axis=0)
