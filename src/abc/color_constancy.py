"""
src/abc/color_constancy.py
--------------------------
Color constancy preprocessing for dermoscopic images.

Dermoscopes from different manufacturers produce images with different
white balances. Without normalization, a model trained on Derm7pt
(SFU/SFU-standardized images) will suffer domain shift on HAM10000
(multi-source, Vienna + Queensland).

Shades of Gray (SoG) Algorithm
--------------------------------
Buchsbaum (1980) / Finlayson & Trezzi (2004).
Estimates the scene illuminant as the p-norm of each colour channel,
then corrects the image by dividing by the estimated illuminant.

  Î_c = (1/N Σ I_c^p)^(1/p)
  I'_c = I_c / Î_c * mean(Î)

p=1: Grey World assumption
p=6: Shades of Gray (best empirical performance on dermoscopy,
     Barata et al. IEEE J-BHI 2014)
p=∞: Max-RGB

Reference:
    Barata, C., et al. (2014). Improving dermoscopy image classification
    using color constancy. IEEE J-BHI, 19(3), 1146-1152.
    https://doi.org/10.1109/JBHI.2014.2336473
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Union


class ShadesOfGray:
    """
    Shades of Gray color constancy preprocessing.

    Implements the p-norm illuminant estimation from Finlayson & Trezzi
    (2004), empirically validated for dermoscopic images by Barata et al.
    (2014) who found p=6 optimal.

    Can be used as:
      - A standalone callable applied to PIL Image or numpy array
      - A torchvision transform (pass instance to transforms.Compose)

    Parameters
    ----------
    p : int or float
        Norm order. Default 6 (recommended for dermoscopy).
        p=1 → Grey World, p=np.inf → Max-RGB.
    eps : float
        Small constant to avoid division by zero.
    """

    def __init__(self, p: float = 6.0, eps: float = 1e-6):
        self.p   = p
        self.eps = eps

    def apply_numpy(self, img: np.ndarray) -> np.ndarray:
        """
        Apply SoG to a uint8 numpy array (H, W, 3).

        Returns
        -------
        np.ndarray  uint8 (H, W, 3)
        """
        img_f = img.astype(np.float32) / 255.0

        if np.isinf(self.p):
            # Max-RGB
            illuminant = img_f.max(axis=(0, 1)) + self.eps
        else:
            # p-norm per channel
            illuminant = (
                np.mean(img_f ** self.p, axis=(0, 1)) + self.eps
            ) ** (1.0 / self.p)

        # Scale to preserve average brightness
        scale = illuminant.mean() / (illuminant + self.eps)
        corrected = img_f * scale[np.newaxis, np.newaxis, :]
        corrected = np.clip(corrected, 0.0, 1.0)
        return (corrected * 255).astype(np.uint8)

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        Apply SoG — auto-detects input type and returns same type.
        """
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
            return Image.fromarray(self.apply_numpy(arr))
        elif isinstance(img, np.ndarray):
            return self.apply_numpy(img)
        elif isinstance(img, torch.Tensor):
            # Tensor (3, H, W) float in [0, 1]
            arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            corrected = self.apply_numpy(arr)
            return torch.from_numpy(corrected).permute(2, 0, 1).float() / 255.0
        else:
            raise TypeError(f"Unsupported type: {type(img)}")


class HairAugmentation:
    """
    Synthetic hair artifact augmentation for dermoscopic images.

    Adds Bézier-curve hair strands with realistic appearance to
    simulate a common dermoscopy artifact that affects lesion feature
    extraction. Forces the model to learn ABC features robust to
    hair occlusion.

    Reference:
        Jütte, L., et al. (2024). Advancing dermoscopy through a
        synthetic hair benchmark dataset and deep learning-based hair
        removal. PubMed 39564076.

    Parameters
    ----------
    p : float
        Probability of applying hair augmentation. Default 0.5.
    n_hairs : tuple
        Range of hair count to draw. Default (1, 6).
    thickness_range : tuple
        Min/max hair thickness in pixels. Default (1, 3).
    """

    def __init__(
        self,
        p: float = 0.5,
        n_hairs: tuple = (1, 6),
        thickness_range: tuple = (1, 3),
    ):
        self.p               = p
        self.n_hairs         = n_hairs
        self.thickness_range = thickness_range

    def _draw_hair(self, img: np.ndarray) -> np.ndarray:
        """Draw synthetic hair strands on image array (H,W,3)."""
        import cv2  # lazy import — avoids pickle error in multiprocessing
        result = img.copy()
        H, W   = img.shape[:2]
        n      = np.random.randint(*self.n_hairs)

        for _ in range(n):
            # Bézier-like curve: 3 random control points
            pts = np.array([
                [np.random.randint(0, W), np.random.randint(0, H)],
                [np.random.randint(0, W), np.random.randint(0, H)],
                [np.random.randint(0, W), np.random.randint(0, H)],
            ], dtype=np.float32)

            # Sample points along quadratic Bézier
            t     = np.linspace(0, 1, 60)
            curve = np.outer((1-t)**2, pts[0]) + \
                    np.outer(2*(1-t)*t, pts[1]) + \
                    np.outer(t**2, pts[2])
            curve = curve.astype(np.int32)

            # Hair colour: dark brown to black
            color = (
                np.random.randint(0, 50),
                np.random.randint(0, 40),
                np.random.randint(0, 30),
            )
            thick = np.random.randint(*self.thickness_range)

            for i in range(len(curve) - 1):
                cv2.line(result, tuple(curve[i]), tuple(curve[i+1]),
                         color, thick, cv2.LINE_AA)

        return result

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img
        arr    = np.array(img.convert("RGB"))
        result = self._draw_hair(arr)
        return Image.fromarray(result)


def build_dermoscopy_transforms(
    image_size: int,
    image_mean: tuple,
    image_std: tuple,
    augment: bool = False,
    color_constancy: bool = True,
) -> transforms.Compose:
    """
    Build dermoscopy-specific transform pipeline.

    Augmentation strategy based on:
    - Perez et al. (2018): augmentation > new data for small sets
    - Barata et al. (2014): color constancy preprocessing
    - Jütte et al. (2024): synthetic hair artifacts

    Parameters
    ----------
    image_size : int
    image_mean, image_std : tuple  (HAM10000 statistics)
    augment : bool
    color_constancy : bool
        Apply Shades of Gray normalization (recommended).

    Returns
    -------
    transforms.Compose
    """
    sog  = ShadesOfGray(p=6.0)
    hair = HairAugmentation(p=0.4)

    if augment:
        tf_list = []

        # 1. Color constancy (before any spatial transforms)
        if color_constancy:
            tf_list.append(sog)

        # 2. Synthetic hair artifacts
        tf_list.append(hair)

        # 3. Geometric augmentations
        tf_list += [
            transforms.Resize((image_size + 40, image_size + 40)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),  # dermoscopy: any angle
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)
            ),
        ]

        # 4. Colour augmentation (constrained to preserve diagnostic colours)
        tf_list += [
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.15,
                hue=0.03,          # small hue shift: preserve dermoscopic colors
            ),
            transforms.RandomGrayscale(p=0.03),
        ]

        # 5. Tensor + normalize
        tf_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # occlusion
        ]

    else:
        tf_list = []
        if color_constancy:
            tf_list.append(sog)
        tf_list += [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]

    return transforms.Compose(tf_list)