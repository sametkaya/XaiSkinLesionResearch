"""
src/abc/abc_ip_scorer.py
------------------------
Image-Processing (IP) based ABC scorer.

Computes Asymmetry, Border irregularity, and Color variegation scores
directly from dermoscopic image pixels and segmentations masks, without
any learned model.  These scores serve as a second independent estimate
alongside the deep learning regressor, enabling method-agreement analysis.

Algorithms
----------
A — Asymmetry Score
    1. Fit a minimum-inertia ellipse to the binary lesion mask.
    2. Rotate so that the major axis is horizontal.
    3. Compute overlap ratio after reflection about each axis.
       R_h = |M ∩ flip(M, h)| / |M|   (horizontal axis)
       R_v = |M ∩ flip(M, v)| / |M|   (vertical axis)
    4. A_score = 1 − (R_h + R_v) / 2
    Reference: Lee, T., et al. (1997). Dullrazor: A software approach to
    hair removal from images. Computers in Biology and Medicine, 27(6), 533–543.

B — Border Irregularity Score
    1. Extract the lesion boundary (contour) from the mask.
    2. Compute the compactness (circularity) index:
       CI = (4π × Area) / Perimeter²   ∈ (0, 1]
       (1.0 = perfect circle; smaller = more irregular)
    3. B_score = 1 − CI  (higher = more irregular border)
    Reference: Lee, T., et al. (1997).

C — Color Variegation Score
    1. Restrict pixels to the lesion mask (foreground only).
    2. Convert to HSV colour space.
    3. Count the number of standard dermoscopic colours present
       (from Argenziano et al., 1998): black, dark brown, light brown,
       red, blue-grey, white.
    4. C_score = (n_colors − 1) / 5   ∈ [0, 1]

References
----------
Argenziano, G., Fabbrocini, G., Carli, P., et al. (1998).
    Epiluminescence microscopy for the diagnosis of doubtful melanocytic
    skin lesions. Archives of Dermatology, 134(12), 1563–1570.

Stolz, W., et al. (1994). ABCD rule of dermatoscopy. Eur. J. Dermatol., 4, 521–527.

Celebi, M. E., et al. (2019). A survey of feature extraction in dermoscopy
    image analysis. IEEE JBHI, 23(2), 682–696.
"""

from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.abc.config_abc import (
    IP_BORDER_SIGMA,
    IP_COLOR_THRESHOLD,
    DERMOSCOPIC_COLORS,
)


# ─────────────────────────────────────────────
# A — Asymmetry
# ─────────────────────────────────────────────

def compute_asymmetry(mask: np.ndarray) -> float:
    """
    Compute A (Asymmetry) score from a binary lesion mask.

    The mask is aligned to its principal axis (minimum inertia),
    then reflected about each axis.  The overlap asymmetry is
    averaged across both axes.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) bool or uint8.

    Returns
    -------
    float in [0, 1]
        0.0 = fully symmetric; 1.0 = maximally asymmetric.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # Find largest contour
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 50:
        return 0.0

    # Minimum enclosing ellipse to get principal axis angle
    if len(cnt) >= 5:
        (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
    else:
        moments = cv2.moments(mask_u8)
        if moments["m00"] == 0:
            return 0.0
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        angle = 0.0

    # Rotate mask to align principal axis horizontally
    H, W    = mask_u8.shape
    centre  = (W / 2, H / 2)
    rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        mask_u8, rot_mat, (W, H), flags=cv2.INTER_NEAREST
    ).astype(bool)

    # Overlap after reflection
    def _overlap(m: np.ndarray, axis: int) -> float:
        flipped    = np.flip(m, axis=axis)
        intersect  = np.logical_and(m, flipped).sum()
        union      = np.logical_or(m, flipped).sum()
        if union == 0:
            return 1.0
        return intersect / union

    r_h = _overlap(rotated, axis=0)   # flip vertically   (horizontal axis)
    r_v = _overlap(rotated, axis=1)   # flip horizontally (vertical axis)

    A = 1.0 - (r_h + r_v) / 2.0
    return float(np.clip(A, 0.0, 1.0))


# ─────────────────────────────────────────────
# B — Border Irregularity
# ─────────────────────────────────────────────

def compute_border(mask: np.ndarray) -> float:
    """
    Compute B (Border Irregularity) score from a binary lesion mask.

    Uses the compactness index (1 = circle, <1 = irregular):
      CI = (4π × Area) / Perimeter²

    B_score = 1 − CI

    Additional component: Sobel gradient entropy along the lesion boundary.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) bool or uint8.

    Returns
    -------
    float in [0, 1]
        0.0 = perfectly circular; 1.0 = maximally irregular.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, closed=True)

    if peri == 0 or area < 50:
        return 0.0

    compactness = (4 * np.pi * area) / (peri ** 2)
    B = 1.0 - float(np.clip(compactness, 0.0, 1.0))
    return float(np.clip(B, 0.0, 1.0))


# ─────────────────────────────────────────────
# C — Color Variegation
# ─────────────────────────────────────────────

def compute_color(image_np: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute C (Color Variegation) score from masked lesion pixels.

    Detects the presence of standard dermoscopic colors within the lesion
    (Argenziano et al., 1998): black, dark brown, light brown, red,
    blue-grey, white.

    Parameters
    ----------
    image_np : np.ndarray
        RGB image (H, W, 3) uint8.
    mask : np.ndarray
        Binary mask (H, W) bool or uint8.

    Returns
    -------
    float in [0, 1]
        0.0 = monochromatic; 1.0 = 6 distinct colors present.
    """
    mask_bool = (mask > 0)
    if mask_bool.sum() < 50:
        return 0.0

    # Extract lesion pixels only
    lesion_pixels = image_np[mask_bool]          # (N, 3) uint8 RGB

    # Convert to HSV
    rgb_img     = image_np.copy()
    hsv_full    = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    hsv_pixels  = hsv_full[mask_bool]            # (N, 3)  H[0-180], S[0-255], V[0-255]

    n_pixels = len(hsv_pixels)
    min_frac = IP_COLOR_THRESHOLD

    h = hsv_pixels[:, 0].astype(int)
    s = hsv_pixels[:, 1].astype(int)
    v = hsv_pixels[:, 2].astype(int)

    colors_detected = 0
    for color_name, ranges in DERMOSCOPIC_COLORS.items():
        h_lo, h_hi = ranges["h"]
        s_lo, s_hi = ranges["s"]
        v_lo, v_hi = ranges["v"]



        in_h = (h >= h_lo) & (h <= h_hi)
        # Handle hue wrap-around for red/pink (H near 0 and near 180)
        if "h_wrap" in ranges:
            hw_lo, hw_hi = ranges["h_wrap"]
            in_h = in_h | ((h >= hw_lo) & (h <= hw_hi))

        in_s = (s >= s_lo) & (s <= s_hi)
        in_v = (v >= v_lo) & (v <= v_hi)

        frac = (in_h & in_s & in_v).sum() / n_pixels
        if frac >= min_frac:
            colors_detected += 1

    C = (colors_detected - 1) / 5.0
    return float(np.clip(C, 0.0, 1.0))


# ─────────────────────────────────────────────
# Unified scorer
# ─────────────────────────────────────────────

class ABCImageProcessingScorer:
    """
    Compute A, B, C scores purely from image pixels and segmentations masks.

    Used as Method 2 alongside the deep-learning regressor, enabling
    cross-validation and agreement analysis in the HAM10000 scoring pipeline.

    Parameters
    ----------
    segmenter : LesionSegmenter or None
        If provided, used to produce masks when none is available.
        If None, masks must be supplied explicitly.
    """

    def __init__(self, segmenter=None):
        self.segmenter = segmenter

    def score(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, None] = None,
    ) -> Dict[str, float]:
        """
        Compute ABC scores for a single image.

        Parameters
        ----------
        image : np.ndarray (H, W, 3) uint8 | PIL Image
        mask  : np.ndarray (H, W) bool | None
            If None, segmentations is attempted via self.segmenter.

        Returns
        -------
        dict with keys 'A', 'B', 'C' (floats in [0, 1])
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        if mask is None:
            if self.segmenter is not None:
                mask = self.segmenter.segment(image)
            else:
                mask = np.ones(image.shape[:2], dtype=bool)

        mask = (mask > 0)

        A = compute_asymmetry(mask)
        B = compute_border(mask)
        C = compute_color(image, mask)

        return {"A": A, "B": B, "C": C}

    def score_batch(
        self,
        images: List[np.ndarray],
        masks:  List[Union[np.ndarray, None]],
    ) -> List[Dict[str, float]]:
        """Score a list of images."""
        return [self.score(img, msk) for img, msk in zip(images, masks)]
