"""
src/abc/ph2_loader.py
---------------------
PH2 dermoscopic image database loader.

The PH2 database (Mendonça et al., 2013) contains 200 dermoscopic images
of melanocytic lesions with full expert annotations covering:
  - Asymmetry (0 / 1 / 2)
  - Pigment network (absent / typical / atypical)
  - Dots/Globules (absent / regular / irregular)
  - Streaks (absent / regular / irregular)
  - Regression Areas (absent / present)
  - Blue-Whitish Veil (absent / present)
  - Colors (count 1–6)
  - Lesion binary segmentations mask

This module maps PH2 annotations to the unified ABC [0, 1] normalisation
scheme used across PH2, Derm7pt, and the HAM10000 pseudo-scoring pipeline.

ABC Normalisation for PH2
--------------------------
  A (Asymmetry):
    0 → 0.000  (symmetric in both axes)
    1 → 0.500  (asymmetric in one axis)
    2 → 1.000  (asymmetric in both axes)

  B (Border Irregularity):
    Derived as: (streaks_score) / 2
    - absent   → 0.0
    - regular  → 0.5
    - irregular→ 1.0
    Additionally, Dots/Globules irregularity contributes +0.25 (capped at 1.0)
    weighted average to better capture border disruption.

  C (Color Variegation):
    (color_count − 1) / 5   →  maps {1,2,3,4,5,6} to {0.0,0.2,0.4,0.6,0.8,1.0}

References
----------
Mendonça, T., Ferreira, P. M., Marques, J. S., Marcal, A. R., & Rozeira, J. (2013).
    PH2 – A dermoscopic image database for research and benchmarking.
    Proceedings of the 35th Annual International Conference of the IEEE EMBC,
    5437–5440. https://doi.org/10.1109/EMBC.2013.6610779
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.abc.config_abc import (
    PH2_DIR, PH2_METADATA_TXT, PH2_IMAGES_DIR,
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, RANDOM_SEED,
)


# ─────────────────────────────────────────────
# Raw → normalised ABC mappings
# ─────────────────────────────────────────────

def _asymmetry_to_abc(val: int) -> float:
    """Map PH2 asymmetry score {0,1,2} to [0,1]."""
    return float(val) / 2.0


def _border_to_abc(streaks: str, dots: str) -> float:
    """
    Derive a border irregularity score from PH2 streaks and dots/globules.

    Parameters
    ----------
    streaks : str
        'absent' | 'regular' | 'irregular'
    dots : str
        'absent' | 'regular' | 'irregular'

    Returns
    -------
    float in [0, 1]
    """
    streak_map  = {"absent": 0.0, "regular": 0.5, "irregular": 1.0}
    dot_map     = {"absent": 0.0, "regular": 0.25, "irregular": 0.5}
    s = streak_map.get(str(streaks).lower().strip(), 0.0)
    d = dot_map.get(str(dots).lower().strip(), 0.0)
    return float(min(1.0, 0.7 * s + 0.3 * d))


def _color_to_abc(color_count: int) -> float:
    """
    Map PH2 color count {1,2,3,4,5,6} to [0,1].

    1 color  → 0.0 (least variegated)
    6 colors → 1.0 (most variegated)
    """
    count = max(1, min(6, int(color_count)))
    return (count - 1) / 5.0


# ─────────────────────────────────────────────
# Metadata parser
# ─────────────────────────────────────────────

def parse_ph2_metadata(txt_path: Path) -> pd.DataFrame:
    """
    Parse the PH2_dataset.txt annotation file.

    The file uses a fixed-width tabular format with header lines.
    Each row describes one lesion image with clinical and dermoscopic
    feature annotations by expert dermatologists.

    Parameters
    ----------
    txt_path : Path
        Path to PH2_dataset.txt.

    Returns
    -------
    pd.DataFrame
        Columns include:
          image_id, asymmetry, pigment_network, dots_globules,
          streaks, regression, blue_whitish_veil, colors,
          clinical_diagnosis, A_score, B_score, C_score
    """
    rows: List[Dict] = []

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip() for ln in f.readlines()]

    # Skip header / separator lines; data lines contain '|'
    for line in lines:
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        # Filter out separator rows (contain only dashes)
        if any(re.fullmatch(r"-+", p) for p in parts if p):
            continue
        # Expect at least 10 columns
        parts = [p for p in parts]  # keep empties as ''
        if len(parts) < 10:
            continue
        # Skip the column header row
        if parts[1].lower() in ("image name", "image_name", "name"):
            continue

        try:
            image_id  = parts[1].strip()
            asymmetry = int(parts[2].strip() or "0")
            pig_net   = parts[3].strip().lower()
            dots_glob = parts[4].strip().lower()
            streaks   = parts[5].strip().lower()
            regression= parts[6].strip().lower()
            bwv       = parts[7].strip().lower()
            colors    = int(parts[8].strip() or "1")
            dx        = parts[9].strip().lower() if len(parts) > 9 else "unknown"

            A = _asymmetry_to_abc(asymmetry)
            B = _border_to_abc(streaks, dots_glob)
            C = _color_to_abc(colors)

            rows.append({
                "image_id"          : image_id,
                "asymmetry_raw"     : asymmetry,
                "pigment_network"   : pig_net,
                "dots_globules"     : dots_glob,
                "streaks"           : streaks,
                "regression"        : regression,
                "blue_whitish_veil" : bwv,
                "colors_raw"        : colors,
                "clinical_diagnosis": dx,
                "A_score"           : A,
                "B_score"           : B,
                "C_score"           : C,
                "dataset_source"    : "PH2",
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"[PH2Loader] No valid rows parsed from {txt_path}. "
            "Check file format / encoding."
        )

    print(
        f"[PH2Loader] Parsed {len(df)} images  "
        f"| A ∈ [{df.A_score.min():.2f}, {df.A_score.max():.2f}]  "
        f"| B ∈ [{df.B_score.min():.2f}, {df.B_score.max():.2f}]  "
        f"| C ∈ [{df.C_score.min():.2f}, {df.C_score.max():.2f}]"
    )
    return df


# ─────────────────────────────────────────────
# Image / mask path helpers
# ─────────────────────────────────────────────

def _find_image(image_id: str, images_dir: Path) -> Optional[Path]:
    """
    Return path to the dermoscopic image for given image_id.

    Supports two PH2 directory layouts:
      1. Flat layout (Kaggle mirror):
           PH2/trainx/IMD002.bmp
      2. Original nested layout:
           PH2/PH2 Images/IMD002/IMD002_Dermoscopic_Image/IMD002.bmp
    """
    # Layout 1: flat trainx/
    for ext in ("bmp", "jpg", "jpeg", "png"):
        p = images_dir / f"{image_id}.{ext}"
        if p.exists():
            return p
    # Layout 2: original nested structure
    sub = images_dir.parent / "PH2 Images" / image_id / f"{image_id}_Dermoscopic_Image"
    for ext in ("bmp", "jpg", "jpeg", "png"):
        p = sub / f"{image_id}.{ext}"
        if p.exists():
            return p
    return None


def _find_mask(image_id: str, images_dir: Path, masks_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Return path to the binary segmentations mask for given image_id.

    Supports two PH2 directory layouts:
      1. Flat layout (Kaggle mirror):
           PH2/trainy/IMD002.bmp  or  PH2/trainy/IMD002_lesion.bmp
      2. Original nested layout:
           PH2/PH2 Images/IMD002/IMD002_lesion/IMD002_lesion.bmp
    """
    # Layout 1: flat trainy/ folder
    if masks_dir is None:
        masks_dir = images_dir.parent / "trainy"
    for suffix in ("", "_lesion"):
        for ext in ("bmp", "png", "jpg"):
            p = masks_dir / f"{image_id}{suffix}.{ext}"
            if p.exists():
                return p
    # Layout 2: original nested structure
    sub = images_dir.parent / "PH2 Images" / image_id / f"{image_id}_lesion"
    for ext in ("bmp", "png", "jpg"):
        p = sub / f"{image_id}_lesion.{ext}"
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class PH2Dataset(Dataset):
    """
    PyTorch Dataset for the PH2 dermoscopic image database.

    Each sample returns:
      - image tensor (3, H, W)
      - mask tensor  (1, H, W) binary
      - ABC label    (3,) float32 in [0, 1]
      - metadata dict

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Parsed PH2 metadata (output of parse_ph2_metadata).
    images_dir : Path
        Root directory containing PH2 image sub-folders.
    augment : bool
        Apply training-time augmentations.
    image_size : int
        Output image size (default: IMAGE_SIZE).
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        images_dir: Path,
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
    ):
        self.df        = metadata_df.reset_index(drop=True)
        self.img_dir   = images_dir
        self.augment   = augment
        self.img_size  = image_size

        # Build transforms
        if augment:
            self.img_tf = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ])
        else:
            self.img_tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ])

        self.mask_tf = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])

        # Filter rows with missing images
        # Determine masks directory (trainy/ sibling of trainx/)
        self.msk_dir = self.img_dir.parent / "trainy"
        if not self.msk_dir.exists():
            self.msk_dir = None

        valid_mask = []
        for _, row in self.df.iterrows():
            p = _find_image(row.image_id, self.img_dir)
            valid_mask.append(p is not None)

        n_before = len(self.df)
        self.df = self.df[valid_mask].reset_index(drop=True)
        if len(self.df) < n_before:
            print(
                f"[PH2Dataset] Dropped {n_before - len(self.df)} rows "
                "with missing images."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        row      = self.df.iloc[idx]
        img_path = _find_image(row.image_id, self.img_dir)
        msk_path = _find_mask(row.image_id, self.img_dir, self.msk_dir)

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image_t = self.img_tf(image)

        # Load or create empty mask
        if msk_path is not None:
            mask = Image.open(msk_path).convert("L")
            mask_t = self.mask_tf(mask)
            mask_t = (mask_t > 0.5).float()
        else:
            mask_t = torch.zeros(1, self.img_size, self.img_size)

        # ABC labels
        abc = torch.tensor(
            [row.A_score, row.B_score, row.C_score],
            dtype=torch.float32,
        )

        meta = {
            "image_id"          : row.image_id,
            "clinical_diagnosis": row.clinical_diagnosis,
            "dataset_source"    : "PH2",
        }

        return image_t, mask_t, abc, meta


# ─────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────

def load_ph2(
    ph2_dir: Path = PH2_DIR,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[PH2Dataset, PH2Dataset, PH2Dataset]:
    """
    Load PH2 dataset and return train / val / test splits.

    Stratified split by clinical diagnosis to preserve class ratios.

    Parameters
    ----------
    ph2_dir : Path
        Root PH2 directory.
    val_split : float
        Fraction for validation set.
    test_split : float
        Fraction for test set.
    seed : int
        Random seed.

    Returns
    -------
    Tuple[PH2Dataset, PH2Dataset, PH2Dataset]
        (train_dataset, val_dataset, test_dataset)
    """
    from sklearn.model_selection import train_test_split

    meta_txt   = ph2_dir / "PH2_dataset.txt"
    images_dir = ph2_dir / "trainx"   # Kaggle mirror layout: PH2/trainx/
    # Fallback for original nested layout
    if not images_dir.exists():
        images_dir = ph2_dir / "PH2 Images"

    df = parse_ph2_metadata(meta_txt)

    # Stratified split
    train_df, tmp_df = train_test_split(
        df, test_size=val_split + test_split,
        stratify=df["clinical_diagnosis"], random_state=seed,
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=test_split / (val_split + test_split),
        stratify=tmp_df["clinical_diagnosis"], random_state=seed,
    )

    train_ds = PH2Dataset(train_df, images_dir, augment=True)
    val_ds   = PH2Dataset(val_df,   images_dir, augment=False)
    test_ds  = PH2Dataset(test_df,  images_dir, augment=False)

    print(
        f"[PH2Loader] Split — Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return train_ds, val_ds, test_ds
