"""
src/abc/derm7pt_loader.py
-------------------------
Derm7pt (Seven-Point Checklist) dataset loader.

The Derm7pt dataset (Kawahara et al., 2019) contains 2,000 dermoscopic images
annotated with the Seven-Point Checklist (Argenziano et al., 1998).

Real meta.csv column names (from official release):
  diagnosis, pigment_network, blue_whitish_veil, vascular_structures,
  pigmentation, streaks, dots_globules, regression_structures,
  derm (path to dermoscopy image), clinic (path to clinical image)

Label values per criterion (string):
  pigment_network:      'absent' | 'typical' | 'atypical'
  blue_whitish_veil:    'absent' | 'present'
  vascular_structures:  'absent' | 'regular' | 'dotted/irregular' | 'within regression' | ...
  pigmentation:         'absent' | 'diffuse regular' | 'localized regular' |
                        'diffuse irregular' | 'localized irregular'
  streaks:              'absent' | 'regular' | 'irregular'
  dots_globules:        'absent' | 'regular' | 'irregular'
  regression_structures:'absent' | 'present'

ABC Normalisation for Derm7pt
------------------------------
  A (Asymmetry) — derived from pigmentation irregularity + regression:
    absent           → 0.0
    regular/typical  → 0.25
    irregular/atypical → 0.75
    present          → 0.5

  B (Border Irregularity) — from streaks + dots_globules:
    B = 0.6 × norm(streaks) + 0.4 × norm(dots_globules)

  C (Color Variegation) — from pigment_network + blue_whitish_veil + pigmentation:
    C = 0.4 × norm(pig_net) + 0.4 × norm(bwv) + 0.2 × norm(pigmentation)

References
----------
Kawahara, J., Daneshvar, S., Argenziano, G., & Hamarneh, G. (2019).
    Seven-point checklist and skin lesion classification using
    multitask multimodal neural nets. IEEE JBHI, 23(2), 538–546.

Argenziano, G., et al. (1998). Epiluminescence microscopy for the diagnosis
    of doubtful melanocytic skin lesions. Arch. Dermatol., 134(12), 1563–1570.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.abc.color_constancy import build_dermoscopy_transforms
from src.abc.config_abc import (
    DERM7PT_DIR, DERM7PT_META_CSV,
    DERM7PT_TRAIN_CSV, DERM7PT_VAL_CSV, DERM7PT_TEST_CSV,
    DERM7PT_IMG_DIR,
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, RANDOM_SEED,
)


# ─────────────────────────────────────────────
# Normalisation helpers (real Derm7pt label values)
# ─────────────────────────────────────────────

def _norm_pignet(val) -> float:
    """pigment_network: absent→0, typical→0.3, atypical→1.0"""
    mapping = {
        "absent": 0.0,
        "typical": 0.3,
        "atypical": 1.0,
    }
    v = str(val).lower().strip()
    for k, score in mapping.items():
        if k in v:
            return score
    return 0.0


def _norm_bwv(val) -> float:
    """blue_whitish_veil: absent→0, present→1"""
    v = str(val).lower().strip()
    return 1.0 if "present" in v else 0.0


def _norm_vascular(val) -> float:
    """vascular_structures: absent→0, regular→0.3, irregular→0.7, within regression→0.5"""
    v = str(val).lower().strip()
    if "absent" in v:
        return 0.0
    if "irregular" in v or "dotted" in v:
        return 0.7
    if "regular" in v:
        return 0.3
    if "regression" in v:
        return 0.5
    return 0.0


def _norm_pigmentation(val) -> float:
    """pigmentation: absent→0, regular→0.3, irregular→0.8"""
    v = str(val).lower().strip()
    if "absent" in v:
        return 0.0
    if "irregular" in v:
        return 0.8
    if "regular" in v:
        return 0.3
    return 0.0


def _norm_streaks(val) -> float:
    """streaks: absent→0, regular→0.4, irregular→1.0"""
    v = str(val).lower().strip()
    if "absent" in v:
        return 0.0
    if "irregular" in v:
        return 1.0
    if "regular" in v:
        return 0.4
    return 0.0


def _norm_dots(val) -> float:
    """dots_globules: absent→0, regular→0.4, irregular→1.0"""
    v = str(val).lower().strip()
    if "absent" in v:
        return 0.0
    if "irregular" in v:
        return 1.0
    if "regular" in v:
        return 0.4
    return 0.0


def _norm_regression(val) -> float:
    """regression_structures: absent→0, present→1"""
    v = str(val).lower().strip()
    return 1.0 if "present" in v else 0.0


# ─────────────────────────────────────────────
# Row-level ABC computation
# ─────────────────────────────────────────────

def _compute_abc(row: pd.Series) -> Tuple[float, float, float]:
    """
    Compute A, B, C scores from a single Derm7pt metadata row.

    Returns
    -------
    (A, B, C) each in [0, 1]
    """
    def get(col):
        for c in row.index:
            if col in c.lower():
                v = row[c]
                if pd.notna(v):
                    return v
        return "absent"

    pig_net    = get("pigment_network")
    bwv        = get("blue_whitish_veil")
    pigment    = get("pigmentation")
    streaks    = get("streaks")
    dots       = get("dots_globules")
    regression = get("regression")

    # A: Asymmetry ← pigmentation irregularity + regression presence
    A = 0.6 * _norm_pigmentation(pigment) + 0.4 * _norm_regression(regression)

    # B: Border irregularity ← streaks + dots/globules
    B = 0.6 * _norm_streaks(streaks) + 0.4 * _norm_dots(dots)

    # C: Color variegation ← pigment network + blue-whitish veil + pigmentation
    C = 0.4 * _norm_pignet(pig_net) + 0.4 * _norm_bwv(bwv) + 0.2 * _norm_pigmentation(pigment)

    return (
        float(np.clip(A, 0.0, 1.0)),
        float(np.clip(B, 0.0, 1.0)),
        float(np.clip(C, 0.0, 1.0)),
    )


# ─────────────────────────────────────────────
# Metadata loader
# ─────────────────────────────────────────────

def parse_derm7pt_metadata(meta_csv: Path) -> pd.DataFrame:
    """
    Load meta.csv and compute ABC scores.

    Returns
    -------
    pd.DataFrame with A_score, B_score, C_score columns added.
    """
    df = pd.read_csv(meta_csv)
    # Normalise column names: lowercase + strip
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    print(f"[Derm7ptLoader] Columns: {list(df.columns)}")

    # Compute ABC for every row
    abc = [_compute_abc(row) for _, row in df.iterrows()]
    df["A_score"] = [r[0] for r in abc]
    df["B_score"] = [r[1] for r in abc]
    df["C_score"] = [r[2] for r in abc]
    df["dataset_source"] = "Derm7pt"

    print(
        f"[Derm7ptLoader] Loaded {len(df)} images  "
        f"| A ∈ [{df.A_score.min():.2f}, {df.A_score.max():.2f}]  "
        f"| B ∈ [{df.B_score.min():.2f}, {df.B_score.max():.2f}]  "
        f"| C ∈ [{df.C_score.min():.2f}, {df.C_score.max():.2f}]"
    )
    return df


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class Derm7ptDataset(Dataset):
    """
    PyTorch Dataset for Derm7pt.

    Uses the 'derm' column for dermoscopy image paths.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        images_dir: Path,
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
    ):
        self.df       = metadata_df.reset_index(drop=True)
        self.img_dir  = images_dir
        self.augment  = augment
        self.img_size = image_size

        # Dermoscopy-specific transforms: color constancy + hair augmentation
        # Reference: Barata et al. (2014), Jütte et al. (2024), Perez et al. (2018)
        self.img_tf = build_dermoscopy_transforms(
            image_size=image_size,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD,
            augment=augment,
            color_constancy=True,
        )

        # Filter rows with resolvable images
        valid = [self._resolve(row) is not None for _, row in self.df.iterrows()]
        n_before = len(self.df)
        self.df = self.df[valid].reset_index(drop=True)
        dropped = n_before - len(self.df)
        if dropped > 0:
            print(f"[Derm7ptDataset] Dropped {dropped} rows (missing images).")

    def _resolve(self, row: pd.Series) -> Optional[Path]:
        """Resolve dermoscopy image path from 'derm' column."""
        # Try 'derm' column first (official Derm7pt column name)
        for col in ["derm", "derm_image", "image_path", "image"]:
            if col in row.index and pd.notna(row[col]):
                img_val = str(row[col])
                # Try as-is
                p = Path(img_val)
                if p.exists():
                    return p
                # Try relative to images_dir
                p2 = self.img_dir / img_val
                if p2.exists():
                    return p2
                # Try just the filename
                p3 = self.img_dir / Path(img_val).name
                if p3.exists():
                    return p3
        return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        row  = self.df.iloc[idx]
        path = self._resolve(row)

        image   = Image.open(path).convert("RGB")
        image_t = self.img_tf(image)
        mask_t  = torch.zeros(1, self.img_size, self.img_size)  # no masks in Derm7pt

        abc = torch.tensor(
            [row.A_score, row.B_score, row.C_score],
            dtype=torch.float32,
        )

        diag_col = next(
            (c for c in row.index if "diagnosis" in c.lower()), None
        )
        diag = str(row[diag_col]) if diag_col else "unknown"

        meta = {
            "image_id"      : str(row.name),
            "diagnosis"     : diag,
            "dataset_source": "Derm7pt",
        }
        return image_t, mask_t, abc, meta


# ─────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────

def load_derm7pt(
    derm7pt_dir: Path = DERM7PT_DIR,
) -> Tuple[Derm7ptDataset, Derm7ptDataset, Derm7ptDataset]:
    """
    Load Derm7pt using official train/val/test index splits.

    The index CSV files contain a column named 'indexes' with
    integer row positions into meta.csv.

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)
    """
    meta_csv   = derm7pt_dir / "meta" / "meta.csv"
    train_csv  = derm7pt_dir / "meta" / "train_indexes.csv"
    val_csv    = derm7pt_dir / "meta" / "valid_indexes.csv"
    test_csv   = derm7pt_dir / "meta" / "test_indexes.csv"
    images_dir = derm7pt_dir / "images"

    df_full = parse_derm7pt_metadata(meta_csv)

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        # Official splits: index files contain integer row positions
        def _load_idx(path):
            idx_df = pd.read_csv(path)
            # Column name may be 'indexes', 'index', or the first column
            col = idx_df.columns[0]
            return idx_df[col].astype(int).tolist()

        train_idx = _load_idx(train_csv)
        val_idx   = _load_idx(val_csv)
        test_idx  = _load_idx(test_csv)

        # iloc-based split (integer row positions)
        train_df = df_full.iloc[train_idx].copy()
        val_df   = df_full.iloc[val_idx].copy()
        test_df  = df_full.iloc[test_idx].copy()

        print(
            f"[Derm7ptLoader] Official splits used — "
            f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
        )
    else:
        # Fallback: random 70/15/15 split
        from sklearn.model_selection import train_test_split
        train_df, tmp_df = train_test_split(
            df_full, test_size=0.30, random_state=RANDOM_SEED
        )
        val_df, test_df = train_test_split(
            tmp_df, test_size=0.50, random_state=RANDOM_SEED
        )
        print("[Derm7ptLoader] No index files found — using random 70/15/15 split")

    train_ds = Derm7ptDataset(train_df, images_dir, augment=True)
    val_ds   = Derm7ptDataset(val_df,   images_dir, augment=False)
    test_ds  = Derm7ptDataset(test_df,  images_dir, augment=False)

    print(
        f"[Derm7ptLoader] Datasets — "
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return train_ds, val_ds, test_ds
