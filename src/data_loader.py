"""
data_loader.py
--------------
Data loading, preprocessing, and DataLoader construction for HAM10000.

Design decisions:
  - Stratified train / val / test split to preserve class distribution.
  - Weighted random sampling during training to mitigate the severe class
    imbalance in HAM10000 (Nevi account for ~67 % of samples).
  - Patient-level deduplication: some patients appear multiple times;
    all images belonging to one patient are kept in the same split to
    prevent data leakage between train and test sets.
  - HAM10000-specific normalization constants are used instead of
    generic ImageNet statistics.

References:
    Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset,
    a large collection of multi-source dermatoscopic images of common pigmented
    skin lesions. Scientific Data, 5, 180161.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

from src import config


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set seeds for full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metadata() -> pd.DataFrame:
    """
    Load HAM10000 metadata and resolve image file paths.

    The dataset is split into two image directories (part_1 and part_2).
    This function:
      1. Reads the CSV file.
      2. Scans both image directories and builds an image_id → path mapping.
      3. Attaches an integer label and the resolved file path to every row.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: image_id, dx, label (int), filepath.

    Raises
    ------
    FileNotFoundError
        If the metadata CSV is missing.
    ValueError
        If no images are found in the configured image directories.
    """
    if not config.METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata file not found: {config.METADATA_CSV}")

    df = pd.read_csv(config.METADATA_CSV)

    # Build image_id → absolute path mapping
    image_path_map: Dict[str, Path] = {}
    for img_dir in config.IMAGE_DIRS:
        if img_dir.exists():
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    image_path_map[img_file.stem] = img_file

    if not image_path_map:
        raise ValueError(
            "No images found. Check IMAGE_DIRS in config.py and your data folder."
        )

    # Attach resolved paths
    df["filepath"] = df["image_id"].map(image_path_map)

    # Drop rows whose images could not be located
    missing = df["filepath"].isna().sum()
    if missing > 0:
        print(f"[WARNING] {missing} rows have no matching image file and will be dropped.")
    df = df.dropna(subset=["filepath"]).reset_index(drop=True)

    # Encode class labels as integers (sorted alphabetically for stability)
    label_map = {name: idx for idx, name in enumerate(config.CLASS_LABELS)}
    df["label"] = df["dx"].map(label_map)

    print(
        f"[DataLoader] Loaded {len(df)} images across {df['label'].nunique()} classes "
        f"from {len(config.IMAGE_DIRS)} directories."
    )
    return df


def stratified_patient_split(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a patient-aware stratified split.

    All images belonging to the same patient (lesion_id) are assigned to the
    same partition, preventing data leakage.  The split is stratified by
    diagnostic class at the patient level.

    Parameters
    ----------
    df : pd.DataFrame
        Full metadata DataFrame with columns 'lesion_id', 'dx', 'label'.
    train_ratio : float
        Fraction of patients to assign to the training set.
    val_ratio : float
        Fraction of patients to assign to the validation set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df
    """
    # One representative label per patient (most frequent diagnosis)
    patient_df = (
        df.groupby("lesion_id")["dx"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"dx": "primary_dx"})
    )

    test_ratio = 1.0 - train_ratio - val_ratio

    # Split patients → train / (val + test)
    train_patients, temp_patients = train_test_split(
        patient_df["lesion_id"].values,
        test_size=(1.0 - train_ratio),
        stratify=patient_df["primary_dx"].values,
        random_state=seed,
    )

    # Split (val + test) → val / test
    temp_df = patient_df[patient_df["lesion_id"].isin(temp_patients)]
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_patients, test_patients = train_test_split(
        temp_df["lesion_id"].values,
        test_size=relative_test_ratio,
        stratify=temp_df["primary_dx"].values,
        random_state=seed,
    )

    train_df = df[df["lesion_id"].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df["lesion_id"].isin(val_patients)].reset_index(drop=True)
    test_df  = df[df["lesion_id"].isin(test_patients)].reset_index(drop=True)

    print(
        f"[DataLoader] Split sizes — "
        f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df["dx"].value_counts().to_dict()
        print(f"  {split_name} class distribution: {dist}")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_transforms(split: str) -> T.Compose:
    """
    Build torchvision transform pipelines.

    Training augmentations follow best practices for dermoscopy images
    (Codella et al., 2018; Kawahara & Hamarneh, 2016):
      - Random horizontal / vertical flips (lesions are rotation-invariant)
      - Random rotation up to 180 degrees
      - Colour jitter (illumination variation between imaging devices)
      - Random resized crop

    Parameters
    ----------
    split : str
        One of 'train', 'val', or 'test'.

    Returns
    -------
    torchvision.transforms.Compose
    """
    normalize = T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)

    if split == "train":
        return T.Compose([
            T.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
            T.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            normalize,
        ])
    else:  # val / test
        return T.Compose([
            T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            T.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 dermoscopy image collection.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Subset DataFrame with columns 'filepath' and 'label'.
    transform : torchvision.transforms.Compose
        Image transformation pipeline.
    """

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[T.Compose] = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(str(row["filepath"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


# ─────────────────────────────────────────────
# Weighted sampler for class imbalance
# ─────────────────────────────────────────────

def build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-samples minority classes.

    Each sample's weight is set to the inverse frequency of its class,
    so every class is seen with equal probability during training.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split with a 'label' column.

    Returns
    -------
    torch.utils.data.WeightedRandomSampler
    """
    class_counts = np.bincount(train_df["label"].values, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_df["label"].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct train, validation, and test DataLoaders.

    The training loader uses a WeightedRandomSampler to address class
    imbalance.  Validation and test loaders iterate sequentially without
    replacement.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames produced by stratified_patient_split().

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        train_loader, val_loader, test_loader
    """
    train_dataset = HAM10000Dataset(train_df, transform=get_transforms("train"))
    val_dataset   = HAM10000Dataset(val_df,   transform=get_transforms("val"))
    test_dataset  = HAM10000Dataset(test_df,  transform=get_transforms("test"))

    sampler = build_weighted_sampler(train_df)

    # pin_memory and persistent_workers are only effective with CUDA
    import torch as _torch
    _pin     = config.PIN_MEMORY and _torch.cuda.is_available()
    _persist = config.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=_pin,
        persistent_workers=_persist,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=_pin,
        persistent_workers=_persist,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=_pin,
        persistent_workers=_persist,
    )

    return train_loader, val_loader, test_loader
