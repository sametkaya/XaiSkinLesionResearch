"""
score_ham10000.py
-----------------
Script 2 of 3 — HAM10000 ABC Scoring & Dataset Construction

Applies the trained ABC regressor (Script 1) and the image-processing
ABC scorer to all 10,015 HAM10000 images, producing a scored dataset
intended for public release.

Two independent scoring methods are used:
  Method 1 — DL Regressor (abc_dl):
    ABCRegressor inference on each image.
  Method 2 — Image Processing (abc_ip):
    Algorithm-based scoring (principal-axis asymmetry, compactness
    index, dermoscopic color detection).

Final score per image: mean of both methods (abc_mean).

Output files (saved to experiment directory):
  10_ham10000_abc_scores/
    ham10000_abc_scores.csv
    histograms/score_distributions.png
    scatter/method_agreement.png
    per_class_abc.png
    result.txt
  dataset/
    ham10000_abc_scores.csv
    ham10000_abc_scored.h5 (optional)
    dataset-metadata.json
    README.md

Usage
-----
  # Mevcut run dizinine kaydet (varsayılan):
  python score_ham10000.py \\
      --abc-checkpoint results/run_01_xai_dermoscopy/09_abc_regressor/checkpoints/best_abc_model.pth

  # Özel deney dizinine kaydet:
  python score_ham10000.py \\
      --abc-checkpoint results/run_01_xai_dermoscopy/09_abc_regressor/checkpoints/best_abc_model.pth \\
      --experiment-dir results/experiment_01_full_mask_scoring \\
      --mask-dir datas/HAM10000/segmentations \\
      --no-hdf5
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.abc.config_abc import (
    make_abc_experiment_dir, RESULTS_DIR,
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
    HAM10000_SCORE_BATCH, HAM10000_SCORE_WORKERS,
    DATASET_HDF5_NAME, DATASET_CSV_NAME, KAGGLE_CARD_NAME,
    ABC_CRITERIA, HAM10000_MASK_DIR,
)
from src.abc.abc_model      import ABCRegressor
from src.abc.ham10000_scorer import HAM10000Scorer
from src import config as ham_cfg
from src.data_loader import load_metadata


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score HAM10000 with ABC criteria",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--abc-checkpoint", type=str, required=True,
        help="Path to best_abc_model.pth from train_abc_regressor.py"
    )
    p.add_argument(
        "--mask-dir", type=str,
        default=str(HAM10000_MASK_DIR),
        help="Directory containing segmentation masks."
    )
    p.add_argument(
        "--experiment-dir", type=str, default=None,
        help="Custom experiment directory (e.g. results/experiment_01_full_mask_scoring). "
             "If not set, uses the latest run_XX_xai_dermoscopy/ folder."
    )
    p.add_argument(
        "--unet-weights", type=str, default=None,
        help="Path to trained U-Net weights for segmentation (optional)."
    )
    p.add_argument(
        "--no-hdf5", action="store_true",
        help="Skip HDF5 creation (faster, CSV only)."
    )
    return p.parse_args()


# ─────────────────────────────────────────────
# HDF5 builder
# ─────────────────────────────────────────────

def build_hdf5(
    scores_df: pd.DataFrame,
    image_dirs: list,
    output_path: Path,
    image_size: int = IMAGE_SIZE,
) -> None:
    """
    Build an HDF5 file containing HAM10000 images and ABC scores.

    The file provides a ready-to-use PyTorch-compatible dataset for
    ABC-guided model training, evaluation, and ABC counterfactual research.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of HAM10000Scorer.run().
    image_dirs : list of Path
        HAM10000 image directories.
    output_path : Path
        Output .h5 file path.
    image_size : int
        Spatial resolution for stored images.
    """
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

    def find_image(image_id: str):
        for d in image_dirs:
            for ext in ("jpg", "jpeg", "png"):
                p = d / f"{image_id}.{ext}"
                if p.exists():
                    return p
        return None

    N    = len(scores_df)
    print(f"\n[HDF5Builder] Creating {output_path.name} ({N:,} images) …")

    label_map = {cls: i for i, cls in enumerate(sorted(ham_cfg.CLASS_NAMES.keys()))}

    with h5py.File(output_path, "w") as f:
        # Pre-allocate datasets
        images  = f.create_dataset("images",   (N, 3, image_size, image_size), dtype="float32",
                                    chunks=(32, 3, image_size, image_size))
        labels  = f.create_dataset("labels",   (N,),    dtype="int64")
        abc_dl  = f.create_dataset("abc_dl",   (N, 3),  dtype="float32")
        abc_ip  = f.create_dataset("abc_ip",   (N, 3),  dtype="float32")
        abc_mn  = f.create_dataset("abc_mean", (N, 3),  dtype="float32")

        # Metadata as variable-length strings
        dt_str = h5py.string_dtype(encoding="utf-8")
        img_ids = f.create_dataset("image_ids", (N,), dtype=dt_str)
        dxs     = f.create_dataset("dx",         (N,), dtype=dt_str)

        for i, row in tqdm(scores_df.iterrows(), total=N, desc="Building HDF5"):
            idx   = int(i)
            p     = find_image(row.image_id)
            if p is not None:
                img_t = tf(Image.open(p).convert("RGB")).numpy()
            else:
                img_t = np.zeros((3, image_size, image_size), dtype="float32")

            images[idx]   = img_t
            labels[idx]   = label_map.get(row.dx, 0)
            abc_dl[idx]   = [row.A_dl,   row.B_dl,   row.C_dl]
            abc_ip[idx]   = [row.A_ip,   row.B_ip,   row.C_ip]
            abc_mn[idx]   = [row.A_mean, row.B_mean, row.C_mean]
            img_ids[idx]  = row.image_id
            dxs[idx]      = row.dx

        # Metadata attributes
        f.attrs["description"] = (
            "HAM10000 dermoscopic images with pseudo-ABC scores "
            "(Asymmetry, Border, Color). "
            "Scores produced by two methods: "
            "(1) EfficientNet-B0 ABC regressor trained on PH2+Derm7pt, "
            "(2) image-processing algorithm (principal-axis asymmetry, "
            "compactness index, dermoscopic color detection). "
            "License: CC BY-NC 4.0."
        )
        f.attrs["image_size"] = image_size
        f.attrs["n_classes"]  = 7
        f.attrs["class_names"] = str(sorted(ham_cfg.CLASS_NAMES.keys()))
        f.attrs["abc_criteria"] = "A=Asymmetry, B=Border_Irregularity, C=Color_Variegation"
        f.attrs["source_dataset"] = "HAM10000 (Tschandl et al., 2018)"
        f.attrs["abc_training_data"] = "PH2 (Mendonça et al., 2013) + Derm7pt (Kawahara et al., 2019)"

    print(f"[HDF5Builder] Saved: {output_path}  ({output_path.stat().st_size / 1e9:.2f} GB)")


# ─────────────────────────────────────────────
# Kaggle dataset card
# ─────────────────────────────────────────────

def write_kaggle_card(output_dir: Path, n_images: int) -> None:
    """Write Kaggle dataset-metadata.json and README.md."""
    card = {
        "title"      : "HAM10000 ABC-Scored Dermoscopy Dataset",
        "id"         : "ham10000-abc-scored",
        "licenses"   : [{"name": "CC BY-NC 4.0"}],
        "keywords"   : [
            "dermoscopy", "skin lesion", "ABCD rule",
            "explainable AI", "HAM10000",
        ],
        "collaborators": [],
        "data"       : [
            {"description": "Per-image ABC scores (CSV)"},
            {"description": "Images + ABC scores (HDF5)"},
            {"description": "Methodology README"},
        ],
    }
    (output_dir / KAGGLE_CARD_NAME).write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )

    readme = f"""# HAM10000 ABC-Scored Dermoscopy Dataset

## Overview
This dataset extends the [HAM10000](https://doi.org/10.1038/sdata.2018.161)
dermoscopy dataset ({n_images:,} images, 7 diagnostic classes) with
pseudo-ABC (Asymmetry, Border irregularity, Color variegation) scores
for each image.

## ABC Scoring Methodology

### Why ABC (not ABCD)?
The diameter (D) criterion is excluded because HAM10000 aggregates images
from multiple devices at varying magnifications. After standard 224x224 px
resizing, no pixel-to-physical-scale conversion is possible. See:
Choi et al. (2024), *Applied Sciences* 14(22), 10294.

### Scoring methods

#### Method 1: DL Regressor (abc_dl)
EfficientNet-B0 backbone (pre-trained on HAM10000) + multi-output
regression head trained on PH2 + Derm7pt with ground-truth ABC annotations.

#### Method 2: Image Processing (abc_ip)
- **A (Asymmetry):** Principal-axis overlap ratio after reflection
- **B (Border):** Compactness index (4pi*Area/Perimeter^2), inverted
- **C (Color):** Count of detected standard dermoscopic colors (Argenziano 1998)

### Normalisation
All scores normalised to [0.0, 1.0]:
| Score | 0.0 | 1.0 |
|-------|-----|-----|
| A | Symmetric | Fully asymmetric |
| B | Circular border | Maximally irregular |
| C | Monochromatic | 6 colors present |

## CSV Columns
`image_id, dx, A_dl, B_dl, C_dl, A_ip, B_ip, C_ip, A_mean, B_mean, C_mean, mask_source`

## License
CC BY-NC 4.0 — compatible with the HAM10000 source license.

## Citation
If you use this dataset, please cite:
1. Tschandl, P., et al. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161.
2. Mendonca, T., et al. (2013). PH2. *IEEE EMBC*, 5437-5440.
3. Kawahara, J., et al. (2019). Seven-point checklist. *IEEE JBHI*, 23(2), 538-546.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[DatasetBuilder] Kaggle card and README saved to {output_dir}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Experiment directory ──────────────────
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        exp_dir = make_abc_experiment_dir(RESULTS_DIR)

    scores_dir  = exp_dir / "10_ham10000_abc_scores"
    dataset_dir = exp_dir / "dataset"
    for sub in [scores_dir, scores_dir / "histograms", scores_dir / "scatter",
                dataset_dir]:
        sub.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  HAM10000 ABC Scoring Pipeline")
    print(f"  Experiment: {exp_dir}")
    print(f"  Device    : {device}")
    print("=" * 60 + "\n")

    # ── Load ABC regressor ────────────────────
    ckpt_path = Path(args.abc_checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] ABC checkpoint not found: {ckpt_path}")
        sys.exit(1)

    state = torch.load(ckpt_path, map_location=device)

    # Auto-detect architecture from checkpoint keys
    keys = list(state["model_state_dict"].keys())
    use_sord = any("feat_head" in k or "ordinal_head" in k for k in keys)
    num_bins = 5 if use_sord else 1

    model = ABCRegressor(
        backbone_weights=None,
        freeze_backbone=False,
        num_bins=num_bins,
    )
    model.load_state_dict(state["model_state_dict"])
    print(f"[Main] Architecture: {'SORD ordinal' if use_sord else 'Regression'} head")
    model = model.to(device)
    model.eval()
    print(f"[Main] ABC regressor loaded from {ckpt_path}")

    # ── Load HAM10000 metadata ─────────────────
    print("─" * 60)
    print("  Loading HAM10000 metadata …")
    print("─" * 60)
    metadata_df = load_metadata()

    mask_dir     = Path(args.mask_dir) if args.mask_dir else None
    unet_weights = Path(args.unet_weights) if args.unet_weights else None

    # ── Score ─────────────────────────────────
    print("─" * 60)
    print("  Running ABC scoring …")
    print("─" * 60)
    scorer = HAM10000Scorer(
        abc_model   = model,
        metadata_df = metadata_df,
        image_dirs  = ham_cfg.IMAGE_DIRS,
        mask_dir    = mask_dir if (mask_dir and mask_dir.exists()) else None,
        device      = device,
        result_dir  = scores_dir,
        unet_weights= unet_weights,
    )
    scores_df = scorer.run()

    # Copy CSV to dataset dir
    csv_out = dataset_dir / DATASET_CSV_NAME
    scores_df.to_csv(csv_out, index=False)
    print(f"[Main] CSV saved to {csv_out}")

    # ── Build HDF5 ────────────────────────────
    if not args.no_hdf5:
        print("─" * 60)
        print("  Building HDF5 dataset …")
        print("─" * 60)
        build_hdf5(
            scores_df  = scores_df,
            image_dirs = ham_cfg.IMAGE_DIRS,
            output_path= dataset_dir / DATASET_HDF5_NAME,
        )

    # ── Kaggle card ───────────────────────────
    write_kaggle_card(dataset_dir, len(scores_df))

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print("  Scoring Complete")
    print(f"  Experiment: {exp_dir}")
    print(f"  CSV:    {csv_out}")
    if not args.no_hdf5:
        print(f"  HDF5:   {dataset_dir / DATASET_HDF5_NAME}")
    print(f"  Mean A (DL/IP): "
          f"{scores_df.A_dl.mean():.3f} / {scores_df.A_ip.mean():.3f}")
    print(f"  Mean B (DL/IP): "
          f"{scores_df.B_dl.mean():.3f} / {scores_df.B_ip.mean():.3f}")
    print(f"  Mean C (DL/IP): "
          f"{scores_df.C_dl.mean():.3f} / {scores_df.C_ip.mean():.3f}")
    if "mask_source" in scores_df.columns:
        n_expert = (scores_df["mask_source"] == "expert").sum()
        n_otsu   = (scores_df["mask_source"] == "otsu").sum()
        print(f"  Masks : {n_expert:,} expert + {n_otsu:,} Otsu")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()