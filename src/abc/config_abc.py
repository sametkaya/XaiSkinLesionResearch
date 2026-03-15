"""
src/abc/config_abc.py
---------------------
Configuration for the ABC regressor pipeline:
  - PH2 + Derm7pt data loading and normalisation
  - ABC regressor training hyperparameters
  - HAM10000 pseudo-scoring settings
  - Dataset export (HDF5 + CSV + Kaggle card)
  - ABC-guided counterfactual hyperparameters (v2 — fixed)

Notes on the "D" (Diameter) criterion
--------------------------------------
Diameter is intentionally excluded from this pipeline for two reasons:

1. *Technical*: HAM10000 aggregates images from multiple devices at
   varying magnifications. After standard resizing to 224×224 pixels,
   no pixel-to-physical-scale conversion is possible.  See:
   Choi et al. (2024). Enhancing Skin Lesion Classification Performance
   with the ABC Ensemble Model. Applied Sciences, 14(22), 10294.

2. *Clinical*: The 6 mm diameter threshold has declining sensitivity
   (≈40 % of contemporary melanomas) as dermoscopy enables earlier
   detection of sub-centimetre lesions. See:
   Ferris, L. K., et al. (2021). Re-evaluating the ABCD criteria using
   a consecutive series of melanomas. JAAD, 84(5), 1311–1318.

References
----------
Stolz, W., et al. (1994). ABCD rule of dermatoscopy. Eur. J. Dermatol., 4, 521–527.
Mendonça, T., et al. (2013). PH2. IEEE EMBC, 5437–5440.
Kawahara, J., et al. (2019). Seven-point checklist. IEEE JBHI, 23(2), 538–546.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Base directories (relative to project root)
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "datas"
RESULTS_DIR = BASE_DIR / "results"

# ─────────────────────────────────────────────
# PH2 dataset paths
# ─────────────────────────────────────────────
# Download: https://www.fc.up.pt/addi/ph2%20database.html
# Structure expected:
#   datas/PH2Dataset/
#     PH2_dataset.txt          ← metadata / annotations
#     PH2 Images/
#       IMD002/
#         IMD002_Dermoscopic_Image/IMD002.bmp
#         IMD002_lesion/IMD002_lesion.bmp       ← binary mask
PH2_DIR          = DATA_DIR / "PH2"
PH2_METADATA_TXT = PH2_DIR  / "PH2_dataset.txt"   # annotation file (see note below)
PH2_IMAGES_DIR   = PH2_DIR  / "trainx"             # dermoscopic images
PH2_MASKS_DIR    = PH2_DIR  / "trainy"             # binary segmentation masks
# NOTE: PH2_dataset.txt contains ABC ground-truth annotations (asymmetry, colors, etc.)
# If missing from your download, get it from: https://www.fc.up.pt/addi/ph2%20database.html
# The Kaggle mirror (athina123/ph2dataset) may not include this file.

# ─────────────────────────────────────────────
# Derm7pt dataset paths
# ─────────────────────────────────────────────
# Download: https://derm.cs.sfu.ca/Download.html
# Structure expected:
#   datas/derm7pt/
#     meta/
#       meta.csv               ← annotations
#       train_indexes.csv
#       valid_indexes.csv
#       test_indexes.csv
#     images/
#       *.jpg  (dermoscopy images)
#     masks/
#       *.png  (segmentation masks, if available)
DERM7PT_DIR       = DATA_DIR  / "DERM7PT"
DERM7PT_META_CSV  = DERM7PT_DIR / "meta" / "meta.csv"
DERM7PT_TRAIN_CSV = DERM7PT_DIR / "meta" / "train_indexes.csv"
DERM7PT_VAL_CSV   = DERM7PT_DIR / "meta" / "valid_indexes.csv"
DERM7PT_TEST_CSV  = DERM7PT_DIR / "meta" / "test_indexes.csv"
DERM7PT_IMG_DIR   = DERM7PT_DIR / "images"

# ─────────────────────────────────────────────
# HAM10000 segmentation masks (ISIC 2018 Task 1)
# ─────────────────────────────────────────────
# Download from: https://challenge.isic-archive.com/data/#2018
# Only HAM10000 subset (Task 1 ground truth masks)
HAM10000_MASK_DIR = DATA_DIR / "HAM10000" / "segmentations"

# ─────────────────────────────────────────────
# ABC label definitions
# ─────────────────────────────────────────────
# Three dermoscopic criteria from Stolz (1994), all normalised to [0, 1]:
#
#   A — Asymmetry:
#       0.0 = symmetric in both axes
#       0.5 = asymmetric in one axis
#       1.0 = asymmetric in both axes
#
#   B — Border irregularity:
#       Continuous 0–1 (derived from 0–8 segment scores in PH2;
#       absent/present binary mapped to 0.0/1.0 in Derm7pt)
#
#   C — Color variegation:
#       Continuous 0–1 (derived from 1–6 color count in PH2;
#       sum of color-related 7-pt criteria normalised in Derm7pt)
#
# See normalisation details in ph2_loader.py and derm7pt_loader.py.

ABC_CRITERIA    = ["A", "B", "C"]
ABC_NAMES       = {
    "A": "Asymmetry",
    "B": "Border Irregularity",
    "C": "Color Variegation",
}
NUM_ABC         = 3   # regressor output dimension

# ─────────────────────────────────────────────
# Image preprocessing (shared with main pipeline)
# ─────────────────────────────────────────────
IMAGE_SIZE = 224
IMAGE_MEAN = (0.7630392, 0.5456477, 0.5700950)   # HAM10000-specific
IMAGE_STD  = (0.1409286, 0.1526128, 0.1694007)

# ─────────────────────────────────────────────
# ABC Regressor training (v2 — improved)
# ─────────────────────────────────────────────
ABC_BATCH_SIZE          = 16    # small datasets → small batch
ABC_NUM_EPOCHS          = 100   # more epochs with ordinal loss
ABC_LEARNING_RATE       = 2e-4  # slightly lower for stability
ABC_WEIGHT_DECAY        = 1e-4
ABC_EARLY_STOP_PATIENCE = 20    # more patience for ordinal convergence
ABC_FREEZE_EPOCHS       = 10    # epochs to freeze backbone
ABC_NUM_WORKERS         = 2
ABC_USE_AMP             = True
ABC_DROPOUT             = 0.4

# Ordinal loss settings
# Reference: Díaz & Marathe (CVPR 2019), Cao et al. (Pattern Recog. Letters 2020)
ABC_LOSS_TYPE           = "ordinal_huber"  # "sord" | "ordinal_huber" | "huber"
ABC_ORDINAL_BINS        = 5     # for SORD: number of ordinal bins per criterion
ABC_SORD_SIGMA          = 1.5   # Gaussian bandwidth for SORD soft labels
ABC_RANK_LAMBDA         = 0.05  # rank consistency penalty weight (small: avoid interfering with MAE)

# Color constancy preprocessing
# Reference: Barata et al. (IEEE J-BHI 2014)
ABC_COLOR_CONSTANCY     = True  # apply Shades of Gray normalization
ABC_SOG_P               = 6.0   # p=6 optimal for dermoscopy (Barata 2014)

# Test-time augmentation
# Reference: Perez et al. (MICCAI Workshop 2018)
ABC_TTA_N               = 1     # TTA disabled (set >1 only after verifying denorm pipeline)

# ─────────────────────────────────────────────
# Segmentation (U-Net inference for Derm7pt)
# ─────────────────────────────────────────────
# We use a lightweight U-Net trained on ISIC 2018 Task 1 data.
# Pre-trained weights are loaded if available; otherwise segmentation
# falls back to Otsu thresholding on the green channel.
SEGMENTATION_THRESHOLD  = 0.5   # binary threshold on sigmoid output
SEGMENTATION_MIN_AREA   = 500   # minimum lesion area in pixels (post-resize)

# ─────────────────────────────────────────────
# Image-processing (IP) ABC scorer
# ─────────────────────────────────────────────
# Asymmetry: overlap ratio after flipping along principal axis
# Border:    normalised gradient entropy of lesion boundary
# Color:     number of distinct ISIC-standard colors detected
IP_COLOR_BINS           = 32    # histogram bins per channel for color analysis
IP_BORDER_SIGMA         = 2.0   # Gaussian σ for border smoothing
IP_COLOR_THRESHOLD      = 0.05  # minimum fraction to count a color as present

# Standard dermoscopic colors (HSV approximations)
# Reference: Argenziano et al. (1998). Dermoscopy of pigmented skin lesions.
DERMOSCOPIC_COLORS = {
    "black"       : {"h": (0,   20),  "s": (0,   80),  "v": (0,   60)},
    "dark_brown"  : {"h": (10,  25),  "s": (50,  255), "v": (30,  100)},
    "light_brown" : {"h": (15,  35),  "s": (30,  180), "v": (100, 200)},
    "red"         : {"h": (0,   15),  "s": (80,  255), "v": (80,  255)},
    "blue_gray"   : {"h": (100, 140), "s": (20,  150), "v": (50,  180)},
    "white"       : {"h": (0,   180), "s": (0,   40),  "v": (200, 255)},
}

# ─────────────────────────────────────────────
# HAM10000 scoring
# ─────────────────────────────────────────────
HAM10000_SCORE_BATCH    = 64    # inference batch size
HAM10000_SCORE_WORKERS  = 2

# ─────────────────────────────────────────────
# Dataset export
# ─────────────────────────────────────────────
DATASET_HDF5_NAME       = "ham10000_abc_scored.h5"
DATASET_CSV_NAME        = "ham10000_abc_scores.csv"
KAGGLE_CARD_NAME        = "dataset-metadata.json"

# ─────────────────────────────────────────────
# ABC-guided Counterfactual  (v4 — segmentation-guided)
# ─────────────────────────────────────────────
# Loss formulation:
#   L = λ_cls  · CE(f(x + M⊙δ), c_tgt)       ← class change
#     + λ_A    · |g(x + M⊙δ)_A − s_A|         ← asymmetry preservation
#     + λ_B    · |g(x + M⊙δ)_B − s_B|         ← border preservation
#     + λ_C    · |g(x + M⊙δ)_C − s_C|         ← color preservation
#     + λ_l1   · ‖M⊙δ‖₁                        ← sparsity
#     + λ_TV   · TV(M⊙δ)                        ← spatial smoothness
#     + λ_perc · Σ‖φ_j(x) − φ_j(x+M⊙δ)‖²      ← perceptual similarity
#
# where M is a soft lesion mask (dilated + Gaussian-blurred).
#
# v4 key changes from v3:
#
#   1. Segmentation-guided perturbation (M⊙δ):
#      δ_effective = soft_mask ⊙ δ_raw  at each optimization step.
#      This is "hard masking" per RCSB (Sobieski et al., ICLR 2025):
#      pixels outside the lesion mask remain EXACTLY unchanged.
#      Clinical motivation: the counterfactual question is "what if the
#      LESION looked different?", not "what if the skin looked different?"
#
#   2. Perceptual loss (VGG-16 feature matching):
#      L_perc = Σ_j ‖φ_j(x) − φ_j(x_cf)‖² / (C_j·H_j·W_j)
#      at layers relu2_2 and relu3_3 of VGG-16 (no batch norm).
#      This prevents adversarial high-frequency perturbations and
#      preserves semantic structure within the lesion.
#      Reference: Johnson et al. (ECCV 2016); DiME uses λ_perc=30.
#
#   3. Gaussian blur on δ at each step:
#      δ = GaussianBlur(δ, kernel=5, σ=1.0) after each gradient step.
#      This acts as an implicit smoothness prior, preventing per-pixel
#      noise even within the lesion mask.
#      Reference: Mirror-CFE (2024) uses blur as validity test.
#
#   4. Soft mask preprocessing:
#      Binary mask → dilate 3px → Gaussian blur σ=5 → [0,1] float mask.
#      Smooth edges prevent visible seams at the lesion boundary.
#      Reference: SoftSeg (Medical Image Analysis, 2021).
#
#   5. SSIM evaluation metric added for counterfactual quality.
#
# References:
#   Sobieski et al. (2025). RCSB. ICLR 2025.
#   Johnson et al. (2016). Perceptual Losses. ECCV 2016.
#   Jeanneret et al. (2022). DiME. ACCV 2022.
#   Rudin, Osher & Fatemi (1992). TV denoising. Physica D.
#
ABC_CF_MAX_ITER          = 300
ABC_CF_LEARNING_RATE     = 0.005
ABC_CF_LAMBDA_CLS        = 5.0    # classification loss weight
ABC_CF_LAMBDA_A          = 1.0    # asymmetry preservation weight
ABC_CF_LAMBDA_B          = 0.8    # border preservation weight
ABC_CF_LAMBDA_C          = 0.6    # color preservation weight
ABC_CF_LAMBDA_L1         = 0.05   # pixel sparsity weight
ABC_CF_LAMBDA_TV         = 0.005  # total variation weight (lower with mask)
ABC_CF_LAMBDA_PERC       = 0.1    # perceptual loss weight (VGG features)
ABC_CF_CONFIDENCE_THRES  = 0.75   # target class probability threshold
ABC_CF_NUM_IMAGES        = 10     # images per class transition pair
ABC_CF_PIXEL_THRESHOLD   = 0.02   # threshold for sparsity mask

# Segmentation-guided mask preprocessing
ABC_CF_MASK_DILATE_PX    = 3      # dilate mask by N pixels (capture border)
ABC_CF_MASK_BLUR_SIGMA   = 5.0    # Gaussian blur σ for soft mask edges
ABC_CF_DELTA_BLUR_SIGMA  = 1.0    # Gaussian blur σ applied to δ per step
ABC_CF_DELTA_BLUR_KERNEL = 5      # kernel size for δ blur

# Transition pairs for ABC-guided CF evaluation
# (source_class, target_class)
ABC_CF_PAIRS = [
    ("nv",  "mel"),   # benign → malignant (most clinically relevant)
    ("mel", "nv"),    # malignant → benign
    ("bkl", "mel"),   # confusable pair
    ("akiec", "bcc"), # confusable pair
]

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# Experiment folder helpers
# ─────────────────────────────────────────────
def _next_experiment_dir(base: Path, prefix: str = "experiment") -> Path:
    """
    Return the next available experiment directory path.

    Parameters
    ----------
    base : Path
        Parent results directory.
    prefix : str
        Folder prefix (default: "experiment").

    Returns
    -------
    Path
        New experiment directory path (not yet created).
    """
    existing = sorted(base.glob(f"{prefix}_*"))
    if not existing:
        return base / f"{prefix}_01"
    last = existing[-1].name
    num  = int(last.split("_")[-1])
    return base / f"{prefix}_{num + 1:02d}"


def make_abc_experiment_dir(base: Path = RESULTS_DIR) -> Path:
    """
    Return the shared run directory for the ABC pipeline stages.

    Instead of creating separate abc_experiment_XX folders, all ABC
    pipeline outputs (stages 09-11) live inside the most recent
    run_XX_xai_dermoscopy folder alongside the classifier stages (01-08).

    Sub-directories created under run_XX_xai_dermoscopy/:
      09_abc_regressor/          — model training outputs
      09_abc_regressor/checkpoints/
      09_abc_regressor/plots/
      10_ham10000_abc_scores/    — per-image ABC scores
      10_ham10000_abc_scores/histograms/
      10_ham10000_abc_scores/scatter/
      10_ham10000_abc_scores/dataset/
      10_ham10000_abc_scores/segmentation/
      10_ham10000_abc_scores/segmentation/examples/
      11_abc_guided_counterfactuals/
      11_abc_guided_counterfactuals/per_class/
      11_abc_guided_counterfactuals/ablation/
      11_abc_guided_counterfactuals/metrics/
      11_abc_guided_counterfactuals/narratives/    ← v2: textual explanations

    Parameters
    ----------
    base : Path
        Parent results directory.

    Returns
    -------
    Path
        The run root directory.
    """
    # Re-use latest run folder (created by main.py) or create if needed
    existing = sorted(base.glob("run_*_xai_dermoscopy"))
    if existing:
        exp_dir = existing[-1]
    else:
        exp_dir = base / "run_01_xai_dermoscopy"

    sub_dirs = [
        exp_dir / "09_abc_regressor",
        exp_dir / "09_abc_regressor" / "checkpoints",
        exp_dir / "09_abc_regressor" / "plots",
        exp_dir / "10_ham10000_abc_scores",
        exp_dir / "10_ham10000_abc_scores" / "histograms",
        exp_dir / "10_ham10000_abc_scores" / "scatter",
        exp_dir / "10_ham10000_abc_scores" / "dataset",
        exp_dir / "10_ham10000_abc_scores" / "segmentation",
        exp_dir / "10_ham10000_abc_scores" / "segmentation" / "examples",
        exp_dir / "11_abc_guided_counterfactuals",
        exp_dir / "11_abc_guided_counterfactuals" / "per_class",
        exp_dir / "11_abc_guided_counterfactuals" / "ablation",
        exp_dir / "11_abc_guided_counterfactuals" / "metrics",
        exp_dir / "11_abc_guided_counterfactuals" / "narratives",
    ]
    for d in sub_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Config] ABC pipeline folder: {exp_dir}")
    return exp_dir