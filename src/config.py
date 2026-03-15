"""
config.py
---------
Central configuration for the XAI Skin Lesion Research project.

All hyperparameters, paths, and experiment settings are defined here
to ensure reproducibility across runs.

Reference dataset:
    Tschandl, P., Rosendahl, C., & Kittler, H. (2018).
    The HAM10000 dataset, a large collection of multi-source dermatoscopic
    images of common pigmented skin lesions.
    Scientific Data, 5, 180161. https://doi.org/10.1038/sdata.2018.161
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED: int = 42

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "datas"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# HAM10000 directory structure:
#   datas/HAM10000/
#     HAM10000_images_part_1/   ← part 1 images
#     HAM10000_images_part_2/   ← part 2 images
#     HAM10000_metadata.csv
#     segmentations/            ← ISIC 2018 Task 1 masks (optional)
HAM10000_DIR = DATA_DIR / "HAM10000"
IMAGE_DIRS = [
    HAM10000_DIR / "HAM10000_images_part_1",
    HAM10000_DIR / "HAM10000_images_part_2",
]
METADATA_CSV = HAM10000_DIR / "HAM10000_metadata.csv"

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
# HAM10000 diagnostic labels
CLASS_NAMES = {
    "nv"   : "Melanocytic Nevi",
    "mel"  : "Melanoma",
    "bkl"  : "Benign Keratosis",
    "bcc"  : "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses",
    "vasc" : "Vascular Lesions",
    "df"   : "Dermatofibroma",
}
# Sorted for stable label encoding
CLASS_LABELS = sorted(CLASS_NAMES.keys())  # alphabetical index → int
NUM_CLASSES  = len(CLASS_LABELS)           # 7

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─────────────────────────────────────────────
# Image pre-processing
# ─────────────────────────────────────────────
IMAGE_SIZE    = 224          # pixels (height = width)
IMAGE_MEAN    = (0.7630392, 0.5456477, 0.5700950)   # HAM10000-specific
IMAGE_STD     = (0.1409286, 0.1526128, 0.1694007)   # computed from training set

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
# RTX 3080 optimised (16 GB VRAM, AMP enabled)
# v2: reduced batch for better gradient estimates on minority classes
BATCH_SIZE         = 32    # smaller batch → better minority class gradients
NUM_EPOCHS         = 150   # more epochs with cosine warm restarts
LEARNING_RATE      = 3e-4  # higher initial LR with warm restarts (He et al. 2016)
WEIGHT_DECAY       = 1e-4
LR_SCHEDULER_STEP  = 10
LR_SCHEDULER_GAMMA = 0.1
EARLY_STOP_PATIENCE= 25    # more patience with longer training
NUM_WORKERS        = 2    # Windows: keep <=4; set 0 if DataLoader errors
PIN_MEMORY         = True  # faster GPU data transfer
USE_AMP            = True  # Automatic Mixed Precision (FP16)

# Loss function
# Focal Loss addresses HAM10000's severe class imbalance (nv: 67%, df: 1%)
# Reference: Lin et al. (2017). Focal loss for dense object detection. ICCV.
LOSS_TYPE          = "focal"   # "focal" | "label_smoothing" | "cross_entropy"
FOCAL_GAMMA        = 2.0       # focusing parameter — higher = more focus on hard examples
FOCAL_ALPHA        = None      # None = inverse class frequency weighting (auto-computed)
LABEL_SMOOTHING    = 0.1       # used only when LOSS_TYPE = "label_smoothing"

# LR schedule: Cosine Annealing with Warm Restarts (SGDR)
# Reference: Loshchilov & Hutter (2017). SGDR: ICLR 2017.
LR_T0              = 30    # first restart period (epochs)
LR_T_MULT          = 2     # period doubles after each restart
LR_ETA_MIN         = 1e-6  # minimum LR

# Augmentation
# Mixup: Zhang et al. (2018). mixup: Beyond empirical risk minimization. ICLR.
USE_MIXUP          = True
MIXUP_ALPHA        = 0.4   # Beta distribution parameter
USE_CUTMIX         = True
CUTMIX_ALPHA       = 1.0

# Color constancy for domain robustness (Barata et al., 2014)
USE_COLOR_CONSTANCY = True

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
# Supported: "resnet50" | "efficientnet_b0"
MODEL_NAME       = "efficientnet_b0"
PRETRAINED       = True
FREEZE_BACKBONE  = False     # Fine-tune entire network

# ─────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────
GRADCAM_NUM_SAMPLES = 30     # images per class for visual analysis

# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────
# 2000 samples: more stable, better superpixel attribution
# Reference: Ribeiro et al. (2016) recommend 1k–5k for images
LIME_NUM_SAMPLES     = 2000  # perturbation samples (2x for stability)
LIME_NUM_SUPERPIXELS = 15    # segments shown (more granular)
LIME_NUM_IMAGES      = 30    # images per class

# ─────────────────────────────────────────────
# SHAP (GradientSHAP / DeepSHAP)
# ─────────────────────────────────────────────
# Reference: Lundberg & Lee (2017). NeurIPS 2017.
SHAP_NUM_BACKGROUND  = 100   # background images for baseline distribution
SHAP_NUM_SAMPLES     = 50    # interpolation steps per explanation
SHAP_NUM_IMAGES      = 30    # images per class
SHAP_SUPERPIXEL_KS   = 4     # QuickSHIFT kernel_size

# ─────────────────────────────────────────────
# Counterfactual (gradient-based ACE-style)
# ─────────────────────────────────────────────
# Reference: Singla et al. (2023) – Explaining the black-box smoothly—
# A counterfactual approach. Medical Image Analysis, 84, 102721.
# v2: higher confidence threshold for semantically meaningful CFs
CF_MAX_ITER         = 2000   # more gradient steps → better optimisation
CF_LEARNING_RATE    = 0.005  # smaller LR → more controlled perturbation
CF_LAMBDA_L1        = 2.0    # stronger sparsity → fewer pixels changed
CF_LAMBDA_L2        = 0.5    # proximity weight
CF_CONFIDENCE_THRES = 0.85   # high threshold: real semantic class change
CF_NUM_IMAGES       = 30     # images per class
CF_PIXEL_THRESHOLD  = 0.01   # lower threshold: detect subtle changes

# ─────────────────────────────────────────────
# Evaluation / XAI Metrics
# ─────────────────────────────────────────────
FAITHFULNESS_STEPS = 20      # more steps → smoother deletion/insertion curves
FID_BATCH_SIZE     = 64

# ─────────────────────────────────────────────
# Experiment folder (auto-incremented per run)
# ─────────────────────────────────────────────
# Each full pipeline run gets its own experiment_XX folder.
# Sub-results (eda, training, gradcam …) are sub-folders inside it.
# Example layout:
#   results/
#     experiment_01/
#       eda/
#       training/
#       evaluation/
#       gradcam/
#       lime/
#       counterfactual/
#       comparison/

def _next_run_dir(base: Path) -> Path:
    """
    Return the next available full-pipeline run directory.

    Each complete run (all 4 stages) gets its own folder:
      run_01_xai_dermoscopy/
      run_02_xai_dermoscopy/
      ...

    Parameters
    ----------
    base : Path
        Parent results directory.

    Returns
    -------
    Path
        New run directory (not yet created).
    """
    existing = sorted(base.glob("run_*_xai_dermoscopy"))
    if not existing:
        return base / "run_01_xai_dermoscopy"
    last = existing[-1].name           # e.g. "run_02_xai_dermoscopy"
    num  = int(last.split("_")[1])     # 2
    return base / f"run_{num + 1:02d}_xai_dermoscopy"


# Resolved once per process — all 4 scripts share the same run folder.
# Sub-folders reflect the pipeline stage and content clearly.
EXPERIMENT_DIR = _next_run_dir(RESULTS_DIR)

# Stage 1+2: HAM10000 classifier training & evaluation
RESULT_EDA            = EXPERIMENT_DIR / "01_eda"
RESULT_TRAINING       = EXPERIMENT_DIR / "02_classifier_training"
RESULT_EVALUATION     = EXPERIMENT_DIR / "03_classifier_evaluation"

# Stage 3+4+5: XAI explanations
RESULT_GRADCAM        = EXPERIMENT_DIR / "04_gradcam_explanations"
RESULT_LIME           = EXPERIMENT_DIR / "05_lime_explanations"
RESULT_SHAP           = EXPERIMENT_DIR / "06_shap_explanations"
RESULT_COUNTERFACTUAL = EXPERIMENT_DIR / "07_counterfactual_explanations"

# Stage 6: Quantitative comparison
RESULT_COMPARISON     = EXPERIMENT_DIR / "08_xai_comparison"

# Stage 7+8+9: ABC dermoscopy scoring pipeline
RESULT_ABC_REGRESSION = EXPERIMENT_DIR / "09_abc_regressor"
RESULT_ABC_SCORING    = EXPERIMENT_DIR / "10_ham10000_abc_scores"
RESULT_ABC_CF         = EXPERIMENT_DIR / "11_abc_guided_counterfactuals"


def create_result_dirs() -> Path:
    """
    Create the full-pipeline run directory and all sub-folders.

    Returns
    -------
    Path
        The run root directory (e.g. results/run_01_xai_dermoscopy/).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for d in [
        RESULT_EDA, RESULT_TRAINING, RESULT_EVALUATION,
        RESULT_GRADCAM, RESULT_LIME, RESULT_SHAP,
        RESULT_COUNTERFACTUAL, RESULT_COMPARISON,
        RESULT_ABC_REGRESSION, RESULT_ABC_SCORING, RESULT_ABC_CF,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Config] Run folder: {EXPERIMENT_DIR}")
    return EXPERIMENT_DIR