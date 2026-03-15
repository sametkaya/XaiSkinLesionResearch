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
# RTX 3080 optimised (10 GB VRAM, AMP enabled)
BATCH_SIZE         = 64    # FP16 AMP → fits 64 on RTX 3080
NUM_EPOCHS         = 100
LEARNING_RATE      = 1e-4
WEIGHT_DECAY       = 1e-4
LR_SCHEDULER_STEP  = 10
LR_SCHEDULER_GAMMA = 0.1
EARLY_STOP_PATIENCE= 20
NUM_WORKERS        = 2    # Windows: keep <=4; set 0 if DataLoader errors
PIN_MEMORY         = True  # faster GPU data transfer
USE_AMP            = True  # Automatic Mixed Precision (FP16)

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
GRADCAM_NUM_SAMPLES = 20     # images per class for visual analysis

# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────
LIME_NUM_SAMPLES     = 1000  # perturbation samples
LIME_NUM_SUPERPIXELS = 10    # segments shown in explanation
LIME_NUM_IMAGES      = 20    # images per class

# ─────────────────────────────────────────────
# Counterfactual (gradient-based ACE-style)
# ─────────────────────────────────────────────
# Reference: Singla et al. (2023) – Explaining the black-box smoothly—
# A counterfactual approach. Medical Image Analysis, 84, 102721.
CF_MAX_ITER         = 500    # gradient steps
CF_LEARNING_RATE    = 0.01
CF_LAMBDA_L1        = 1.0    # sparsity weight
CF_LAMBDA_L2        = 0.5    # proximity weight
CF_CONFIDENCE_THRES = 0.6    # target class probability threshold
CF_NUM_IMAGES       = 20     # images per class
CF_PIXEL_THRESHOLD  = 0.05   # threshold for sparsity binary mask

# ─────────────────────────────────────────────
# Evaluation / XAI Metrics
# ─────────────────────────────────────────────
FAITHFULNESS_STEPS = 10      # steps for deletion/insertion curves
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

def _next_experiment_dir(base: Path) -> Path:
    """
    Return the next available experiment directory path.

    Scans base for existing experiment_XX folders and returns
    the next integer-incremented path (experiment_01, experiment_02 …).

    Parameters
    ----------
    base : Path
        Parent results directory.

    Returns
    -------
    Path
        New experiment directory (not yet created).
    """
    existing = sorted(base.glob("experiment_*"))
    if not existing:
        return base / "experiment_01"
    last = existing[-1].name          # e.g. "experiment_03"
    num  = int(last.split("_")[-1])   # 3
    return base / f"experiment_{num + 1:02d}"


# Resolved once per process — all stages share the same experiment folder
EXPERIMENT_DIR = _next_experiment_dir(RESULTS_DIR)

RESULT_EDA            = EXPERIMENT_DIR / "eda"
RESULT_TRAINING       = EXPERIMENT_DIR / "training"
RESULT_EVALUATION     = EXPERIMENT_DIR / "evaluation"
RESULT_GRADCAM        = EXPERIMENT_DIR / "gradcam"
RESULT_LIME           = EXPERIMENT_DIR / "lime"
RESULT_COUNTERFACTUAL = EXPERIMENT_DIR / "counterfactual"
RESULT_COMPARISON     = EXPERIMENT_DIR / "comparison"


def create_result_dirs() -> Path:
    """
    Create the experiment directory and all sub-folders.

    Returns
    -------
    Path
        The experiment root directory (e.g. results/experiment_03/).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for d in [
        RESULT_EDA, RESULT_TRAINING, RESULT_EVALUATION,
        RESULT_GRADCAM, RESULT_LIME, RESULT_COUNTERFACTUAL,
        RESULT_COMPARISON,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Config] Experiment folder: {EXPERIMENT_DIR}")
    return EXPERIMENT_DIR
