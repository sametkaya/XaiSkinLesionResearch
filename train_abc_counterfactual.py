"""
train_abc_counterfactual.py
----------------------------
Script 3 of 3 — ABC-Guided Counterfactual Explanations

The ABC-guided counterfactual loss:
  L = λ_cls · CE(f(x+δ), c_tgt)     ← classification objective
    + λ_A   · |g(x+δ)_A − s_A|       ← asymmetry preservation
    + λ_B   · |g(x+δ)_B − s_B|       ← border preservation
    + λ_C   · |g(x+δ)_C − s_C|       ← color preservation
    + λ_l1  · ‖δ‖₁                    ← pixel sparsity

Ablation study: four modes compared
  baseline — no ABC constraints
  A_only   — asymmetry preservation only
  AB       — asymmetry + border
  ABC      — full ABC-guided (proposed method)

Class-pair transitions evaluated:
  nv → mel,  mel → nv,  bkl → mel,  akiec → bcc

Usage
-----
  # Mevcut run dizinine kaydet (varsayılan):
  python train_abc_counterfactual.py \\
      --ham-checkpoint results/run_01_xai_dermoscopy/training/best_model.pth \\
      --abc-checkpoint results/run_01_xai_dermoscopy/09_abc_regressor/checkpoints/best_abc_model.pth

  # Özel deney dizinine kaydet:
  python train_abc_counterfactual.py \\
      --ham-checkpoint results/run_01_xai_dermoscopy/training/best_model.pth \\
      --abc-checkpoint results/run_01_xai_dermoscopy/09_abc_regressor/checkpoints/best_abc_model.pth \\
      --experiment-dir results/experiment_02_full_mask_counterfactual \\
      --mask-dir datas/HAM10000/segmentations \\
      --n-images 10
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import random

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.abc.config_abc import (
    make_abc_experiment_dir, RESULTS_DIR,
    ABC_CF_NUM_IMAGES, RANDOM_SEED,
    HAM10000_MASK_DIR,
)
from src.abc.abc_model               import ABCRegressor
from src.segmentation.segmenter    import LesionSegmenter
from src.model                       import SkinLesionClassifier, build_model
from src.train                       import load_best_model
from src.data_loader                 import load_metadata, stratified_patient_split, get_dataloaders
from src.explainers.abc_counterfactual import ABCCounterfactualExperiment
from src import config as ham_cfg


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ABC-Guided Counterfactual Explanations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ham-checkpoint", type=str, required=True,
        help="Path to HAM10000 classifier best_model.pth"
    )
    p.add_argument(
        "--abc-checkpoint", type=str, required=True,
        help="Path to ABC regressor best_abc_model.pth"
    )
    p.add_argument(
        "--mask-dir", type=str,
        default=str(HAM10000_MASK_DIR),
        help="Directory containing segmentation masks for δ masking."
    )
    p.add_argument(
        "--experiment-dir", type=str, default=None,
        help="Custom experiment directory (e.g. results/experiment_02_full_mask_cf). "
             "If not set, uses the latest run_XX_xai_dermoscopy/ folder."
    )
    p.add_argument(
        "--n-images", type=int, default=ABC_CF_NUM_IMAGES,
        help="Number of images per class transition pair."
    )
    p.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed."
    )
    return p.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Experiment directory ───────────────────
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        exp_dir = make_abc_experiment_dir(RESULTS_DIR)

    cf_dir = exp_dir / "11_abc_guided_counterfactuals"
    for sub in ["per_class", "ablation", "metrics"]:
        (cf_dir / sub).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ABC-Guided Counterfactual Explanations")
    print(f"  Experiment: {exp_dir}")
    print(f"  Device    : {device}")
    print("=" * 60 + "\n")

    # ── Maske dizini bilgi ─────────────────────
    mask_dir = Path(args.mask_dir)
    if mask_dir.exists():
        n_masks = len(list(mask_dir.glob("*_segmentation.png")))
        print(f"[Main] Mask dir: {mask_dir} ({n_masks:,} masks)")
    else:
        print(f"[Main] Mask dir not found: {mask_dir} — Otsu fallback will be used")

    # ── Load HAM10000 classifier ───────────────
    print("─" * 60)
    print("  Loading HAM10000 classifier …")
    print("─" * 60)
    clf_ckpt = Path(args.ham_checkpoint)
    if not clf_ckpt.exists():
        print(f"[ERROR] Classifier checkpoint not found: {clf_ckpt}")
        sys.exit(1)

    classifier = build_model(device)
    classifier = load_best_model(classifier, clf_ckpt, device)
    classifier.eval()

    # ── Load ABC regressor ─────────────────────
    print("─" * 60)
    print("  Loading ABC regressor …")
    print("─" * 60)
    abc_ckpt = Path(args.abc_checkpoint)
    if not abc_ckpt.exists():
        print(f"[ERROR] ABC checkpoint not found: {abc_ckpt}")
        sys.exit(1)

    state = torch.load(abc_ckpt, map_location=device)
    keys = list(state["model_state_dict"].keys())
    use_sord = any("feat_head" in k or "ordinal_head" in k for k in keys)
    abc_model = ABCRegressor(
        backbone_weights=None,
        freeze_backbone=False,
        num_bins=5 if use_sord else 1,
    )
    abc_model.load_state_dict(state["model_state_dict"])
    print(f"[Main] ABC architecture: {'SORD ordinal' if use_sord else 'Regression'} head")
    abc_model = abc_model.to(device)
    abc_model.eval()
    print(f"[Main] ABC regressor loaded — Val MAE: {state.get('best_val_mae', 'N/A')}")

    # ── Load U-Net segmenter ────────────────────
    print("─" * 60)
    print("  Loading U-Net segmenter …")
    print("─" * 60)
    unet_path = exp_dir / "12_segmentation" / "best_unet.pth"
    segmenter = None
    if unet_path.exists():
        segmenter = LesionSegmenter(model_weights=unet_path, device=device)
        print(f"[Main] U-Net loaded: {unet_path}")
    else:
        print(f"[Main] U-Net not found at {unet_path} — using Otsu fallback")

    # ── HAM10000 test loader ───────────────────
    print("─" * 60)
    print("  Preparing HAM10000 test set …")
    print("─" * 60)
    df = load_metadata()
    train_df, val_df, test_df = stratified_patient_split(df)
    _, _, test_loader = get_dataloaders(train_df, val_df, test_df)

    # ── Run ABC-CF experiment ──────────────────
    print("─" * 60)
    print("  Running ABC-Guided Counterfactual Experiment …")
    print("─" * 60)

    # Override n_images from args
    from src.abc import config_abc
    config_abc.ABC_CF_NUM_IMAGES = args.n_images

    experiment = ABCCounterfactualExperiment(
        classifier    = classifier,
        abc_regressor = abc_model,
        test_loader   = test_loader,
        device        = device,
        result_dir    = cf_dir,
        class_labels  = ham_cfg.CLASS_LABELS,
        segmenter     = segmenter,
    )
    stats = experiment.run()

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print("  Counterfactual Experiment Complete")
    print(f"  Experiment: {exp_dir}")
    print()
    print("  Ablation Summary:")
    print(f"  {'Mode':<12} {'Validity':>9} {'Prox L1':>9} {'ΔA':>8} {'ΔB':>8} {'ΔC':>8}")
    print("  " + "-" * 60)
    for mode in ["baseline", "A_only", "AB", "ABC"]:
        v  = stats.get(f"{mode}_validity", float("nan"))
        l1 = stats.get(f"{mode}_prox_l1",  float("nan"))
        dA = stats.get(f"{mode}_delta_A",   float("nan"))
        dB = stats.get(f"{mode}_delta_B",   float("nan"))
        dC = stats.get(f"{mode}_delta_C",   float("nan"))
        print(
            f"  {mode:<12} {v:>9.4f} {l1:>9.5f} {dA:>8.4f} {dB:>8.4f} {dC:>8.4f}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        torch.multiprocessing.set_start_method("spawn", force=True)
    main()