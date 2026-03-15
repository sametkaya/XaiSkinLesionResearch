"""
train_abc_counterfactual.py
---------------------------
Script 3 of 3 — ABC-Guided Counterfactual Explanations + Ablation Study (v3 — TV regularized)

Generates and evaluates counterfactual explanations for HAM10000 skin
lesion classification using ABC (Asymmetry, Border, Color) preservation
constraints.

The ABC-guided counterfactual loss:
  L = λ_cls · CE(f(x+δ), c_tgt)     ← classification objective
    + λ_A   · |g(x+δ)_A − s_A|       ← asymmetry preservation
    + λ_B   · |g(x+δ)_B − s_B|       ← border preservation
    + λ_C   · |g(x+δ)_C − s_C|       ← color preservation
    + λ_l1  · ‖δ‖₁                    ← pixel sparsity

v3 changes:
  - Adam optimizer (per Singla et al., 2023) replaces raw SGD
  - Fixed hyperparameters: lr, λ_l1, λ_cls (see config_abc.py)
  - Textual counterfactual explanations (EN + TR)
  - Auto-detect backbone architecture from checkpoint

Ablation study: four modes compared
  baseline — no ABC constraints
  A_only   — asymmetry preservation only
  AB       — asymmetry + border
  ABC      — full ABC-guided (proposed method)

Class-pair transitions evaluated:
  nv → mel,  mel → nv,  bkl → mel,  akiec → bcc

Usage
-----
  python train_abc_counterfactual.py \\
      --ham-checkpoint results/run_01_xai_dermoscopy/02_classifier_training/best_model.pth \\
      --abc-checkpoint results/run_01_xai_dermoscopy/09_abc_regressor/checkpoints/best_abc_model.pth \\
      [--n-images 10]

References
----------
Singla, S., Pollack, B., Chen, J., & Batmanghelich, K. (2023).
    Explaining the black-box smoothly—A counterfactual approach.
    Medical Image Analysis, 84, 102721.

Wachter, S., Mittelstadt, B., & Russell, C. (2017).
    Counterfactual explanations without opening the black box.
    Harvard Journal of Law & Technology, 31(2), 841–887.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.abc.config_abc import (
    make_abc_experiment_dir, RESULTS_DIR,
    ABC_CF_NUM_IMAGES, RANDOM_SEED,
)
from src.abc.abc_model               import ABCRegressor
from src.model                       import SkinLesionClassifier
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
# Checkpoint-aware model loading
# ─────────────────────────────────────────────

def _detect_backbone_from_checkpoint(state_dict: dict) -> str:
    """
    Auto-detect which EfficientNet backbone was used from checkpoint keys.

    EfficientNet-B0 final conv BN weight shape: [1280]
    EfficientNet-B4 final conv BN weight shape: [1792]

    We inspect feature_extractor.features.8.1.weight to determine the
    backbone variant.
    """
    key_8_1 = "feature_extractor.features.8.1.weight"
    if key_8_1 in state_dict:
        n_features = state_dict[key_8_1].shape[0]
        if n_features == 1792:
            return "efficientnet_b4"
        elif n_features == 1280:
            return "efficientnet_b0"

    # Fallback: count total parameters to distinguish
    total_params = sum(v.numel() for v in state_dict.values())
    if total_params > 15_000_000:
        return "efficientnet_b4"
    else:
        return "efficientnet_b0"


def load_classifier_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Load HAM10000 classifier with architecture auto-detected from checkpoint.

    This avoids the EfficientNet-B0/B4 mismatch error that occurs when
    config.MODEL_NAME does not match the checkpoint architecture.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Detect backbone from saved weights
    backbone = _detect_backbone_from_checkpoint(state_dict)
    print(f"[Classifier] Auto-detected backbone: {backbone}")

    # Build model with the CORRECT backbone (not from config)
    model = SkinLesionClassifier(
        backbone=backbone,
        num_classes=ham_cfg.NUM_CLASSES,
        pretrained=False,          # loading from checkpoint, not ImageNet
        freeze_backbone=False,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    total = model.get_num_total_params()
    print(
        f"[Classifier] {backbone.upper()} loaded — "
        f"Total params: {total:,} | Device: {device}"
    )
    return model


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ABC-Guided Counterfactual Explanations (v3 — TV regularized)",
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
    exp_dir = make_abc_experiment_dir(RESULTS_DIR)
    cf_dir  = exp_dir / "11_abc_guided_counterfactuals"

    print("\n" + "=" * 60)
    print("  ABC-Guided Counterfactual Explanations (v3 — TV regularized)")
    print(f"  Experiment: {exp_dir.name}")
    print(f"  Output:     {cf_dir}")
    print("=" * 60 + "\n")

    # ── Load HAM10000 classifier ───────────────
    print("─" * 60)
    print("  Loading HAM10000 classifier …")
    print("─" * 60)
    clf_ckpt = Path(args.ham_checkpoint)
    if not clf_ckpt.exists():
        print(f"[ERROR] Classifier checkpoint not found: {clf_ckpt}")
        sys.exit(1)

    classifier = load_classifier_from_checkpoint(clf_ckpt, device)

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

    # ── HAM10000 test loader ───────────────────
    print("─" * 60)
    print("  Preparing HAM10000 test set …")
    print("─" * 60)
    df = load_metadata()
    train_df, val_df, test_df = stratified_patient_split(df)
    _, _, test_loader = get_dataloaders(train_df, val_df, test_df)

    # ── Run ABC-CF experiment ──────────────────
    print("─" * 60)
    print("  Running ABC-Guided Counterfactual Experiment (v3 — TV regularized) …")
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
    )
    stats = experiment.run()

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print("  Counterfactual Experiment Complete (v3 — TV regularized)")
    print(f"  Experiment: {exp_dir.name}")
    print(f"  Elapsed:    {stats.get('elapsed_seconds', 'N/A')}s")
    print(f"  Records:    {stats.get('total_records', 'N/A')}")
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
    print()
    print("  Outputs:")
    print(f"    Panels:     {cf_dir / 'per_class'}")
    print(f"    Narratives: {cf_dir / 'narratives'}")
    print(f"    Ablation:   {cf_dir / 'ablation'}")
    print(f"    Metrics:    {cf_dir / 'metrics'}")
    print(f"    Report:     {cf_dir / 'result.txt'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        torch.multiprocessing.set_start_method("spawn", force=True)
    main()