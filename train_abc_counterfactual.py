"""
train_abc_counterfactual.py
---------------------------
Script 3 of 3 — ABC-Guided Counterfactual Explanations + Ablation Study

Generates and evaluates counterfactual explanations for HAM10000 skin
lesion classification using ABC (Asymmetry, Border, Color) preservation
constraints.

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

Outputs (auto-incremented experiment directory):
  results/abc_experiment_XX/
    abc_cf/
      per_class/
        nv_to_mel_baseline.png
        nv_to_mel_A_only.png
        nv_to_mel_AB.png
        nv_to_mel_ABC.png
        …
      ablation/
        ablation_table.csv
        ablation_comparison.png
      metrics/
        all_records.csv
      result.txt

Usage
-----
  python train_abc_counterfactual.py \\
      --ham-checkpoint results/experiment_01/training/best_model.pth \\
      --abc-checkpoint results/abc_experiment_01/abc_regressor/checkpoints/best_abc_model.pth \\
      [--n-images 10]
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
)
from src.abc.abc_model               import ABCRegressor
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
    cf_dir  = exp_dir / "abc_cf"

    print("\n" + "=" * 60)
    print("  ABC-Guided Counterfactual Explanations")
    print(f"  Experiment: {exp_dir.name}")
    print("=" * 60 + "\n")

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
    )
    stats = experiment.run()

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print("  Counterfactual Experiment Complete")
    print(f"  Experiment: {exp_dir.name}")
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