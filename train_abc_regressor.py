"""
train_abc_regressor.py
----------------------
Script 1 of 3 — ABC Regressor Training

Trains an EfficientNet-B0 based multi-output regression model to predict
ABC (Asymmetry, Border irregularity, Color variegation) scores from
dermoscopic images.

Training data: Derm7pt (2,000 images) [+ PH2 (200 images) if available]
Backbone initialisation: HAM10000 pretrained weights (if available)
Two-phase strategy:
  Phase 1: Freeze backbone, train regression head only (ABC_FREEZE_EPOCHS)
  Phase 2: Unfreeze backbone, full fine-tuning at lower LR

Outputs (auto-incremented experiment directory):
  results/abc_experiment_XX/
    abc_regressor/
      checkpoints/
        best_abc_model.pth
      plots/
        training_curves.png
        scatter_all.png
        scatter_A.png / scatter_B.png / scatter_C.png
      training_log.csv
      evaluation_metrics.csv
      result.txt

Usage
-----
  python train_abc_regressor.py [--ham-checkpoint PATH] [--no-amp]

Arguments
---------
  --ham-checkpoint PATH   Path to HAM10000 best_model.pth for backbone init.
                          If omitted, ImageNet weights are used.
  --no-amp                Disable Automatic Mixed Precision.
  --ph2-dir PATH          Override default PH2 dataset directory.
                          If omitted or missing PH2_dataset.txt, PH2 is skipped.
  --derm7pt-dir PATH      Override default Derm7pt dataset directory.
  --epochs INT            Override number of training epochs.
  --seed INT              Random seed (default: 42).

Academic integrity note
-----------------------
All results are reproducible given the same random seed and dataset splits.
No test labels are accessed during training or validation.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
import random

# ── make project root importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.abc.config_abc import (
    make_abc_experiment_dir, RESULTS_DIR,
    PH2_DIR, DERM7PT_DIR, ABC_NUM_EPOCHS, RANDOM_SEED,
)
from src.abc.ph2_loader    import load_ph2
from src.abc.derm7pt_loader import load_derm7pt
from src.abc.abc_model     import build_abc_regressor
from src.abc.abc_trainer   import ABCTrainer, build_combined_loaders
from src.abc.abc_evaluator import ABCEvaluator


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ABC Regressor on PH2 + Derm7pt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ham-checkpoint", type=str, default=None,
        help="Path to HAM10000 best_model.pth for backbone transfer."
    )
    p.add_argument(
        "--no-amp", action="store_true",
        help="Disable Automatic Mixed Precision."
    )
    p.add_argument(
        "--ph2-dir", type=str, default=str(PH2_DIR),
        help="PH2 dataset root directory."
    )
    p.add_argument(
        "--derm7pt-dir", type=str, default=str(DERM7PT_DIR),
        help="Derm7pt dataset root directory."
    )
    p.add_argument(
        "--epochs", type=int, default=ABC_NUM_EPOCHS,
        help="Maximum number of training epochs."
    )
    p.add_argument(
        "--skip-ph2", action="store_true",
        help="Skip PH2 dataset even if available (train on Derm7pt only)."
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

    # ── Device ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(
            f"[Main] Device: CUDA — {torch.cuda.get_device_name(0)}\n"
            f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("[Main] Device: CPU")

    # ── Experiment directory ───────────────────
    exp_dir = make_abc_experiment_dir(RESULTS_DIR)
    reg_dir = exp_dir / "09_abc_regressor"

    # ── Print banner ──────────────────────────
    print("\n" + "=" * 60)
    print("  XAI Skin Lesion — ABC Regressor Training")
    print("  Backbone: EfficientNet-B0")
    print("  Training data: PH2 + Derm7pt")
    print("=" * 60 + "\n")

    # ── Load datasets ─────────────────────────
    ph2_dir    = Path(args.ph2_dir)
    d7_dir     = Path(args.derm7pt_dir)

    print("─" * 60)
    print("  Loading Derm7pt dataset …")
    print("─" * 60)
    d7_train, d7_val, d7_test = load_derm7pt(d7_dir)

    # ── PH2 (optional) ────────────────────────
    ph2_available = (
        not args.skip_ph2
        and ph2_dir.exists()
        and (ph2_dir / "PH2_dataset.txt").exists()
    )

    if ph2_available:
        print("─" * 60)
        print("  Loading PH2 dataset …")
        print("─" * 60)
        ph2_train, ph2_val, ph2_test = load_ph2(ph2_dir)
        print(f"[Main] Training data: Derm7pt + PH2 combined")
    else:
        ph2_train = ph2_val = ph2_test = None
        reason = "--skip-ph2 flag set" if args.skip_ph2 else "PH2_dataset.txt not found"
        print(f"[Main] PH2 skipped ({reason}) — training on Derm7pt only")

    train_loader, val_loader, test_loader = build_combined_loaders(
        ph2_train, ph2_val, ph2_test,
        d7_train,  d7_val,  d7_test,
        seed=args.seed,
    )

    # ── Build model ───────────────────────────
    print("─" * 60)
    print("  Building ABC Regressor …")
    print("─" * 60)
    ham_ckpt = Path(args.ham_checkpoint) if args.ham_checkpoint else None
    model    = build_abc_regressor(
        device=device,
        ham_checkpoint=ham_ckpt,
        freeze_backbone=True,    # Phase 1 starts with frozen backbone
    )

    # ── Train ─────────────────────────────────
    print("─" * 60)
    print("  Training …")
    print("─" * 60)
    use_amp = not args.no_amp

    trainer = ABCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        result_dir=reg_dir,
        use_amp=use_amp,
    )
    history = trainer.train(num_epochs=args.epochs)

    # ── Evaluate on test set ───────────────────
    print("─" * 60)
    print("  Evaluating on test set …")
    print("─" * 60)
    evaluator = ABCEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        result_dir=reg_dir,
    )
    metrics = evaluator.evaluate(history=history)

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete")
    print(f"  Experiment: {exp_dir.name}")
    print(f"  Best model: {reg_dir / 'checkpoints' / 'best_abc_model.pth'}")
    print(f"  Test MAE (mean): {metrics['mae_mean']:.4f}")
    print(f"  Test ICC  — A: {metrics['icc_A']:.4f}  "
          f"B: {metrics['icc_B']:.4f}  C: {metrics['icc_C']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if sys.platform == "win32" or "microsoft" in open("/proc/version").read().lower() \
            if Path("/proc/version").exists() else False:
        torch.multiprocessing.set_start_method("spawn", force=True)
    main()