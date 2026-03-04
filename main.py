"""
main.py
-------
Full experiment pipeline for:

  "Towards Clinically Interpretable Skin Lesion Diagnosis:
   A Quantitative Comparison of Counterfactual, Grad-CAM, and LIME
   Explanations on HAM10000"

Pipeline stages
---------------
1.  Data loading & patient-aware stratified split
2.  Model training  (EfficientNet-B0 / ResNet-50 fine-tuned on HAM10000)
3.  Classification evaluation (accuracy, balanced acc., F1, AUC, Cohen κ)
4.  Grad-CAM & Grad-CAM++ explanations
5.  LIME explanations
6.  Counterfactual explanations (ACE-style gradient optimisation)
7.  Quantitative XAI comparison
    - Deletion / Insertion faithfulness curves (Grad-CAM, LIME)
    - Counterfactual validity, proximity, sparsity
    - FID between original and counterfactual images
8.  Side-by-side comparison plots & final result.txt

Usage
-----
    python main.py [--skip-training] [--skip-gradcam] [--skip-lime]
                   [--skip-cf] [--skip-compare]

All intermediate outputs are stored to results/result_0X_<name>/.
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch

from src import config
from src.data_loader import load_metadata, set_seed, stratified_patient_split, get_dataloaders
from src.model import build_model, SkinLesionClassifier
from src.train import Trainer, load_best_model
from src.evaluate import Evaluator
from src.explainers.gradcam import GradCAMExperiment, GradCAM, GradCAMPlusPlus, denormalize
from src.explainers.lime_explainer import LIMEExperiment
from src.explainers.counterfactual import CounterfactualExperiment, CounterfactualExplainer
from src.metrics.xai_metrics import (
    compute_faithfulness_metrics,
    compute_cf_metrics,
    deletion_auc,
    insertion_auc,
)
from src.metrics.fid import FIDCalculator
from src.utils.visualization import (
    plot_training_curves,
    plot_xai_comparison_metrics,
    plot_faithfulness_curves,
)
from src.utils.result_manager import ResultManager

# Windows requires 'spawn' start method for CUDA multiprocessing.
# This must be set before any CUDA operations.
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # already set


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Main] Using device: CUDA — {torch.cuda.get_device_name(0)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        print(f"       CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("[Main] CUDA NOT available — running on CPU.")
        print("[Main] >>> Check: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    return device


def collect_gradcam_samples_for_faithfulness(
    model: SkinLesionClassifier,
    test_loader,
    device: torch.device,
    n: int = 20,
):
    """
    Collect (image_tensor, heatmap, pred_class) tuples using Grad-CAM
    for faithfulness evaluation.
    """
    model.eval()
    target_layer = model.get_feature_layer()
    gcam         = GradCAM(model, target_layer, device)

    # NOTE: torch.no_grad() must NOT wrap GradCAM.generate() because
    # Grad-CAM requires gradient computation (backward pass) to produce
    # the class activation maps.  Only the initial label prediction uses
    # no_grad, which is handled inside GradCAM.generate() itself.
    samples = []
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            if len(samples) >= n:
                break
            inp = img.unsqueeze(0)
            heatmap, pred_cls, _conf = gcam.generate(inp)
            samples.append((img.detach().cpu(), heatmap, pred_cls))
        if len(samples) >= n:
            break

    gcam.remove_hooks()
    return samples


def collect_lime_heatmaps_for_faithfulness(
    lime_exp,
    model: SkinLesionClassifier,
    test_loader,
    device: torch.device,
    n: int = 10,
):
    """
    Generate LIME attribution maps (as pixel-level heatmaps) for faithfulness.
    """
    import numpy as np
    from src.explainers.lime_explainer import LIMEExplainer

    lime_obj = LIMEExplainer(model, device)
    model.eval()
    samples  = []

    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            if len(samples) >= n:
                break
            exp, _, pred_cls, _ = lime_obj.explain(
                img,
                num_samples=500,    # reduce for speed
                num_features=config.LIME_NUM_SUPERPIXELS,
            )
            # Convert LIME segment weights to pixel-level heatmap
            local_exp     = exp.local_exp.get(pred_cls, [])
            segments      = exp.segments                     # (H, W) int labels
            heatmap       = np.zeros(segments.shape, dtype=np.float32)
            for seg_id, weight in local_exp:
                heatmap[segments == seg_id] = max(weight, 0.0)

            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            samples.append((img, heatmap, pred_cls))
        if len(samples) >= n:
            break

    return samples


# ─────────────────────────────────────────────
# Stage functions
# ─────────────────────────────────────────────

def stage_train(model, train_loader, val_loader, device):
    """Train model and return history."""
    print("\n" + "─" * 60)
    print("  STAGE 2 — Model Training")
    print("─" * 60)

    trainer = Trainer(
        model, train_loader, val_loader, device,
        result_dir=config.RESULT_TRAINING,
        use_amp=config.USE_AMP,
    )
    history = trainer.train()
    plot_training_curves(history, config.RESULT_TRAINING / "training_curves.png")
    return history


def stage_evaluate(model, test_loader, device):
    """Evaluate the best model on the test set."""
    print("\n" + "─" * 60)
    print("  STAGE 3 — Classification Evaluation")
    print("─" * 60)

    evaluator = Evaluator(model, test_loader, device, config.RESULT_EVALUATION)
    return evaluator.evaluate()


def stage_gradcam(model, test_loader, device):
    """Run Grad-CAM experiment."""
    print("\n" + "─" * 60)
    print("  STAGE 4 — Grad-CAM Explanations")
    print("─" * 60)

    exp = GradCAMExperiment(model, test_loader, device, config.RESULT_GRADCAM)
    return exp.run()


def stage_lime(model, test_loader, device):
    """Run LIME experiment."""
    print("\n" + "─" * 60)
    print("  STAGE 5 — LIME Explanations")
    print("─" * 60)

    exp = LIMEExperiment(model, test_loader, device, config.RESULT_LIME)
    return exp.run()


def stage_counterfactual(model, test_loader, device):
    """Run counterfactual experiment."""
    print("\n" + "─" * 60)
    print("  STAGE 6 — Counterfactual Explanations")
    print("─" * 60)

    exp = CounterfactualExperiment(model, test_loader, device,
                                   config.RESULT_COUNTERFACTUAL)
    return exp.run()


def stage_compare(model, test_loader, device, cf_stats, eval_metrics):
    """
    Compute all quantitative XAI metrics and produce comparison outputs.
    """
    print("\n" + "─" * 60)
    print("  STAGE 7 — Quantitative XAI Comparison")
    print("─" * 60)

    t0 = time.time()

    # ── Faithfulness: Grad-CAM ────────────────────────────────────────────
    print("[Comparison] Computing Grad-CAM faithfulness …")
    gcam_samples = collect_gradcam_samples_for_faithfulness(
        model, test_loader, device, n=20
    )
    gcam_faith = compute_faithfulness_metrics(model, device, gcam_samples, n_steps=10)

    # ── Faithfulness: LIME ────────────────────────────────────────────────
    print("[Comparison] Computing LIME faithfulness (slow) …")
    lime_faith_samples = collect_lime_heatmaps_for_faithfulness(
        None, model, test_loader, device, n=10
    )
    lime_faith = compute_faithfulness_metrics(model, device, lime_faith_samples, n_steps=10)

    # ── FID ───────────────────────────────────────────────────────────────
    print("[Comparison] Computing FID …")
    cf_explainer = CounterfactualExplainer(model, device)
    label_map    = {k: i for i, k in enumerate(config.CLASS_LABELS)}

    orig_tensors = []
    cf_tensors   = []

    # NOTE: torch.no_grad() must NOT wrap cf_explainer.generate() because
    # counterfactual generation requires gradient-based optimization (backward).
    # torch.enable_grad() is used inside generate() for safety.
    model.eval()
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            if len(orig_tensors) >= 50:
                break
            src = int(lbl)
            tgt = (src + 1) % config.NUM_CLASSES
            result = cf_explainer.generate(img, src, tgt)
            orig_tensors.append(img.detach().cpu())
            cf_tensors.append(result["cf_tensor"].detach().cpu())
        if len(orig_tensors) >= 50:
            break

    fid_calc = FIDCalculator(device)
    fid_score = fid_calc.compute(orig_tensors, cf_tensors)
    print(f"[Comparison] FID = {fid_score}")

    # ── Aggregate comparison table ────────────────────────────────────────
    comparison_metrics = {
        "Grad-CAM": {
            "deletion_auc"  : gcam_faith["mean_deletion_auc"],
            "insertion_auc" : gcam_faith["mean_insertion_auc"],
        },
        "LIME": {
            "deletion_auc"  : lime_faith["mean_deletion_auc"],
            "insertion_auc" : lime_faith["mean_insertion_auc"],
        },
        "Counterfactual": {
            "validity_rate"    : cf_stats.get("global_validity_rate", 0.0),
            "mean_proximity_l1": cf_stats.get("global_mean_proximity_l1", 0.0),
            "mean_sparsity"    : cf_stats.get("global_mean_sparsity", 0.0),
        },
    }

    # Faithfulness comparison plot
    del_data = {
        "Grad-CAM": (
            np.linspace(0, 1, 11),
            np.full(11, gcam_faith["mean_deletion_auc"]),
        ),
        "LIME": (
            np.linspace(0, 1, 11),
            np.full(11, lime_faith["mean_deletion_auc"]),
        ),
    }
    ins_data = {
        "Grad-CAM": (
            np.linspace(0, 1, 11),
            np.full(11, gcam_faith["mean_insertion_auc"]),
        ),
        "LIME": (
            np.linspace(0, 1, 11),
            np.full(11, lime_faith["mean_insertion_auc"]),
        ),
    }
    plot_faithfulness_curves(del_data, ins_data,
                             config.RESULT_COMPARISON / "faithfulness_curves.png")

    # XAI metric comparison bar charts
    for method, metrics in comparison_metrics.items():
        plot_xai_comparison_metrics(
            {method: metrics},
            config.RESULT_COMPARISON / f"metrics_{method.lower().replace(' ', '_')}.png",
        )

    elapsed = time.time() - t0

    # ── Final result.txt ──────────────────────────────────────────────────
    full_stats = {
        "classification": eval_metrics,
        "gradcam_faithfulness": gcam_faith,
        "lime_faithfulness": lime_faith,
        "counterfactual": {
            "global_validity_rate"    : cf_stats.get("global_validity_rate"),
            "global_mean_proximity_l1": cf_stats.get("global_mean_proximity_l1"),
            "global_mean_proximity_l2": cf_stats.get("global_mean_proximity_l2"),
            "global_mean_sparsity"    : cf_stats.get("global_mean_sparsity"),
            "global_mean_n_iter"      : cf_stats.get("global_mean_n_iter"),
        },
        "fid_score"     : fid_score,
        "elapsed_seconds": round(elapsed, 2),
    }

    rm = ResultManager(config.RESULT_COMPARISON)
    rm.write_result(
        experiment_name="Quantitative XAI Method Comparison",
        conditions={
            "model"                 : config.MODEL_NAME,
            "xai_methods_compared"  : ["Grad-CAM", "Grad-CAM++", "LIME",
                                       "Counterfactual (ACE)"],
            "faithfulness_steps"    : config.FAITHFULNESS_STEPS,
            "fid_n_samples"         : len(orig_tensors),
            "random_seed"           : config.RANDOM_SEED,
        },
        statistics=full_stats,
    )

    print(
        f"\n[Comparison] Summary:\n"
        f"  Grad-CAM  deletion AUC  : {gcam_faith['mean_deletion_auc']:.4f}\n"
        f"  Grad-CAM  insertion AUC : {gcam_faith['mean_insertion_auc']:.4f}\n"
        f"  LIME      deletion AUC  : {lime_faith['mean_deletion_auc']:.4f}\n"
        f"  LIME      insertion AUC : {lime_faith['mean_insertion_auc']:.4f}\n"
        f"  CF validity rate        : {cf_stats.get('global_validity_rate', 0):.4f}\n"
        f"  CF proximity L1         : {cf_stats.get('global_mean_proximity_l1', 0):.6f}\n"
        f"  CF sparsity             : {cf_stats.get('global_mean_sparsity', 0):.4f}\n"
        f"  FID (orig vs CF)        : {fid_score}\n"
        f"  Elapsed: {elapsed:.1f}s"
    )
    return full_stats


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def _resolve_checkpoint(explicit: str, results_dir) -> "Path | None":
    """
    Find the best checkpoint to load when --skip-training is used.

    Priority:
      1. Explicit path given via --checkpoint flag.
      2. Most recently modified best_model.pth found in any
         results/experiment_*/training/ directory.

    Parameters
    ----------
    explicit : str or None
        Value of the --checkpoint argument.
    results_dir : Path
        Root results directory (config.RESULTS_DIR).

    Returns
    -------
    Path or None
    """
    from pathlib import Path
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            print(f"[Main] Using explicit checkpoint: {p}")
            return p
        else:
            print(f"[Main] ERROR: specified checkpoint not found: {p}")
            return None

    # Search for the most recent checkpoint across all experiment folders
    candidates = sorted(
        results_dir.glob("experiment_*/training/best_model.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        chosen = candidates[0]
        print(f"[Main] Auto-detected checkpoint: {chosen}")
        return chosen

    print("[Main] No checkpoint found in any experiment_*/training/ folder.")
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="XAI Skin Lesion Research Pipeline"
    )
    parser.add_argument("--skip-training",  action="store_true")
    parser.add_argument("--skip-gradcam",   action="store_true")
    parser.add_argument("--skip-lime",      action="store_true")
    parser.add_argument("--skip-cf",        action="store_true")
    parser.add_argument("--skip-compare",   action="store_true")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Path to a specific best_model.pth to load. "
            "If omitted and --skip-training is set, the most recent "
            "checkpoint found in results/experiment_*/training/ is used."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    set_seed(config.RANDOM_SEED)
    config.create_result_dirs()
    device = get_device()

    print("\n" + "=" * 60)
    print("  XAI Skin Lesion Research — Full Pipeline")
    print("  Model : " + config.MODEL_NAME)
    print("=" * 60)

    # ── Stage 1: Data ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STAGE 1 — Data Loading & Splitting")
    print("─" * 60)
    df                         = load_metadata()
    train_df, val_df, test_df  = stratified_patient_split(df)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df
    )

    # ── Stage 2: Training ─────────────────────────────────────────────────
    model = build_model(device)

    if not args.skip_training:
        history = stage_train(model, train_loader, val_loader, device)
        plot_training_curves(history, config.RESULT_TRAINING / "training_curves.png")
        checkpoint = config.RESULT_TRAINING / "best_model.pth"
    else:
        print("[Main] Skipping training.")
        checkpoint = _resolve_checkpoint(args.checkpoint, config.RESULTS_DIR)

    if checkpoint is not None and checkpoint.exists():
        model = load_best_model(model, checkpoint, device)
    else:
        print("[Main] WARNING: No checkpoint found. Using randomly initialised model.")

    # ── Stage 3: Evaluation ───────────────────────────────────────────────
    eval_metrics = stage_evaluate(model, test_loader, device)

    # ── Stage 4: Grad-CAM ─────────────────────────────────────────────────
    if not args.skip_gradcam:
        gcam_stats = stage_gradcam(model, test_loader, device)
    else:
        print("[Main] Skipping Grad-CAM.")
        gcam_stats = {}

    # ── Stage 5: LIME ─────────────────────────────────────────────────────
    if not args.skip_lime:
        lime_stats = stage_lime(model, test_loader, device)
    else:
        print("[Main] Skipping LIME.")
        lime_stats = {}

    # ── Stage 6: Counterfactual ───────────────────────────────────────────
    if not args.skip_cf:
        cf_stats = stage_counterfactual(model, test_loader, device)
    else:
        print("[Main] Skipping Counterfactual.")
        cf_stats = {}

    # ── Stage 7: Comparison ───────────────────────────────────────────────
    if not args.skip_compare:
        compare_stats = stage_compare(model, test_loader, device,
                                      cf_stats, eval_metrics)
    else:
        print("[Main] Skipping comparison.")
        compare_stats = {}

    print("\n" + "=" * 60)
    print("  Pipeline complete.  All results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()