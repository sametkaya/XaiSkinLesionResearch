"""
src/abc/abc_evaluator.py
------------------------
Evaluation and visualisation for the ABC Regressor.

Outputs produced (all saved to result_dir):
  plots/
    scatter_A.png        — Predicted vs ground-truth scatter (Asymmetry)
    scatter_B.png        — Predicted vs ground-truth scatter (Border)
    scatter_C.png        — Predicted vs ground-truth scatter (Color)
    scatter_all.png      — Combined 3-panel scatter
    bland_altman_A.png   — Bland-Altman agreement plot (A)
    bland_altman_B.png   — Bland-Altman agreement plot (B)
    bland_altman_C.png   — Bland-Altman agreement plot (C)
    training_curves.png  — Loss and MAE training/val curves
    per_class_abc.png    — ABC score distributions per HAM10000 class
    method_agreement.png — DL vs IP method agreement scatter
  evaluation_metrics.csv
  result.txt
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.abc.config_abc import ABC_CRITERIA, ABC_NAMES
from src.abc.abc_model import ABCRegressor
from src.utils.result_manager import ResultManager


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────

CRITERION_COLORS = {"A": "#ef4444", "B": "#3b82f6", "C": "#22c55e"}

def _scatter_plot(
    preds: np.ndarray,
    targets: np.ndarray,
    criterion: str,
    save_path: Path,
    title_suffix: str = "",
) -> None:
    """Scatter plot with regression line and 95% CI."""
    fig, ax = plt.subplots(figsize=(5, 5))
    color = CRITERION_COLORS[criterion]

    ax.scatter(targets, preds, alpha=0.5, s=20, color=color, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect agreement")

    if len(preds) > 2:
        slope, intercept, r, p, se = stats.linregress(targets, preds)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, slope * x_line + intercept, "-", color=color,
                lw=1.5, label=f"Fit (r={r:.3f})")

    mae  = float(np.abs(preds - targets).mean())
    rmse = float(np.sqrt(((preds - targets) ** 2).mean()))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"Ground-truth {ABC_NAMES[criterion]}", fontsize=10)
    ax.set_ylabel(f"Predicted {ABC_NAMES[criterion]}", fontsize=10)
    ax.set_title(
        f"{criterion} — {ABC_NAMES[criterion]}{title_suffix}\n"
        f"MAE={mae:.4f}  RMSE={rmse:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _bland_altman_plot(
    method1: np.ndarray,
    method2: np.ndarray,
    criterion: str,
    label1: str,
    label2: str,
    save_path: Path,
) -> None:
    """Bland-Altman agreement plot between two measurement methods."""
    mean_vals = (method1 + method2) / 2.0
    diff_vals = method1 - method2
    md        = diff_vals.mean()
    sd        = diff_vals.std()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=15,
               color=CRITERION_COLORS[criterion], edgecolors="none")
    ax.axhline(md,          color="red",  lw=1.5, label=f"Bias = {md:.4f}")
    ax.axhline(md + 1.96*sd, color="gray", lw=1, ls="--",
               label=f"+1.96 SD = {md+1.96*sd:.4f}")
    ax.axhline(md - 1.96*sd, color="gray", lw=1, ls="--",
               label=f"−1.96 SD = {md-1.96*sd:.4f}")
    ax.set_xlabel("Mean of both methods", fontsize=10)
    ax.set_ylabel(f"{label1} − {label2}", fontsize=10)
    ax.set_title(
        f"Bland-Altman: {criterion} — {ABC_NAMES[criterion]}\n"
        f"({label1} vs. {label2})",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(history: List[Dict], save_path: Path) -> None:
    """Plot loss and MAE training / validation curves."""
    df  = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df.epoch, df.train_loss, label="Train", color="#3b82f6")
    axes[0].plot(df.epoch, df.val_loss,   label="Val",   color="#ef4444")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].set_title("ABC Regressor — Loss")
    axes[0].legend()

    axes[1].plot(df.epoch, df.train_mae, label="Train", color="#3b82f6")
    axes[1].plot(df.epoch, df.val_mae,   label="Val",   color="#ef4444")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("ABC Regressor — MAE")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_combined_scatter(
    preds: np.ndarray,   # (N, 3)
    targets: np.ndarray, # (N, 3)
    save_path: Path,
) -> None:
    """3-panel scatter plot (A, B, C) in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, crit in enumerate(ABC_CRITERIA):
        ax    = axes[i]
        color = CRITERION_COLORS[crit]
        ax.scatter(targets[:, i], preds[:, i], alpha=0.4, s=15,
                   color=color, edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        if len(preds) > 2:
            r, _ = stats.pearsonr(targets[:, i], preds[:, i])
        else:
            r = float("nan")
        mae  = float(np.abs(preds[:, i] - targets[:, i]).mean())
        ax.set_xlabel(f"Ground-truth {crit}", fontsize=9)
        ax.set_ylabel(f"Predicted {crit}", fontsize=9)
        ax.set_title(f"{crit} — {ABC_NAMES[crit]}\nr={r:.3f}, MAE={mae:.4f}", fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

    plt.suptitle("ABC Regressor — Predicted vs Ground-Truth", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class ABCEvaluator:
    """
    Evaluate ABCRegressor on a test DataLoader and save all outputs.

    Parameters
    ----------
    model : ABCRegressor
    test_loader : DataLoader
    device : torch.device
    result_dir : Path
    """

    def __init__(
        self,
        model: ABCRegressor,
        test_loader: DataLoader,
        device: torch.device,
        result_dir: Path,
    ):
        self.model       = model
        self.loader      = test_loader
        self.device      = device
        self.result_dir  = result_dir
        self.plots_dir   = result_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def predict(self, tta_n: int = 1) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Run inference with optional Test-Time Augmentation (TTA).

        TTA averages predictions over tta_n augmented views, providing
        ~2-3% free performance improvement at inference.
        Reference: Perez et al. (MICCAI Workshop 2018).

        Parameters
        ----------
        tta_n : int
            Number of TTA views (1 = no TTA).

        Returns
        -------
        preds   : np.ndarray (N, 3)
        targets : np.ndarray (N, 3)
        meta    : list of dicts
        """
        from src.abc.color_constancy import build_dermoscopy_transforms
        from src.abc.config_abc import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD

        self.model.eval()
        all_preds, all_targets, all_meta = [], [], []

        # Build TTA transform (augmented)
        tta_tf = build_dermoscopy_transforms(
            IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
            augment=True, color_constancy=True
        ) if tta_n > 1 else None

        for images, _masks, abc_targets, meta in tqdm(self.loader, desc="Evaluating"):
            images = images.to(self.device)

            if tta_n > 1 and tta_tf is not None:
                # Properly denormalize tensors before TTA PIL transforms
                mean = np.array(IMAGE_MEAN)
                std  = np.array(IMAGE_STD)
                tta_preds = []
                for _ in range(tta_n):
                    tta_imgs = []
                    for i in range(images.shape[0]):
                        # Denormalize: reverse ImageNet normalization
                        img_np = images[i].cpu().permute(1,2,0).numpy()
                        img_np = np.clip((img_np * std + mean) * 255, 0, 255).astype(np.uint8)
                        pil    = __import__("PIL.Image", fromlist=["Image"]).Image.fromarray(img_np)
                        tta_imgs.append(tta_tf(pil))
                    tta_batch = torch.stack(tta_imgs).to(self.device)
                    out = self.model(tta_batch)
                    if hasattr(self.model, "use_sord") and self.model.use_sord:
                        from src.abc.ordinal_loss import SORDLoss
                        sl = SORDLoss(num_bins=self.model.num_bins)
                        out = sl.decode(out)
                    tta_preds.append(out.cpu().numpy())
                preds = np.mean(tta_preds, axis=0)
            else:
                out = self.model(images)
                if hasattr(self.model, 'use_sord') and self.model.use_sord:
                    from src.abc.ordinal_loss import SORDLoss
                    sl = SORDLoss(num_bins=self.model.num_bins)
                    out = sl.decode(out)
                preds = out.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(abc_targets.numpy())
            all_meta.extend([
                {k: v[i] if isinstance(v, (list, torch.Tensor)) else v
                 for k, v in meta.items()}
                for i in range(len(preds))
            ])

        return (
            np.vstack(all_preds),
            np.vstack(all_targets),
            all_meta,
        )

    def evaluate(self, history: Optional[List[Dict]] = None) -> Dict:
        """
        Full evaluation: metrics, scatter plots, Bland-Altman, curves.

        Parameters
        ----------
        history : list of epoch dicts (optional, for training curves)

        Returns
        -------
        dict of aggregate metrics
        """
        from src.abc.config_abc import ABC_TTA_N
        preds, targets, meta = self.predict(tta_n=ABC_TTA_N)
        print(f"[ABCEvaluator] TTA views: {ABC_TTA_N}")

        metrics = {}
        for i, crit in enumerate(ABC_CRITERIA):
            p = preds[:, i]
            t = targets[:, i]
            mae   = float(np.abs(p - t).mean())
            rmse  = float(np.sqrt(((p - t) ** 2).mean()))
            r, _  = stats.pearsonr(p, t) if len(p) > 2 else (float("nan"), None)
            from src.abc.abc_trainer import _icc
            icc   = _icc(p, t)
            metrics[f"mae_{crit}"]  = round(mae,  5)
            metrics[f"rmse_{crit}"] = round(rmse, 5)
            metrics[f"r_{crit}"]    = round(r,    5)
            metrics[f"icc_{crit}"]  = round(icc,  5)

        metrics["mae_mean"]  = round(float(np.abs(preds - targets).mean()), 5)
        metrics["rmse_mean"] = round(float(np.sqrt(((preds - targets) ** 2).mean())), 5)

        # ── Plots ────────────────────────────────
        plot_combined_scatter(preds, targets, self.plots_dir / "scatter_all.png")
        for i, crit in enumerate(ABC_CRITERIA):
            _scatter_plot(
                preds[:, i], targets[:, i], crit,
                self.plots_dir / f"scatter_{crit}.png",
            )

        if history:
            plot_training_curves(history, self.plots_dir / "training_curves.png")

        # ── CSV ──────────────────────────────────
        rows = []
        for i, m in enumerate(meta):
            rows.append({
                "image_id"    : m.get("image_id", i),
                "dataset"     : m.get("dataset_source", "?"),
                "pred_A"      : round(float(preds[i, 0]), 4),
                "pred_B"      : round(float(preds[i, 1]), 4),
                "pred_C"      : round(float(preds[i, 2]), 4),
                "true_A"      : round(float(targets[i, 0]), 4),
                "true_B"      : round(float(targets[i, 1]), 4),
                "true_C"      : round(float(targets[i, 2]), 4),
            })
        pd.DataFrame(rows).to_csv(
            self.result_dir / "evaluation_metrics.csv", index=False
        )

        # ── result.txt ───────────────────────────
        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="ABC Regressor — Test Set Evaluation",
            conditions={
                "model"    : "EfficientNet-B0 + ABC Regression Head",
                "criterion": "Huber (Smooth L1)",
                "dataset"  : "PH2 + Derm7pt (combined test set)",
                "criteria" : "A (Asymmetry), B (Border), C (Color)",
            },
            statistics={
                **metrics,
                "n_test_samples": len(preds),
            },
        )

        print(
            f"[ABCEvaluator] MAE — A: {metrics['mae_A']:.4f}  "
            f"B: {metrics['mae_B']:.4f}  C: {metrics['mae_C']:.4f}  "
            f"Mean: {metrics['mae_mean']:.4f}"
        )
        print(
            f"[ABCEvaluator] ICC — A: {metrics['icc_A']:.4f}  "
            f"B: {metrics['icc_B']:.4f}  C: {metrics['icc_C']:.4f}"
        )

        return metrics