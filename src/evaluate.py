"""
evaluate.py
-----------
Classification performance evaluation for HAM10000.

Metrics reported:
  - Per-class Precision, Recall, F1-score (weighted and macro averages)
  - Overall Accuracy
  - Balanced Accuracy (accounts for class imbalance)
  - Cohen's Kappa (inter-rater agreement analogue for multi-class)
  - ROC-AUC (One-vs-Rest, macro-averaged)
  - Confusion matrix (absolute counts and row-normalised)

All figures and numerical results are saved to result_03_evaluation/.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src import config
from src.model import SkinLesionClassifier
from src.utils.result_manager import ResultManager


class Evaluator:
    """
    Evaluate a trained SkinLesionClassifier on a held-out test set.

    Parameters
    ----------
    model : SkinLesionClassifier
    test_loader : DataLoader
    device : torch.device
    result_dir : Path
        Directory where outputs are written.
    """

    def __init__(
        self,
        model: SkinLesionClassifier,
        test_loader: DataLoader,
        device: torch.device,
        result_dir: Path,
    ):
        self.model       = model
        self.test_loader = test_loader
        self.device      = device
        self.result_dir  = result_dir
        self.class_names = [config.CLASS_NAMES[k] for k in config.CLASS_LABELS]

    # ── Inference ─────────────────────────────────────────────────────────

    def _predict(self, tta_n: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on the full test set with Test-Time Augmentation.

        TTA averages predictions over multiple augmented views for ~2-3%
        free performance gain at inference.
        Reference: Perez et al. (MICCAI Workshop 2018).

        Parameters
        ----------
        tta_n : int
            Number of augmented views (default: 5).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (true_labels, predicted_labels, class_probabilities)
        """
        from torchvision import transforms as TF
        tta_transform = TF.Compose([
            TF.RandomHorizontalFlip(p=0.5),
            TF.RandomVerticalFlip(p=0.5),
            TF.RandomRotation(degrees=15),
        ])

        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device, non_blocking=True)

                if tta_n > 1:
                    # Average softmax probabilities over TTA views
                    tta_prob_list = []
                    # Original view
                    logits = self.model(images)
                    tta_prob_list.append(torch.softmax(logits, dim=1))
                    # Augmented views
                    for _ in range(tta_n - 1):
                        aug = torch.stack([tta_transform(img) for img in images])
                        aug = aug.to(self.device)
                        lgt = self.model(aug)
                        tta_prob_list.append(torch.softmax(lgt, dim=1))
                    probs = torch.stack(tta_prob_list).mean(0).cpu().numpy()
                else:
                    logits = self.model(images)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()

                preds = probs.argmax(axis=1)
                all_labels.extend(labels.numpy().tolist())
                all_preds.extend(preds.tolist())
                all_probs.append(probs)

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.vstack(all_probs),
        )

    # ── Plots ──────────────────────────────────────────────────────────────

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Plot and save absolute and row-normalised confusion matrices."""
        cm       = confusion_matrix(y_true, y_pred)
        cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        short_names = config.CLASS_LABELS  # use short dx codes as axis labels

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ["d", ".2f"],
            ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised by Row)"],
        ):
            sns.heatmap(
                data,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=short_names,
                yticklabels=short_names,
                ax=ax,
                linewidths=0.5,
            )
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            ax.set_title(title, fontsize=13, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.result_dir / "confusion_matrix.png", dpi=150)
        plt.close()

    def _plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> None:
        """Plot One-vs-Rest ROC curves for all seven classes."""
        y_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, config.NUM_CLASSES))

        for i, (name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.8,
                    label=f"{name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(self.result_dir / "roc_curves.png", dpi=150)
        plt.close()

    def _plot_per_class_f1(self, report_dict: Dict) -> None:
        """Bar chart of per-class F1-scores."""
        f1_scores = [report_dict[name]["f1-score"] for name in self.class_names]
        colors    = ["#d73027" if f < 0.5 else "#4575b4" for f in f1_scores]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(self.class_names, f1_scores, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="0.5 threshold")
        ax.set_xlabel("Diagnostic Category", fontsize=12)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_title("Per-class F1-Scores on Test Set", fontsize=13, fontweight="bold")
        ax.legend()
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(self.result_dir / "per_class_f1.png", dpi=150)
        plt.close()

    # ── Main entry-point ───────────────────────────────────────────────────

    def evaluate(self) -> Dict:
        """
        Run full evaluation and write all outputs to result_dir.

        Returns
        -------
        Dict
            Dictionary of all scalar metrics.
        """
        start = time.time()
        y_true, y_pred, y_probs = self._predict()
        elapsed = time.time() - start

        # ── Scalar metrics ────────────────────────────────────────────────
        acc          = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        kappa        = cohen_kappa_score(y_true, y_pred)
        macro_f1     = float(pd.DataFrame(
            classification_report(y_true, y_pred,
                                  target_names=self.class_names,
                                  output_dict=True)
        ).T.loc["macro avg", "f1-score"])

        y_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
        macro_auc = roc_auc_score(y_bin, y_probs, average="macro",
                                  multi_class="ovr")

        report_dict = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
        )
        report_str = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
        )

        metrics = {
            "accuracy"        : round(acc,          4),
            "balanced_accuracy": round(balanced_acc, 4),
            "macro_f1"        : round(macro_f1,      4),
            "macro_auc"       : round(macro_auc,     4),
            "cohen_kappa"     : round(kappa,         4),
            "inference_time_s": round(elapsed,       3),
            "n_test_samples"  : len(y_true),
        }

        # ── Figures ───────────────────────────────────────────────────────
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curves(y_true, y_probs)
        self._plot_per_class_f1(report_dict)

        # ── result.txt ────────────────────────────────────────────────────
        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="Classification Evaluation",
            conditions={
                "model"           : config.MODEL_NAME,
                "test_samples"    : len(y_true),
                "image_size"      : config.IMAGE_SIZE,
                "batch_size"      : config.BATCH_SIZE,
                "pretrained"      : config.PRETRAINED,
                "random_seed"     : config.RANDOM_SEED,
            },
            statistics={
                **metrics,
                "classification_report": "\n" + report_str,
            },
        )

        print(f"\n[Evaluator] Accuracy: {acc:.4f} | Balanced Acc: {balanced_acc:.4f} "
              f"| Macro F1: {macro_f1:.4f} | Macro AUC: {macro_auc:.4f} "
              f"| Cohen κ: {kappa:.4f}")
        print(f"[Evaluator] Results saved to {self.result_dir}")

        return metrics