"""
utils/visualization.py
-----------------------
Shared visualisation utilities: EDA plots, training curves, and the
side-by-side XAI method comparison figure.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src import config


# ─────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """
    Bar chart of sample counts per diagnostic class.

    Parameters
    ----------
    df : pd.DataFrame
        Full metadata DataFrame with column 'dx'.
    save_path : Path
        Output file path.
    """
    counts = df["dx"].value_counts()
    labels = [config.CLASS_NAMES.get(k, k) for k in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("tab10", n_colors=len(counts))
    bars = ax.bar(labels, counts.values, color=palette, edgecolor="white", linewidth=0.7)
    ax.bar_label(bars, padding=3, fontsize=9)

    ax.set_xlabel("Diagnostic Category", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("HAM10000 Class Distribution", fontsize=13, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Grouped bar chart showing class distribution across train / val / test splits.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x       = np.arange(config.NUM_CLASSES)
    width   = 0.28

    for i, (split_name, split_df, color) in enumerate([
        ("Train", train_df, "#4575b4"),
        ("Val",   val_df,   "#74add1"),
        ("Test",  test_df,  "#abd9e9"),
    ]):
        counts = [split_df[split_df["dx"] == k].shape[0] for k in config.CLASS_LABELS]
        ax.bar(x + i * width - width, counts, width, label=split_name, color=color,
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [config.CLASS_NAMES[k] for k in config.CLASS_LABELS],
        rotation=25, ha="right"
    )
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution per Split", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sample_images(df: pd.DataFrame, save_path: Path, n_per_class: int = 4) -> None:
    """
    Grid of representative sample images for each class.
    """
    from PIL import Image as PILImage

    fig, axes = plt.subplots(
        config.NUM_CLASSES, n_per_class,
        figsize=(n_per_class * 2.5, config.NUM_CLASSES * 2.5)
    )

    for row_idx, cls in enumerate(config.CLASS_LABELS):
        class_df = df[df["dx"] == cls].sample(
            n=min(n_per_class, len(df[df["dx"] == cls])),
            random_state=config.RANDOM_SEED,
        )
        for col_idx in range(n_per_class):
            ax = axes[row_idx, col_idx]
            if col_idx < len(class_df):
                fp  = class_df.iloc[col_idx]["filepath"]
                img = PILImage.open(str(fp)).convert("RGB")
                img = img.resize((128, 128))
                ax.imshow(img)
            else:
                ax.axis("off")
                continue

            if col_idx == 0:
                ax.set_ylabel(config.CLASS_NAMES[cls], fontsize=8, rotation=90,
                              labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Sample Dermoscopy Images per Class (HAM10000)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_metadata_stats(df: pd.DataFrame, save_path: Path) -> None:
    """
    Pie / bar charts for localization and age distribution in the metadata.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Age distribution
    axes[0].hist(df["age"].dropna(), bins=20, color="#4575b4", edgecolor="white")
    axes[0].set_title("Age Distribution", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")

    # Localisation
    loc_counts = df["localization"].value_counts().head(10)
    axes[1].barh(loc_counts.index, loc_counts.values, color="#74add1")
    axes[1].set_title("Top 10 Lesion Localisations", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Count")

    # Sex
    sex_counts = df["sex"].value_counts()
    axes[2].pie(sex_counts.values, labels=sex_counts.index,
                autopct="%1.1f%%", colors=["#4575b4", "#d73027", "#abdda4"])
    axes[2].set_title("Sex Distribution", fontsize=11, fontweight="bold")

    plt.suptitle("HAM10000 Metadata Statistics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────

def plot_training_curves(history: List[Dict], save_path: Path) -> None:
    """
    Line plots for loss and F1-score over epochs.

    Parameters
    ----------
    history : List[Dict]
        Per-epoch stats from Trainer.train().
    save_path : Path
    """
    epochs     = [r["epoch"]      for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss   = [r["val_loss"]   for r in history]
    train_f1   = [r["train_f1"]   for r in history]
    val_f1     = [r["val_f1"]     for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train", color="#4575b4", lw=2)
    axes[0].plot(epochs, val_loss,   label="Val",   color="#d73027", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_f1, label="Train", color="#4575b4", lw=2)
    axes[1].plot(epochs, val_f1,   label="Val",   color="#d73027", lw=2)
    best_epoch = epochs[int(np.argmax(val_f1))]
    axes[1].axvline(best_epoch, color="grey", linestyle="--", lw=1.2,
                    label=f"Best epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1-Score")
    axes[1].set_title("Training & Validation Macro-F1", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# XAI comparison
# ─────────────────────────────────────────────

def plot_xai_comparison_metrics(
    metrics: Dict[str, Dict],
    save_path: Path,
) -> None:
    """
    Grouped bar chart comparing quantitative XAI metrics across methods.

    Parameters
    ----------
    metrics : Dict[str, Dict]
        Keys are method names ('Grad-CAM', 'LIME', 'Counterfactual').
        Values are dicts of metric_name → value.
    save_path : Path
    """
    method_names = list(metrics.keys())
    all_metric_keys = list(next(iter(metrics.values())).keys())

    fig, axes = plt.subplots(1, len(all_metric_keys), figsize=(5 * len(all_metric_keys), 5))
    if len(all_metric_keys) == 1:
        axes = [axes]

    colors = ["#4575b4", "#d73027", "#1a9850"]

    for ax, metric_key in zip(axes, all_metric_keys):
        vals   = [metrics[m].get(metric_key, 0.0) for m in method_names]
        bars   = ax.bar(method_names, vals, color=colors[:len(method_names)],
                        edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(metric_key.replace("_", " ").title(), fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.25 + 0.01)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Quantitative XAI Method Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_faithfulness_curves(
    deletion_data: Dict[str, tuple],
    insertion_data: Dict[str, tuple],
    save_path: Path,
) -> None:
    """
    Plot deletion and insertion curves for each XAI method.

    Parameters
    ----------
    deletion_data : Dict[str, tuple]
        method → (fractions, mean_confidences)
    insertion_data : Dict[str, tuple]
        method → (fractions, mean_confidences)
    save_path : Path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors    = {"Grad-CAM": "#4575b4", "Grad-CAM++": "#74add1",
                 "LIME": "#d73027", "Counterfactual": "#1a9850"}

    for ax, data_dict, title in [
        (axes[0], deletion_data,  "Deletion Curve (lower AUC = better)"),
        (axes[1], insertion_data, "Insertion Curve (higher AUC = better)"),
    ]:
        for method, (fracs, confs) in data_dict.items():
            color = colors.get(method, "#333333")
            auc_v = float(np.trapz(confs, fracs))
            ax.plot(fracs, confs, label=f"{method} (AUC={auc_v:.3f})",
                    color=color, lw=2)
        ax.set_xlabel("Fraction of Pixels Perturbed", fontsize=11)
        ax.set_ylabel("Model Confidence", fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Faithfulness Curves — XAI Method Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
