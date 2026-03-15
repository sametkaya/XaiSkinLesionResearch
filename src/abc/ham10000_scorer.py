"""
src/abc/ham10000_scorer.py
--------------------------
Score all 10,015 HAM10000 images with ABC (Asymmetry, Border, Color)
values using two independent methods:

  Method 1 — DL Regressor (abc_dl):
    Inference using the trained ABCRegressor model.

  Method 2 — Image Processing (abc_ip):
    Algorithm-based scoring using segmentations + feature extraction
    (asymmetry via principal axis overlap, border via compactness index,
     color via dermoscopic color detection).

Both methods are run on every image.  The outputs include per-image
scores, method agreement statistics, and visualisation of score
distributions per diagnostic class.

Agreement between methods is quantified via:
  - Pearson r
  - Bland-Altman bias and limits of agreement

References
----------
Tschandl, P., Rosendahl, C., & Kittler, H. (2018).
    The HAM10000 dataset. Scientific Data, 5, 180161.

Altman, D. G., & Bland, J. M. (1983).
    Measurement in medicine: the analysis of method comparison studies.
    The Statistician, 32(3), 307–317.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.abc.config_abc import (
    IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
    HAM10000_SCORE_BATCH, HAM10000_SCORE_WORKERS,
    ABC_CRITERIA, ABC_NAMES, RANDOM_SEED,
)
from src.abc.abc_model import ABCRegressor
from src.abc.abc_ip_scorer import ABCImageProcessingScorer
from src.segmentation.segmenter import LesionSegmenter
from src.utils.result_manager import ResultManager


# ─────────────────────────────────────────────
# HAM10000 inference dataset
# ─────────────────────────────────────────────

class HAM10000ScoringDataset(Dataset):
    """
    Minimal Dataset for ABC scoring of HAM10000 images.
    Returns image tensor, image_id, and (optionally) mask array.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dirs: List[Path],
        mask_dir: Optional[Path] = None,
        image_size: int = IMAGE_SIZE,
    ):
        self.df       = metadata_df.reset_index(drop=True)
        self.img_dirs = image_dirs
        self.mask_dir = mask_dir
        self.size     = image_size

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])

    def _find_image(self, image_id: str) -> Optional[Path]:
        for d in self.img_dirs:
            for ext in ("jpg", "jpeg", "png"):
                p = d / f"{image_id}.{ext}"
                if p.exists():
                    return p
        return None

    def _find_mask(self, image_id: str) -> Optional[Path]:
        if self.mask_dir is None:
            return None
        for ext in ("png", "jpg"):
            p = self.mask_dir / f"{image_id}_segmentation.{ext}"
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row      = self.df.iloc[idx]
        img_path = self._find_image(row.image_id)
        if img_path is None:
            img_t = torch.zeros(3, self.size, self.size)
        else:
            img_t = self.tf(Image.open(img_path).convert("RGB"))

        return img_t, row.image_id, row.dx


# ─────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────

class HAM10000Scorer:
    """
    Score all HAM10000 images with ABC criteria using DL + IP methods.

    Parameters
    ----------
    abc_model : ABCRegressor
        Trained ABC regressor for DL scoring.
    metadata_df : pd.DataFrame
        HAM10000 metadata (with image_id, dx, etc.)
    image_dirs : list of Path
        Directories containing HAM10000 images.
    mask_dir : Path or None
        Directory containing ISIC 2018 Task 1 segmentations masks.
    device : torch.device
    result_dir : Path
        Directory to save scores, plots, result.txt.
    unet_weights : Path or None
        Trained U-Net weights for segmentations.
    """

    def __init__(
        self,
        abc_model: ABCRegressor,
        metadata_df: pd.DataFrame,
        image_dirs: List[Path],
        mask_dir: Optional[Path],
        device: torch.device,
        result_dir: Path,
        unet_weights: Optional[Path] = None,
    ):
        self.model       = abc_model
        self.df          = metadata_df
        self.img_dirs    = image_dirs
        self.mask_dir    = mask_dir
        self.device      = device
        self.result_dir  = result_dir

        self.segmenter  = LesionSegmenter(
            model_weights=unet_weights,
            device=device,
            image_size=IMAGE_SIZE,
        )
        self.ip_scorer  = ABCImageProcessingScorer(segmenter=self.segmenter)

    @torch.no_grad()
    def run(self) -> pd.DataFrame:
        """
        Execute full scoring pipeline.

        Returns
        -------
        pd.DataFrame with columns:
            image_id, dx, A_dl, B_dl, C_dl, A_ip, B_ip, C_ip,
            A_mean, B_mean, C_mean, method_agreement_A/B/C
        """
        start = time.time()
        print(f"\n[HAM10000Scorer] Scoring {len(self.df):,} images …")

        dataset = HAM10000ScoringDataset(
            self.df, self.img_dirs, self.mask_dir
        )
        loader  = DataLoader(
            dataset,
            batch_size=HAM10000_SCORE_BATCH,
            shuffle=False,
            num_workers=HAM10000_SCORE_WORKERS,
            pin_memory=True,
        )

        records = []
        self.model.eval()

        for img_tensors, image_ids, dxs in tqdm(loader, desc="Scoring HAM10000"):
            img_tensors = img_tensors.to(self.device)

            # ── Method 1: DL ─────────────────────
            dl_scores = self.model(img_tensors).cpu().numpy()  # (B, 3)

            # ── Method 2: IP ─────────────────────
            for i in range(len(image_ids)):
                img_np = self._tensor_to_numpy(img_tensors[i].cpu())
                mask   = self._load_mask(str(image_ids[i]))
                ip_dict = self.ip_scorer.score(img_np, mask)

                A_dl, B_dl, C_dl = float(dl_scores[i, 0]), float(dl_scores[i, 1]), float(dl_scores[i, 2])
                A_ip = ip_dict["A"]
                B_ip = ip_dict["B"]
                C_ip = ip_dict["C"]

                records.append({
                    "image_id"            : str(image_ids[i]),
                    "dx"                  : str(dxs[i]),
                    "A_dl"                : round(A_dl, 4),
                    "B_dl"                : round(B_dl, 4),
                    "C_dl"                : round(C_dl, 4),
                    "A_ip"                : round(A_ip, 4),
                    "B_ip"                : round(B_ip, 4),
                    "C_ip"                : round(C_ip, 4),
                    "A_mean"              : round((A_dl + A_ip) / 2, 4),
                    "B_mean"              : round((B_dl + B_ip) / 2, 4),
                    "C_mean"              : round((C_dl + C_ip) / 2, 4),
                })

        scores_df = pd.DataFrame(records)
        elapsed   = time.time() - start

        # ── Agreement stats ──────────────────────
        agreement = {}
        for crit in ABC_CRITERIA:
            from scipy.stats import pearsonr
            r, _ = pearsonr(scores_df[f"{crit}_dl"], scores_df[f"{crit}_ip"])
            bias = float((scores_df[f"{crit}_dl"] - scores_df[f"{crit}_ip"]).mean())
            scores_df[f"method_agreement_{crit}"] = (
                1.0 - np.abs(scores_df[f"{crit}_dl"] - scores_df[f"{crit}_ip"])
            ).round(4)
            agreement[crit] = {"pearson_r": round(r, 4), "bias": round(bias, 5)}

        # ── Save CSV ─────────────────────────────
        csv_path = self.result_dir / "ham10000_abc_scores.csv"
        scores_df.to_csv(csv_path, index=False)
        print(f"[HAM10000Scorer] Scores saved: {csv_path}")

        # ── Plots ────────────────────────────────
        self._plot_score_distributions(scores_df)
        self._plot_method_agreement(scores_df)
        self._plot_per_class_abc(scores_df)

        # ── result.txt ───────────────────────────
        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="HAM10000 Pseudo-ABC Scoring",
            conditions={
                "n_images"      : len(scores_df),
                "dl_model"      : "ABCRegressor (EfficientNet-B0)",
                "ip_method"     : "Principal-axis asymmetry + compactness + color detection",
                "segmentations"  : "ISIC 2018 Task 1 masks (Otsu fallback)",
                "criteria"      : "A (Asymmetry), B (Border), C (Color)",
            },
            statistics={
                "elapsed_seconds": round(elapsed, 1),
                "n_scored"       : len(scores_df),
                **{f"mean_A_{k}_{c}": round(float(scores_df[f"{c}_{k}"].mean()), 4)
                   for k in ("dl", "ip", "mean") for c in ("A", "B", "C")},
                **{f"agreement_r_{c}": agreement[c]["pearson_r"]
                   for c in ABC_CRITERIA},
                **{f"agreement_bias_{c}": agreement[c]["bias"]
                   for c in ABC_CRITERIA},
            },
        )

        print(f"[HAM10000Scorer] Done.  {len(scores_df):,} images scored in {elapsed:.1f}s")
        return scores_df

    def _tensor_to_numpy(self, t: torch.Tensor) -> np.ndarray:
        """Denormalise and convert tensor (3,H,W) to uint8 numpy (H,W,3)."""
        mean = np.array(IMAGE_MEAN)
        std  = np.array(IMAGE_STD)
        img  = t.permute(1, 2, 0).numpy()
        img  = img * std + mean
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    def _load_mask(self, image_id: str) -> Optional[np.ndarray]:
        """Load ISIC segmentations mask if available."""
        if self.mask_dir is None:
            return None
        for ext in ("png", "jpg"):
            p = self.mask_dir / f"{image_id}_segmentation.{ext}"
            if p.exists():
                mask = np.array(Image.open(p).convert("L").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST
                ))
                return (mask > 127)
        return None

    def _plot_score_distributions(self, df: pd.DataFrame) -> None:
        """Histograms of DL vs IP scores for each criterion."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for j, method in enumerate(["dl", "ip"]):
            for i, crit in enumerate(ABC_CRITERIA):
                ax  = axes[j, i]
                col = f"{crit}_{method}"
                ax.hist(df[col], bins=30, color=["#3b82f6", "#ef4444"][j],
                        alpha=0.7, edgecolor="white", linewidth=0.4)
                ax.set_xlabel(f"{ABC_NAMES[crit]} Score", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.set_title(
                    f"{crit} — {'DL Regressor' if method=='dl' else 'IP Method'}\n"
                    f"μ={df[col].mean():.3f}  σ={df[col].std():.3f}",
                    fontsize=9,
                )
        plt.suptitle("HAM10000 ABC Score Distributions", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.result_dir / "histograms" / "score_distributions.png", dpi=150)
        plt.close()

    def _plot_method_agreement(self, df: pd.DataFrame) -> None:
        """Scatter plots: DL vs IP scores."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, crit in enumerate(ABC_CRITERIA):
            ax  = axes[i]
            dl  = df[f"{crit}_dl"]
            ip  = df[f"{crit}_ip"]
            ax.scatter(dl, ip, alpha=0.1, s=5, color="#6b7280", rasterized=True)
            ax.plot([0, 1], [0, 1], "r--", lw=1)
            from scipy.stats import pearsonr
            r, _ = pearsonr(dl, ip)
            ax.set_xlabel(f"{crit} DL Score", fontsize=9)
            ax.set_ylabel(f"{crit} IP Score", fontsize=9)
            ax.set_title(f"{crit} — Method Agreement\nr = {r:.4f}", fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
        plt.suptitle("DL Regressor vs Image Processing Agreement", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.result_dir / "scatter" / "method_agreement.png", dpi=150)
        plt.close()

    def _plot_per_class_abc(self, df: pd.DataFrame) -> None:
        """Boxplots of mean ABC scores grouped by HAM10000 diagnostic class."""
        CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, crit in enumerate(ABC_CRITERIA):
            ax   = axes[i]
            data = [df[df.dx == cls][f"{crit}_mean"].dropna().values
                    for cls in CLASS_ORDER]
            bp   = ax.boxplot(data, patch_artist=True, labels=CLASS_ORDER,
                              medianprops={"color": "red", "lw": 2})
            colors = ["#3b82f6","#ef4444","#22c55e","#f59e0b","#8b5cf6","#06b6d4","#e11d48"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_title(f"{crit} — {ABC_NAMES[crit]}\n(mean of DL + IP)", fontsize=9)
            ax.set_ylabel("Score [0,1]", fontsize=9)
            ax.set_xticklabels(CLASS_ORDER, rotation=30, fontsize=8)
        plt.suptitle("ABC Score Distribution by Diagnostic Class", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.result_dir / "per_class_abc.png", dpi=150)
        plt.close()
