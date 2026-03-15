"""
src/explainers/abc_counterfactual.py
-------------------------------------
ABC-Guided Counterfactual Explainer.

Extends the standard gradient-based counterfactual approach with
ABC (Asymmetry, Border, Color) preservation constraints.

Objective
---------
Given:
  - x:   original image
  - f:   skin lesion classifier (HAM10000 trained)
  - g:   ABC regressor (PH2+Derm7pt trained)
  - s_src: predicted ABC scores for x  (shape 3,)
  - c_tgt: target class index

Find δ (perturbation) minimising:

  L = λ_cls · CrossEntropy(f(x + δ), c_tgt)        ← sınıf değiştirme
    + λ_A   · |g(x+δ)_A − s_src_A|                 ← asimetri koruması
    + λ_B   · |g(x+δ)_B − s_src_B|                 ← sınır koruması
    + λ_C   · |g(x+δ)_C − s_src_C|                 ← renk koruması
    + λ_l1  · ‖δ‖₁                                  ← seyreklik kısıtı

This formulation enables:
  1. Clinically interpretable explanations:
     "What must change (in terms of ABC morphology) for a diagnosis change?"
  2. Attribution of ABC component contributions via ablation.
  3. Comparison with unconstrained counterfactuals.

Ablation variants
-----------------
  mode='baseline'   — no ABC constraints (λ_A=λ_B=λ_C=0)
  mode='A_only'     — asymmetry constraint only
  mode='AB'         — asymmetry + border constraints
  mode='ABC'        — full ABC constraints (default)

References
----------
Singla, S., et al. (2023). Explaining the black-box smoothly—
    A counterfactual approach with relaxed white-box assumption.
    Medical Image Analysis, 84, 102721.

Wachter, S., Mittelstadt, B., & Russell, C. (2017).
    Counterfactual explanations without opening the black box:
    Automated decisions and the GDPR. Harvard Journal of Law & Technology,
    31(2), 841–887.
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.abc.config_abc import (
    ABC_CF_MAX_ITER, ABC_CF_LEARNING_RATE,
    ABC_CF_LAMBDA_CLS, ABC_CF_LAMBDA_A, ABC_CF_LAMBDA_B, ABC_CF_LAMBDA_C,
    ABC_CF_LAMBDA_L1, ABC_CF_CONFIDENCE_THRES,
    ABC_CF_NUM_IMAGES, ABC_CF_PIXEL_THRESHOLD, ABC_CF_PAIRS,
    IMAGE_MEAN, IMAGE_STD, IMAGE_SIZE, ABC_NAMES,
)
from src.utils.result_manager import ResultManager


# ─────────────────────────────────────────────
# Ablation mode → λ values
# ─────────────────────────────────────────────

ABLATION_MODES = {
    "baseline": {"A": 0.0,             "B": 0.0,             "C": 0.0},
    "A_only"  : {"A": ABC_CF_LAMBDA_A, "B": 0.0,             "C": 0.0},
    "AB"      : {"A": ABC_CF_LAMBDA_A, "B": ABC_CF_LAMBDA_B, "C": 0.0},
    "ABC"     : {"A": ABC_CF_LAMBDA_A, "B": ABC_CF_LAMBDA_B, "C": ABC_CF_LAMBDA_C},
}


# ─────────────────────────────────────────────
# Core counterfactual generator
# ─────────────────────────────────────────────

class ABCCounterfactualExplainer:
    """
    Generate ABC-constrained counterfactual explanations.

    Parameters
    ----------
    classifier : nn.Module
        Trained HAM10000 skin lesion classifier.
    abc_regressor : nn.Module
        Trained ABC regressor.
    device : torch.device
    class_labels : list of str
        Ordered list of class names (must match classifier output).
    """

    def __init__(
        self,
        classifier: nn.Module,
        abc_regressor: nn.Module,
        device: torch.device,
        class_labels: List[str],
    ):
        self.clf      = classifier
        self.abc_reg  = abc_regressor
        self.device   = device
        self.labels   = class_labels

        self.clf.eval()
        self.abc_reg.eval()

    def generate(
        self,
        image_tensor: torch.Tensor,
        source_class: int,
        target_class: int,
        mode: str = "ABC",
        max_iter: int = ABC_CF_MAX_ITER,
        lr: float = ABC_CF_LEARNING_RATE,
        confidence_threshold: float = ABC_CF_CONFIDENCE_THRES,
    ) -> Dict:
        """
        Generate a single ABC-guided counterfactual.

        Parameters
        ----------
        image_tensor : torch.Tensor (3, H, W) — normalised
        source_class : int
        target_class : int
        mode : str
            Ablation mode: 'baseline' | 'A_only' | 'AB' | 'ABC'
        max_iter, lr, confidence_threshold

        Returns
        -------
        dict with keys:
            cf_tensor      : torch.Tensor (3, H, W) counterfactual image
            delta          : torch.Tensor (3, H, W) perturbation
            validity       : int   1 if target class achieved
            final_prob     : float probability of target class
            proximity_l1   : float mean absolute pixel change
            sparsity       : float fraction of pixels changed
            n_iter         : int   iterations to convergence
            abc_src        : dict  ABC scores of original
            abc_cf         : dict  ABC scores of counterfactual
            delta_A        : float |A_cf - A_src|
            delta_B        : float |B_cf - B_src|
            delta_C        : float |C_cf - C_src|
            mode           : str   ablation mode used
        """
        lambdas  = ABLATION_MODES[mode]
        orig     = image_tensor.unsqueeze(0).to(self.device)
        cf       = orig.clone().detach()
        delta    = torch.zeros_like(orig, requires_grad=True, device=self.device)
        target_t = torch.tensor([target_class], dtype=torch.long, device=self.device)

        # Pre-compute source ABC scores (no grad needed)
        with torch.no_grad():
            src_abc = self.abc_reg(orig).squeeze(0)   # (3,)

        n_iter = 0
        with torch.enable_grad():
            for step in range(max_iter):
                x_cf   = cf + delta

                # Classification loss
                logits = self.clf(x_cf)
                probs  = torch.softmax(logits, dim=1)
                loss_cls = F.cross_entropy(logits, target_t) * ABC_CF_LAMBDA_CLS

                # ABC preservation losses
                cf_abc   = self.abc_reg(x_cf).squeeze(0)   # (3,)
                loss_A   = lambdas["A"] * torch.abs(cf_abc[0] - src_abc[0])
                loss_B   = lambdas["B"] * torch.abs(cf_abc[1] - src_abc[1])
                loss_C   = lambdas["C"] * torch.abs(cf_abc[2] - src_abc[2])

                # Sparsity
                loss_l1  = ABC_CF_LAMBDA_L1 * delta.abs().mean()

                loss = loss_cls + loss_A + loss_B + loss_C + loss_l1
                loss.backward()

                with torch.no_grad():
                    delta -= lr * delta.grad
                    delta.grad.zero_()

                n_iter = step + 1

                if float(probs[0, target_class].item()) >= confidence_threshold:
                    break

        # Final counterfactual
        with torch.no_grad():
            x_cf_final = (cf + delta).squeeze(0).cpu()
            delta_final = delta.detach().squeeze(0).cpu()

            final_logits = self.clf((cf + delta).detach())
            final_probs  = torch.softmax(final_logits, dim=1)
            final_pred   = int(final_probs.argmax(dim=1).item())
            final_prob   = float(final_probs[0, target_class].item())

            cf_abc_final = self.abc_reg((cf + delta).detach()).squeeze(0).cpu()

        validity    = 1 if final_pred == target_class else 0
        prox_l1     = float(delta_final.abs().mean().item())
        sparsity    = float(
            (delta_final.abs() > ABC_CF_PIXEL_THRESHOLD).float().mean().item()
        )

        return {
            "cf_tensor"  : x_cf_final,
            "delta"      : delta_final,
            "validity"   : validity,
            "final_prob" : round(final_prob, 4),
            "proximity_l1": round(prox_l1, 5),
            "sparsity"   : round(sparsity, 5),
            "n_iter"     : n_iter,
            "abc_src"    : {
                "A": round(float(src_abc[0]), 4),
                "B": round(float(src_abc[1]), 4),
                "C": round(float(src_abc[2]), 4),
            },
            "abc_cf"     : {
                "A": round(float(cf_abc_final[0]), 4),
                "B": round(float(cf_abc_final[1]), 4),
                "C": round(float(cf_abc_final[2]), 4),
            },
            "delta_A"    : round(abs(float(cf_abc_final[0]) - float(src_abc[0])), 4),
            "delta_B"    : round(abs(float(cf_abc_final[1]) - float(src_abc[1])), 4),
            "delta_C"    : round(abs(float(cf_abc_final[2]) - float(src_abc[2])), 4),
            "mode"       : mode,
        }


# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────

class ABCCounterfactualExperiment:
    """
    Full ABC-guided counterfactual experiment with ablation study.

    Runs four ablation modes for each class-pair defined in ABC_CF_PAIRS:
      baseline, A_only, AB, ABC

    Saves per-pair visual panels and aggregate metric tables.

    Parameters
    ----------
    classifier, abc_regressor, test_loader, device, result_dir
    """

    def __init__(
        self,
        classifier: nn.Module,
        abc_regressor: nn.Module,
        test_loader,
        device: torch.device,
        result_dir: Path,
        class_labels: List[str],
    ):
        self.explainer  = ABCCounterfactualExplainer(
            classifier, abc_regressor, device, class_labels
        )
        self.loader     = test_loader
        self.device     = device
        self.result_dir = result_dir
        self.labels     = class_labels
        self.label2idx  = {lbl: i for i, lbl in enumerate(class_labels)}

    def run(self) -> Dict:
        """Execute full experiment and return aggregate statistics."""
        start  = time.time()
        all_records = []

        pairs_dir    = self.result_dir / "per_class"
        ablation_dir = self.result_dir / "ablation"
        pairs_dir.mkdir(exist_ok=True)
        ablation_dir.mkdir(exist_ok=True)

        for src_name, tgt_name in ABC_CF_PAIRS:
            src_idx = self.label2idx.get(src_name)
            tgt_idx = self.label2idx.get(tgt_name)
            if src_idx is None or tgt_idx is None:
                print(f"[ABCCounterfactual] Skipping unknown pair {src_name}→{tgt_name}")
                continue

            print(f"\n[ABCCounterfactual] Pair: {src_name} → {tgt_name}")
            pair_samples = self._collect_samples(src_idx, ABC_CF_NUM_IMAGES)

            for mode in ABLATION_MODES:
                mode_records = []
                for img_t, true_lbl in pair_samples:
                    result = self.explainer.generate(
                        img_t, src_idx, tgt_idx, mode=mode
                    )
                    result.update({
                        "src_class": src_name,
                        "tgt_class": tgt_name,
                    })
                    mode_records.append(result)
                    all_records.append(result)

                # Save visual panels for first 3 samples
                self._save_panels(
                    mode_records[:3],
                    pairs_dir / f"{src_name}_to_{tgt_name}_{mode}.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                )

                # Ablation row
                print(
                    f"  [{mode:8s}]  validity={np.mean([r['validity'] for r in mode_records]):.3f}  "
                    f"L1={np.mean([r['proximity_l1'] for r in mode_records]):.4f}  "
                    f"ΔA={np.mean([r['delta_A'] for r in mode_records]):.4f}  "
                    f"ΔB={np.mean([r['delta_B'] for r in mode_records]):.4f}  "
                    f"ΔC={np.mean([r['delta_C'] for r in mode_records]):.4f}"
                )

        elapsed = time.time() - start

        # ── Aggregate stats ────────────────────
        stats = self._compute_stats(all_records)
        stats["elapsed_seconds"] = round(elapsed, 1)

        # ── Ablation comparison table ──────────
        self._save_ablation_table(all_records, ablation_dir / "ablation_table.csv")
        self._plot_ablation(all_records, ablation_dir / "ablation_comparison.png")

        # ── result.txt ─────────────────────────
        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="ABC-Guided Counterfactual Explanations",
            conditions={
                "classifier"   : "EfficientNet-B0 (HAM10000)",
                "abc_regressor": "ABCRegressor (PH2+Derm7pt)",
                "loss_weights" : f"λ_cls={ABC_CF_LAMBDA_CLS}, λ_A={ABC_CF_LAMBDA_A}, λ_B={ABC_CF_LAMBDA_B}, λ_C={ABC_CF_LAMBDA_C}, λ_l1={ABC_CF_LAMBDA_L1}",
                "max_iter"     : ABC_CF_MAX_ITER,
                "ablation_modes": list(ABLATION_MODES.keys()),
                "class_pairs"  : [f"{s}→{t}" for s, t in ABC_CF_PAIRS],
            },
            statistics=stats,
        )

        print(
            f"\n[ABCCounterfactual] Done.  "
            f"Records: {len(all_records)}  Elapsed: {elapsed:.1f}s"
        )
        return stats

    def _collect_samples(
        self, src_class: int, n: int
    ) -> List[Tuple[torch.Tensor, int]]:
        """Collect n correctly classified images of src_class."""
        samples = []
        self.explainer.clf.eval()
        with torch.no_grad():
            for images, labels in self.loader:
                imgs = images.to(self.device)
                preds = self.explainer.clf(imgs).argmax(dim=1).cpu()
                for i, (img, lbl, prd) in enumerate(
                    zip(images, labels, preds)
                ):
                    if int(lbl) == src_class and int(prd) == src_class:
                        samples.append((img, int(lbl)))
                    if len(samples) >= n:
                        break
                if len(samples) >= n:
                    break
        return samples

    def _denorm(self, t: torch.Tensor) -> np.ndarray:
        mean = np.array(IMAGE_MEAN)
        std  = np.array(IMAGE_STD)
        img  = t.permute(1, 2, 0).numpy()
        return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)

    def _save_panels(
        self,
        results: List[Dict],
        save_path: Path,
        title: str,
    ) -> None:
        """Save [Original | CF | Diff | |δ|] panel for each sample."""
        n = len(results)
        if n == 0:
            return
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, res in enumerate(results):
            orig_np = self._denorm(
                res["cf_tensor"] - res["delta"]
            )
            cf_np   = self._denorm(res["cf_tensor"])
            diff_np = res["delta"].permute(1, 2, 0).numpy()
            abs_np  = np.abs(diff_np)

            axes[row, 0].imshow(orig_np)
            axes[row, 0].set_title("Original", fontsize=8)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(cf_np)
            axes[row, 1].set_title(
                f"Counterfactual\nP={res['final_prob']:.3f}", fontsize=8
            )
            axes[row, 1].axis("off")

            axes[row, 2].imshow(
                (diff_np - diff_np.min()) /
                (diff_np.max() - diff_np.min() + 1e-8)
            )
            axes[row, 2].set_title(
                f"Difference (δ)\nΔA={res['delta_A']:.3f} ΔB={res['delta_B']:.3f} ΔC={res['delta_C']:.3f}",
                fontsize=7,
            )
            axes[row, 2].axis("off")

            axes[row, 3].imshow(
                abs_np / (abs_np.max() + 1e-8),
                cmap="hot",
            )
            axes[row, 3].set_title(
                f"|δ| Heatmap\nSparsity={res['sparsity']:.3f}",
                fontsize=8,
            )
            axes[row, 3].axis("off")

        plt.suptitle(title, fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def _compute_stats(self, records: List[Dict]) -> Dict:
        """Aggregate statistics across all records."""
        if not records:
            return {}
        stats = {}
        for mode in ABLATION_MODES:
            sub = [r for r in records if r["mode"] == mode]
            if not sub:
                continue
            stats[f"{mode}_validity"] = round(np.mean([r["validity"]   for r in sub]), 4)
            stats[f"{mode}_prox_l1"]  = round(np.mean([r["proximity_l1"] for r in sub]), 5)
            stats[f"{mode}_sparsity"] = round(np.mean([r["sparsity"]   for r in sub]), 4)
            stats[f"{mode}_delta_A"]  = round(np.mean([r["delta_A"]    for r in sub]), 4)
            stats[f"{mode}_delta_B"]  = round(np.mean([r["delta_B"]    for r in sub]), 4)
            stats[f"{mode}_delta_C"]  = round(np.mean([r["delta_C"]    for r in sub]), 4)
            stats[f"{mode}_n_iter"]   = round(np.mean([r["n_iter"]     for r in sub]), 1)
        return stats

    def _save_ablation_table(self, records: List[Dict], path: Path) -> None:
        """Save ablation comparison as CSV."""
        rows = []
        for mode in ABLATION_MODES:
            sub = [r for r in records if r["mode"] == mode]
            if not sub:
                continue
            rows.append({
                "mode"       : mode,
                "validity"   : round(np.mean([r["validity"]    for r in sub]), 4),
                "prox_l1"    : round(np.mean([r["proximity_l1"]for r in sub]), 5),
                "sparsity"   : round(np.mean([r["sparsity"]    for r in sub]), 4),
                "delta_A"    : round(np.mean([r["delta_A"]     for r in sub]), 4),
                "delta_B"    : round(np.mean([r["delta_B"]     for r in sub]), 4),
                "delta_C"    : round(np.mean([r["delta_C"]     for r in sub]), 4),
                "n_iter"     : round(np.mean([r["n_iter"]      for r in sub]), 1),
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    def _plot_ablation(self, records: List[Dict], path: Path) -> None:
        """4-panel bar chart comparing ablation modes."""
        modes    = list(ABLATION_MODES.keys())
        metrics  = ["validity", "proximity_l1", "delta_A", "delta_B", "delta_C"]
        labels   = ["Validity", "Prox L1", "ΔA", "ΔB", "ΔC"]
        colors   = ["#22c55e", "#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
        for ax, metric, label, color in zip(axes, metrics, labels, colors):
            vals = []
            for mode in modes:
                sub  = [r for r in records if r["mode"] == mode]
                vals.append(np.mean([r[metric] for r in sub]) if sub else 0)
            bars = ax.bar(modes, vals, color=color, alpha=0.8, edgecolor="white")
            ax.set_title(label, fontsize=10)
            ax.set_xticklabels(modes, rotation=25, fontsize=8)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=7,
                )

        plt.suptitle("ABC Counterfactual Ablation Study", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
