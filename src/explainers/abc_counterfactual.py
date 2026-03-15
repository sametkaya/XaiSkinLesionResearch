"""
src/explainers/abc_counterfactual.py
-------------------------------------
ABC-Guided Counterfactual Explainer  (v3 — TV regularized)

Extends the standard gradient-based counterfactual approach with
ABC (Asymmetry, Border, Color) preservation constraints and Total
Variation (TV) regularization for spatially coherent perturbations.

Objective
---------
Find δ minimising:

  L = λ_cls · CE(f(x+δ), c_tgt)           ← class change
    + λ_A   · |g(x+δ)_A − s_A|            ← asymmetry preservation
    + λ_B   · |g(x+δ)_B − s_B|            ← border preservation
    + λ_C   · |g(x+δ)_C − s_C|            ← color preservation
    + λ_l1  · ‖δ‖₁                         ← sparsity
    + λ_TV  · TV(δ)                         ← spatial smoothness

v3 Changes (from v2)
--------------------
  - Added Total Variation (TV) regularization:
    TV(δ) = Σ|δ_{i,j} − δ_{i+1,j}| + Σ|δ_{i,j} − δ_{i,j+1}|
    Forces spatially coherent changes instead of per-pixel noise.
    Reference: Rudin, Osher & Fatemi (1992). Physica D.
  - Rebalanced hyperparameters: λ_cls 10→3, lr 0.01→0.003
    → slower convergence, more time for ABC+TV to shape δ
  - Stronger ABC lambdas (A=1.0, B=0.8, C=0.6)
    → more visible ablation differences between modes

Ablation variants
-----------------
  mode='baseline'   — no ABC constraints (λ_A=λ_B=λ_C=0)
  mode='A_only'     — asymmetry constraint only
  mode='AB'         — asymmetry + border constraints
  mode='ABC'        — full ABC constraints (default)

References
----------
Singla, S., Pollack, B., Chen, J., & Batmanghelich, K. (2023).
    Explaining the black-box smoothly—A counterfactual approach.
    Medical Image Analysis, 84, 102721.

Wachter, S., Mittelstadt, B., & Russell, C. (2017).
    Counterfactual explanations without opening the black box.
    Harvard Journal of Law & Technology, 31(2), 841–887.

Rudin, L. I., Osher, S., & Fatemi, E. (1992).
    Nonlinear total variation based noise removal algorithms.
    Physica D, 60(1-4), 259–268.
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
    ABC_CF_LAMBDA_L1, ABC_CF_LAMBDA_TV, ABC_CF_CONFIDENCE_THRES,
    ABC_CF_NUM_IMAGES, ABC_CF_PIXEL_THRESHOLD, ABC_CF_PAIRS,
    IMAGE_MEAN, IMAGE_STD, IMAGE_SIZE, ABC_NAMES,
)
from src.utils.result_manager import ResultManager


# ─────────────────────────────────────────────
# Class name mappings (for textual explanations)
# ─────────────────────────────────────────────

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses (akiec)",
    "bcc"  : "Basal Cell Carcinoma (bcc)",
    "bkl"  : "Benign Keratosis (bkl)",
    "df"   : "Dermatofibroma (df)",
    "mel"  : "Melanoma (mel)",
    "nv"   : "Melanocytic Nevi (nv)",
    "vasc" : "Vascular Lesions (vasc)",
}

CLASS_FULL_NAMES_TR = {
    "akiec": "Aktinik Keratoz",
    "bcc"  : "Bazal Hücreli Karsinom",
    "bkl"  : "Benign Keratoz",
    "df"   : "Dermatofibrom",
    "mel"  : "Melanom",
    "nv"   : "Melanositik Nevüs",
    "vasc" : "Vasküler Lezyon",
}

# ABC score interpretation thresholds
ABC_LEVEL_THRESHOLDS = {
    "low"   : (0.0, 0.33),
    "medium": (0.33, 0.66),
    "high"  : (0.66, 1.0),
}


def _abc_level(score: float) -> str:
    """Interpret a [0,1] ABC score as low/medium/high."""
    if score < 0.33:
        return "low"
    elif score < 0.66:
        return "moderate"
    else:
        return "high"


def _abc_level_tr(score: float) -> str:
    """Turkish interpretation of ABC scores."""
    if score < 0.33:
        return "düşük"
    elif score < 0.66:
        return "orta"
    else:
        return "yüksek"


def _delta_direction(src: float, cf: float) -> str:
    """Direction of change: increased / decreased / unchanged."""
    diff = cf - src
    if abs(diff) < 0.01:
        return "unchanged"
    return "increased" if diff > 0 else "decreased"


def _delta_direction_tr(src: float, cf: float) -> str:
    """Turkish direction of change."""
    diff = cf - src
    if abs(diff) < 0.01:
        return "değişmedi"
    return "arttı" if diff > 0 else "azaldı"


# ─────────────────────────────────────────────
# Textual explanation generators
# ─────────────────────────────────────────────

def generate_textual_explanation(result: Dict, src_name: str, tgt_name: str) -> str:
    """
    Generate a human-readable textual counterfactual explanation (English).

    Example output:
      "This Melanocytic Nevi lesion (predicted with 92% confidence) would
       be reclassified as Melanoma (87% confidence) if:
         • Asymmetry increased from 0.23 (low) to 0.67 (high)   [+0.44]
         • Color Variegation increased from 0.18 (low) to 0.52 (moderate)  [+0.34]
         • Border Irregularity remained approximately unchanged (0.31 → 0.33)
       The counterfactual required changes to 12.3% of pixels (sparsity)
       with a mean perturbation magnitude of 0.0234 (L1)."
    """
    src_full = CLASS_FULL_NAMES.get(src_name, src_name)
    tgt_full = CLASS_FULL_NAMES.get(tgt_name, tgt_name)
    abc_src  = result["abc_src"]
    abc_cf   = result["abc_cf"]
    valid    = result["validity"]

    if valid:
        header = (
            f"This {src_full} lesion would be reclassified as "
            f"{tgt_full} (confidence: {result['final_prob']:.0%}) if:"
        )
    else:
        header = (
            f"The model could NOT confidently reclassify this {src_full} "
            f"lesion as {tgt_full} (best confidence: {result['final_prob']:.0%}). "
            f"The attempted changes were:"
        )

    changes = []
    for crit, full_name in [("A", "Asymmetry"), ("B", "Border Irregularity"), ("C", "Color Variegation")]:
        s = abc_src[crit]
        c = abc_cf[crit]
        d = c - s
        direction = _delta_direction(s, c)

        if direction == "unchanged":
            changes.append(
                f"  • {full_name} remained approximately unchanged "
                f"({s:.2f} → {c:.2f})"
            )
        else:
            changes.append(
                f"  • {full_name} {direction} from {s:.2f} ({_abc_level(s)}) "
                f"to {c:.2f} ({_abc_level(c)})  [{d:+.2f}]"
            )

    footer = (
        f"Perturbation: sparsity={result['sparsity']:.1%} of pixels changed, "
        f"L1 magnitude={result['proximity_l1']:.4f}, "
        f"converged in {result['n_iter']} iterations."
    )

    return "\n".join([header, *changes, footer])


def generate_textual_explanation_tr(result: Dict, src_name: str, tgt_name: str) -> str:
    """
    Turkish textual counterfactual explanation for clinical accessibility.

    Example output:
      "Bu Melanositik Nevüs lezyonu aşağıdaki değişikliklerle
       Melanom olarak sınıflandırılırdı (%87 güven):
         • Asimetri: 0.23'ten (düşük) → 0.67'ye (yüksek) arttı  [+0.44]
         • Renk Çeşitliliği: 0.18'den (düşük) → 0.52'ye (orta) arttı  [+0.34]
         • Sınır Düzensizliği: yaklaşık aynı kaldı (0.31 → 0.33)
       Pertürbasyon: piksellerin %12.3'ü değişti, ortalama L1=0.0234."
    """
    src_full = CLASS_FULL_NAMES_TR.get(src_name, src_name)
    tgt_full = CLASS_FULL_NAMES_TR.get(tgt_name, tgt_name)
    abc_src  = result["abc_src"]
    abc_cf   = result["abc_cf"]
    valid    = result["validity"]

    ABC_NAMES_TR = {
        "A": "Asimetri",
        "B": "Sınır Düzensizliği",
        "C": "Renk Çeşitliliği",
    }

    if valid:
        header = (
            f"Bu {src_full} lezyonu aşağıdaki değişikliklerle "
            f"{tgt_full} olarak sınıflandırılırdı (güven: %{result['final_prob']*100:.0f}):"
        )
    else:
        header = (
            f"Model bu {src_full} lezyonunu {tgt_full} olarak yeniden "
            f"sınıflandıramadı (en iyi güven: %{result['final_prob']*100:.0f}). "
            f"Denenen değişiklikler:"
        )

    changes = []
    for crit in ["A", "B", "C"]:
        s = abc_src[crit]
        c = abc_cf[crit]
        d = c - s
        direction = _delta_direction_tr(s, c)
        name_tr = ABC_NAMES_TR[crit]

        if abs(d) < 0.01:
            changes.append(
                f"  • {name_tr}: yaklaşık aynı kaldı ({s:.2f} → {c:.2f})"
            )
        else:
            changes.append(
                f"  • {name_tr}: {s:.2f}'den ({_abc_level_tr(s)}) → "
                f"{c:.2f}'ye ({_abc_level_tr(c)}) {direction}  [{d:+.2f}]"
            )

    footer = (
        f"Pertürbasyon: piksellerin %{result['sparsity']*100:.1f}'i değişti, "
        f"ortalama L1={result['proximity_l1']:.4f}, "
        f"{result['n_iter']} iterasyonda yakınsadı."
    )

    return "\n".join([header, *changes, footer])


# ─────────────────────────────────────────────
# Total Variation loss
# ─────────────────────────────────────────────

def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    Compute anisotropic Total Variation of a perturbation tensor.

    TV(δ) = Σ|δ_{i,j} − δ_{i+1,j}| + Σ|δ_{i,j} − δ_{i,j+1}|

    This promotes spatially smooth perturbations: adjacent pixels
    should change by similar amounts. Without TV, gradient-based
    counterfactuals tend to produce per-pixel noise that is
    imperceptible but not clinically interpretable.

    Parameters
    ----------
    delta : torch.Tensor  (B, C, H, W) or (1, C, H, W)

    Returns
    -------
    torch.Tensor (scalar)
        Mean TV value across the batch.

    References
    ----------
    Rudin, L. I., Osher, S., & Fatemi, E. (1992).
        Nonlinear total variation based noise removal algorithms.
        Physica D, 60(1-4), 259–268.
    """
    diff_h = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :])
    diff_w = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()


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
    Generate ABC-constrained counterfactual explanations (v3 — TV regularized).

    Uses Adam optimizer with Total Variation regularization for spatially
    coherent perturbations. ABC preservation constraints (via learned
    regressor) enforce morphological fidelity during class transitions.

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
            cf_tensor, delta, validity, final_prob, proximity_l1,
            sparsity, n_iter, abc_src, abc_cf, delta_A, delta_B,
            delta_C, mode, src_prob
        """
        lambdas  = ABLATION_MODES[mode]
        orig     = image_tensor.unsqueeze(0).to(self.device)
        cf       = orig.clone().detach()
        delta    = torch.zeros_like(orig, requires_grad=True, device=self.device)
        target_t = torch.tensor([target_class], dtype=torch.long, device=self.device)

        # Pre-compute source ABC scores and source probability
        with torch.no_grad():
            src_abc  = self.abc_reg(orig).squeeze(0)   # (3,)
            src_logits = self.clf(orig)
            src_probs  = torch.softmax(src_logits, dim=1)
            src_prob   = float(src_probs[0, source_class].item())

        # ── Adam optimizer for δ (Singla et al., 2023) ──
        optimizer = torch.optim.Adam([delta], lr=lr)

        n_iter = 0
        best_prob   = 0.0
        best_delta  = None

        with torch.enable_grad():
            for step in range(max_iter):
                optimizer.zero_grad()
                x_cf = cf + delta

                # Classification loss (push toward target class)
                logits   = self.clf(x_cf)
                probs    = torch.softmax(logits, dim=1)
                loss_cls = F.cross_entropy(logits, target_t) * ABC_CF_LAMBDA_CLS

                # ABC preservation losses
                cf_abc = self.abc_reg(x_cf).squeeze(0)   # (3,)
                loss_A = lambdas["A"] * torch.abs(cf_abc[0] - src_abc[0])
                loss_B = lambdas["B"] * torch.abs(cf_abc[1] - src_abc[1])
                loss_C = lambdas["C"] * torch.abs(cf_abc[2] - src_abc[2])

                # Sparsity (L1 on perturbation)
                loss_l1 = ABC_CF_LAMBDA_L1 * delta.abs().mean()

                # Total Variation — spatial smoothness (v3)
                # Forces perturbation to be locally coherent rather than
                # per-pixel noise.  Essential for clinical interpretability.
                loss_tv = ABC_CF_LAMBDA_TV * total_variation_loss(delta)

                loss = loss_cls + loss_A + loss_B + loss_C + loss_l1 + loss_tv
                loss.backward()
                optimizer.step()

                # Clip to valid normalised pixel range
                with torch.no_grad():
                    x_clamped = torch.clamp(cf + delta, -3.0, 3.0)
                    delta.data.copy_(x_clamped - cf)

                n_iter = step + 1
                cur_prob = float(probs[0, target_class].item())

                # Track best
                if cur_prob > best_prob:
                    best_prob  = cur_prob
                    best_delta = delta.detach().clone()

                if cur_prob >= confidence_threshold:
                    break

        # Use best delta if final didn't reach threshold
        if best_prob > float(probs[0, target_class].item()):
            delta_use = best_delta
        else:
            delta_use = delta.detach()

        # Final counterfactual evaluation
        with torch.no_grad():
            x_cf_final  = (cf + delta_use).squeeze(0).cpu()
            delta_final  = delta_use.squeeze(0).cpu()

            final_logits = self.clf((cf + delta_use))
            final_probs  = torch.softmax(final_logits, dim=1)
            final_pred   = int(final_probs.argmax(dim=1).item())
            final_prob   = float(final_probs[0, target_class].item())

            cf_abc_final = self.abc_reg((cf + delta_use)).squeeze(0).cpu()

        validity = 1 if final_pred == target_class else 0
        prox_l1  = float(delta_final.abs().mean().item())
        sparsity = float(
            (delta_final.abs() > ABC_CF_PIXEL_THRESHOLD).float().mean().item()
        )

        return {
            "cf_tensor"   : x_cf_final,
            "delta"       : delta_final,
            "validity"    : validity,
            "final_prob"  : round(final_prob, 4),
            "src_prob"    : round(src_prob, 4),
            "proximity_l1": round(prox_l1, 5),
            "sparsity"    : round(sparsity, 5),
            "n_iter"      : n_iter,
            "abc_src"     : {
                "A": round(float(src_abc[0]), 4),
                "B": round(float(src_abc[1]), 4),
                "C": round(float(src_abc[2]), 4),
            },
            "abc_cf"      : {
                "A": round(float(cf_abc_final[0]), 4),
                "B": round(float(cf_abc_final[1]), 4),
                "C": round(float(cf_abc_final[2]), 4),
            },
            "delta_A"     : round(abs(float(cf_abc_final[0]) - float(src_abc[0])), 4),
            "delta_B"     : round(abs(float(cf_abc_final[1]) - float(src_abc[1])), 4),
            "delta_C"     : round(abs(float(cf_abc_final[2]) - float(src_abc[2])), 4),
            "mode"        : mode,
        }


# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────

class ABCCounterfactualExperiment:
    """
    Full ABC-guided counterfactual experiment with ablation study.

    v2 additions:
      - Textual counterfactual explanations (EN + TR)
      - Enhanced visual panels with narrative annotations
      - Per-pair narrative summary files
      - Comprehensive result.txt with all statistics and timings

    Runs four ablation modes for each class-pair defined in ABC_CF_PAIRS:
      baseline, A_only, AB, ABC

    Parameters
    ----------
    classifier, abc_regressor, test_loader, device, result_dir, class_labels
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
        start       = time.time()
        all_records = []

        pairs_dir      = self.result_dir / "per_class"
        ablation_dir   = self.result_dir / "ablation"
        metrics_dir    = self.result_dir / "metrics"
        narrative_dir  = self.result_dir / "narratives"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        ablation_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        narrative_dir.mkdir(parents=True, exist_ok=True)

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
                    # Generate textual explanations
                    result["text_en"] = generate_textual_explanation(
                        result, src_name, tgt_name
                    )
                    result["text_tr"] = generate_textual_explanation_tr(
                        result, src_name, tgt_name
                    )
                    mode_records.append(result)
                    all_records.append(result)

                # Save visual panels for first 3 samples
                self._save_panels(
                    mode_records[:3],
                    pairs_dir / f"{src_name}_to_{tgt_name}_{mode}.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                )

                # Save enhanced panels with textual annotations
                self._save_narrative_panels(
                    mode_records[:3],
                    narrative_dir / f"{src_name}_to_{tgt_name}_{mode}_narrative.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                    src_name, tgt_name,
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

        # ── Save all records CSV ───────────────
        self._save_all_records(all_records, metrics_dir / "all_records.csv")

        # ── Save narrative text files ──────────
        self._save_narrative_texts(all_records, narrative_dir)

        # ── Aggregate stats ────────────────────
        stats = self._compute_stats(all_records)
        stats["elapsed_seconds"] = round(elapsed, 1)

        # ── Ablation comparison table ──────────
        self._save_ablation_table(all_records, ablation_dir / "ablation_table.csv")
        self._plot_ablation(all_records, ablation_dir / "ablation_comparison.png")

        # ── ABC delta comparison plot ──────────
        self._plot_abc_delta_comparison(all_records, ablation_dir / "abc_delta_comparison.png")

        # ── result.txt ─────────────────────────
        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="ABC-Guided Counterfactual Explanations (v3 — TV regularized)",
            conditions={
                "classifier"        : "EfficientNet-B4 (HAM10000)",
                "abc_regressor"     : "ABCRegressor (PH2+Derm7pt)",
                "optimizer"         : "Adam (per Singla et al., 2023)",
                "learning_rate"     : ABC_CF_LEARNING_RATE,
                "max_iterations"    : ABC_CF_MAX_ITER,
                "confidence_thres"  : ABC_CF_CONFIDENCE_THRES,
                "loss_weights"      : f"λ_cls={ABC_CF_LAMBDA_CLS}, λ_A={ABC_CF_LAMBDA_A}, "
                                      f"λ_B={ABC_CF_LAMBDA_B}, λ_C={ABC_CF_LAMBDA_C}, "
                                      f"λ_l1={ABC_CF_LAMBDA_L1}, λ_TV={ABC_CF_LAMBDA_TV}",
                "ablation_modes"    : list(ABLATION_MODES.keys()),
                "class_pairs"       : [f"{s}→{t}" for s, t in ABC_CF_PAIRS],
                "n_images_per_pair" : ABC_CF_NUM_IMAGES,
                "pixel_threshold"   : ABC_CF_PIXEL_THRESHOLD,
            },
            statistics=stats,
        )

        print(
            f"\n[ABCCounterfactual] Done.  "
            f"Records: {len(all_records)}  Elapsed: {elapsed:.1f}s"
        )
        return stats

    # ─────────────────────────────────────────
    # Data collection
    # ─────────────────────────────────────────

    def _collect_samples(
        self, src_class: int, n: int
    ) -> List[Tuple[torch.Tensor, int]]:
        """Collect n correctly classified images of src_class."""
        samples = []
        self.explainer.clf.eval()
        with torch.no_grad():
            for images, labels in self.loader:
                imgs  = images.to(self.device)
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

    # ─────────────────────────────────────────
    # Denormalisation
    # ─────────────────────────────────────────

    def _denorm(self, t: torch.Tensor) -> np.ndarray:
        mean = np.array(IMAGE_MEAN)
        std  = np.array(IMAGE_STD)
        img  = t.permute(1, 2, 0).numpy()
        return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────
    # Visual panels (original style)
    # ─────────────────────────────────────────

    def _save_panels(
        self,
        results: List[Dict],
        save_path: Path,
        title: str,
    ) -> None:
        """Save [Original | CF | Diff | |δ| Heatmap] panel for each sample."""
        n = len(results)
        if n == 0:
            return
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, res in enumerate(results):
            orig_np = self._denorm(res["cf_tensor"] - res["delta"])
            cf_np   = self._denorm(res["cf_tensor"])
            diff_np = res["delta"].permute(1, 2, 0).numpy()
            abs_np  = np.abs(diff_np)

            axes[row, 0].imshow(orig_np)
            axes[row, 0].set_title("Original", fontsize=9)
            axes[row, 0].axis("off")

            validity_icon = "✓" if res["validity"] else "✗"
            axes[row, 1].imshow(cf_np)
            axes[row, 1].set_title(
                f"Counterfactual {validity_icon}\n"
                f"P(target)={res['final_prob']:.3f}", fontsize=8
            )
            axes[row, 1].axis("off")

            # Signed difference — use RdBu colormap for clarity
            diff_gray = diff_np.mean(axis=2)
            vmax = max(abs(diff_gray.min()), abs(diff_gray.max()), 1e-6)
            im = axes[row, 2].imshow(diff_gray, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, 2].set_title(
                f"Difference (δ)\n"
                f"ΔA={res['delta_A']:.3f} ΔB={res['delta_B']:.3f} ΔC={res['delta_C']:.3f}",
                fontsize=7,
            )
            axes[row, 2].axis("off")
            plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

            # Absolute perturbation heatmap
            abs_max = abs_np.max() + 1e-8
            axes[row, 3].imshow(
                abs_np.mean(axis=2) / abs_max,
                cmap="inferno",
                vmin=0, vmax=1,
            )
            axes[row, 3].set_title(
                f"|δ| Heatmap\nSparsity={res['sparsity']:.3f}",
                fontsize=8,
            )
            axes[row, 3].axis("off")

        plt.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    # ─────────────────────────────────────────
    # Narrative panels (textual explanations)
    # ─────────────────────────────────────────

    def _save_narrative_panels(
        self,
        results: List[Dict],
        save_path: Path,
        title: str,
        src_name: str,
        tgt_name: str,
    ) -> None:
        """
        Save [Original | CF | |δ| | Textual Explanation] panel.

        The rightmost column contains a human-readable narrative
        explaining what ABC features would need to change for
        the diagnosis to flip.
        """
        n = len(results)
        if n == 0:
            return

        fig, axes = plt.subplots(n, 4, figsize=(22, 5 * n),
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, res in enumerate(results):
            orig_np = self._denorm(res["cf_tensor"] - res["delta"])
            cf_np   = self._denorm(res["cf_tensor"])
            abs_np  = np.abs(res["delta"].permute(1, 2, 0).numpy())

            # Column 0: Original
            axes[row, 0].imshow(orig_np)
            src_full = CLASS_FULL_NAMES.get(src_name, src_name)
            axes[row, 0].set_title(f"Original\n{src_full}", fontsize=8)
            axes[row, 0].axis("off")

            # Column 1: Counterfactual
            axes[row, 1].imshow(cf_np)
            tgt_full = CLASS_FULL_NAMES.get(tgt_name, tgt_name)
            validity_icon = "✓ Valid" if res["validity"] else "✗ Invalid"
            axes[row, 1].set_title(
                f"Counterfactual → {tgt_full}\n"
                f"Conf: {res['final_prob']:.0%}  {validity_icon}",
                fontsize=8,
            )
            axes[row, 1].axis("off")

            # Column 2: Perturbation heatmap
            abs_max = abs_np.max() + 1e-8
            axes[row, 2].imshow(
                abs_np.mean(axis=2) / abs_max,
                cmap="inferno", vmin=0, vmax=1,
            )
            axes[row, 2].set_title(
                f"|δ| Heatmap\n"
                f"Sparsity={res['sparsity']:.1%}, L1={res['proximity_l1']:.4f}",
                fontsize=8,
            )
            axes[row, 2].axis("off")

            # Column 3: Textual explanation
            axes[row, 3].axis("off")
            text = res.get("text_en", "")
            axes[row, 3].text(
                0.05, 0.95, text,
                transform=axes[row, 3].transAxes,
                fontsize=7.5,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          alpha=0.9, edgecolor="gray"),
                wrap=True,
            )
            axes[row, 3].set_title("Textual Counterfactual Explanation", fontsize=8)

        plt.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────
    # Narrative text file output
    # ─────────────────────────────────────────

    def _save_narrative_texts(
        self, records: List[Dict], narrative_dir: Path
    ) -> None:
        """Save all textual explanations to text files (EN + TR)."""
        # Group by pair and mode
        groups = {}
        for r in records:
            key = (r["src_class"], r["tgt_class"], r["mode"])
            groups.setdefault(key, []).append(r)

        # English
        en_lines = [
            "=" * 70,
            "  TEXTUAL COUNTERFACTUAL EXPLANATIONS (English)",
            "=" * 70, "",
        ]
        for (src, tgt, mode), recs in sorted(groups.items()):
            en_lines.append(f"\n{'─' * 60}")
            en_lines.append(f"  Pair: {src} → {tgt}  |  Mode: {mode}")
            en_lines.append(f"{'─' * 60}\n")
            for i, r in enumerate(recs):
                en_lines.append(f"  Sample {i+1}:")
                en_lines.append(f"  {r.get('text_en', 'N/A')}")
                en_lines.append("")

        (narrative_dir / "explanations_en.txt").write_text(
            "\n".join(en_lines), encoding="utf-8"
        )

        # Turkish
        tr_lines = [
            "=" * 70,
            "  METİNSEL KARŞIOLGUSAL AÇIKLAMALAR (Türkçe)",
            "=" * 70, "",
        ]
        for (src, tgt, mode), recs in sorted(groups.items()):
            tr_lines.append(f"\n{'─' * 60}")
            tr_lines.append(f"  Çift: {src} → {tgt}  |  Mod: {mode}")
            tr_lines.append(f"{'─' * 60}\n")
            for i, r in enumerate(recs):
                tr_lines.append(f"  Örnek {i+1}:")
                tr_lines.append(f"  {r.get('text_tr', 'N/A')}")
                tr_lines.append("")

        (narrative_dir / "explanations_tr.txt").write_text(
            "\n".join(tr_lines), encoding="utf-8"
        )

        print(f"[Narratives] Saved EN + TR explanations to {narrative_dir}")

    # ─────────────────────────────────────────
    # All records CSV
    # ─────────────────────────────────────────

    def _save_all_records(self, records: List[Dict], path: Path) -> None:
        """Save all individual records to CSV."""
        rows = []
        for r in records:
            rows.append({
                "src_class"   : r["src_class"],
                "tgt_class"   : r["tgt_class"],
                "mode"        : r["mode"],
                "validity"    : r["validity"],
                "final_prob"  : r["final_prob"],
                "src_prob"    : r.get("src_prob", None),
                "proximity_l1": r["proximity_l1"],
                "sparsity"    : r["sparsity"],
                "n_iter"      : r["n_iter"],
                "A_src"       : r["abc_src"]["A"],
                "B_src"       : r["abc_src"]["B"],
                "C_src"       : r["abc_src"]["C"],
                "A_cf"        : r["abc_cf"]["A"],
                "B_cf"        : r["abc_cf"]["B"],
                "C_cf"        : r["abc_cf"]["C"],
                "delta_A"     : r["delta_A"],
                "delta_B"     : r["delta_B"],
                "delta_C"     : r["delta_C"],
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    # ─────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────

    def _compute_stats(self, records: List[Dict]) -> Dict:
        """Aggregate statistics across all records."""
        if not records:
            return {}
        stats = {}

        # Per-mode stats
        for mode in ABLATION_MODES:
            sub = [r for r in records if r["mode"] == mode]
            if not sub:
                continue
            stats[f"{mode}_validity"]  = round(np.mean([r["validity"]     for r in sub]), 4)
            stats[f"{mode}_prox_l1"]   = round(np.mean([r["proximity_l1"] for r in sub]), 5)
            stats[f"{mode}_sparsity"]  = round(np.mean([r["sparsity"]     for r in sub]), 4)
            stats[f"{mode}_delta_A"]   = round(np.mean([r["delta_A"]      for r in sub]), 4)
            stats[f"{mode}_delta_B"]   = round(np.mean([r["delta_B"]      for r in sub]), 4)
            stats[f"{mode}_delta_C"]   = round(np.mean([r["delta_C"]      for r in sub]), 4)
            stats[f"{mode}_n_iter"]    = round(np.mean([r["n_iter"]       for r in sub]), 1)

        # Per-pair stats
        for src, tgt in ABC_CF_PAIRS:
            sub = [r for r in records if r["src_class"] == src and r["tgt_class"] == tgt]
            if not sub:
                continue
            key = f"{src}_to_{tgt}"
            stats[f"{key}_validity"]   = round(np.mean([r["validity"] for r in sub]), 4)
            stats[f"{key}_best_prob"]  = round(max(r["final_prob"] for r in sub), 4)

        # Global
        stats["total_records"]       = len(records)
        stats["overall_validity"]    = round(np.mean([r["validity"] for r in records]), 4)
        stats["overall_prox_l1"]     = round(np.mean([r["proximity_l1"] for r in records]), 5)
        stats["overall_sparsity"]    = round(np.mean([r["sparsity"] for r in records]), 4)

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
                "validity"   : round(np.mean([r["validity"]     for r in sub]), 4),
                "prox_l1"    : round(np.mean([r["proximity_l1"] for r in sub]), 5),
                "sparsity"   : round(np.mean([r["sparsity"]     for r in sub]), 4),
                "delta_A"    : round(np.mean([r["delta_A"]      for r in sub]), 4),
                "delta_B"    : round(np.mean([r["delta_B"]      for r in sub]), 4),
                "delta_C"    : round(np.mean([r["delta_C"]      for r in sub]), 4),
                "n_iter"     : round(np.mean([r["n_iter"]       for r in sub]), 1),
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    def _plot_ablation(self, records: List[Dict], path: Path) -> None:
        """5-panel bar chart comparing ablation modes."""
        modes   = list(ABLATION_MODES.keys())
        metrics = ["validity", "proximity_l1", "delta_A", "delta_B", "delta_C"]
        labels  = ["Validity Rate", "Proximity (L1)", "ΔA (Asymmetry)", "ΔB (Border)", "ΔC (Color)"]
        colors  = ["#22c55e", "#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4.5))
        for ax, metric, label, color in zip(axes, metrics, labels, colors):
            vals = []
            for mode in modes:
                sub = [r for r in records if r["mode"] == mode]
                vals.append(np.mean([r[metric] for r in sub]) if sub else 0)
            bars = ax.bar(range(len(modes)), vals, color=color, alpha=0.85, edgecolor="white", linewidth=1.2)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=25, fontsize=8)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=7,
                )
            ax.set_ylim(0, max(vals) * 1.25 + 0.01)

        plt.suptitle(
            "ABC Counterfactual Ablation Study\n"
            "Higher validity = more successful class flip  |  "
            "Lower ΔA/ΔB/ΔC with ABC mode = better preservation",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _plot_abc_delta_comparison(self, records: List[Dict], path: Path) -> None:
        """
        Grouped bar chart: ΔA, ΔB, ΔC side by side for each ablation mode.

        This visualises the core hypothesis: ABC constraints reduce
        morphological drift during counterfactual generation.
        """
        modes  = list(ABLATION_MODES.keys())
        crits  = ["delta_A", "delta_B", "delta_C"]
        labels = ["ΔA (Asymmetry)", "ΔB (Border)", "ΔC (Color)"]
        colors = ["#ef4444", "#f59e0b", "#8b5cf6"]

        x      = np.arange(len(modes))
        width  = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (crit, label, color) in enumerate(zip(crits, labels, colors)):
            vals = []
            for mode in modes:
                sub = [r for r in records if r["mode"] == mode]
                vals.append(np.mean([r[crit] for r in sub]) if sub else 0)
            bars = ax.bar(x + i * width, vals, width, label=label,
                          color=color, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", fontsize=7)

        ax.set_xlabel("Ablation Mode", fontsize=11)
        ax.set_ylabel("Mean ABC Delta", fontsize=11)
        ax.set_title(
            "ABC Feature Preservation Across Ablation Modes\n"
            "(Lower = better morphological preservation)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(modes, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()