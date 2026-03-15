"""
explainers/counterfactual.py
-----------------------------
Gradient-based Adversarial Counterfactual Explanations (ACE) for dermoscopy.

A counterfactual explanation answers the question:
  "What minimal change to this image would cause the model to predict
   a different (target) class?"

This implementation follows the ACE framework (Singla et al., 2023), which
optimises directly in pixel space using a composite loss:

    L = L_class + λ_L1 * ||δ||_1 + λ_L2 * ||δ||_2

where:
  - L_class  : cross-entropy pushing the perturbed image toward target_class
  - λ_L1 * ||δ||_1 : promotes sparsity (few pixels changed)
  - λ_L2 * ||δ||_2 : promotes proximity (small magnitude of change)
  - δ = x_cf − x_orig : the perturbation map

The perturbation is clipped after each gradient step to enforce [0,1] bounds
in normalised space.

Metrics computed per counterfactual:
  - validity    : 1 if target class is the top prediction, else 0
  - proximity_l1: mean absolute pixel difference
  - proximity_l2: mean squared pixel difference
  - sparsity    : fraction of pixels where |δ| > threshold

References:
    Singla, S., Pollack, B., Chen, J., & Batmanghelich, K. (2023).
    Explaining the black-box smoothly—A counterfactual approach.
    Medical Image Analysis, 84, 102721.
    https://doi.org/10.1016/j.media.2022.102721

    Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual
    explanations without opening the black box: Automated decisions and
    the GDPR. Harvard Journal of Law & Technology, 31(2).
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src import config
from src.model import SkinLesionClassifier
from src.explainers.gradcam import denormalize
from src.utils.result_manager import ResultManager


class CounterfactualExplainer:
    """
    ACE-style gradient-based counterfactual generator for image classifiers.

    Parameters
    ----------
    model : SkinLesionClassifier
    device : torch.device
    max_iter : int
        Number of optimisation steps.
    lr : float
        Step size for the perturbation update.
    lambda_l1 : float
        Weight for the L1 sparsity term.
    lambda_l2 : float
        Weight for the L2 proximity term.
    confidence_threshold : float
        Minimum probability the target class must reach for early stopping.
    pixel_threshold : float
        |δ| above this value is counted as "changed" for sparsity metric.
    """

    def __init__(
        self,
        model: SkinLesionClassifier,
        device: torch.device,
        max_iter: int        = config.CF_MAX_ITER,
        lr: float            = config.CF_LEARNING_RATE,
        lambda_l1: float     = config.CF_LAMBDA_L1,
        lambda_l2: float     = config.CF_LAMBDA_L2,
        confidence_threshold: float = config.CF_CONFIDENCE_THRES,
        pixel_threshold: float      = config.CF_PIXEL_THRESHOLD,
    ):
        self.model                = model
        self.device               = device
        self.max_iter             = max_iter
        self.lr                   = lr
        self.lambda_l1            = lambda_l1
        self.lambda_l2            = lambda_l2
        self.confidence_threshold = confidence_threshold
        self.pixel_threshold      = pixel_threshold

    def generate(
        self,
        image_tensor: torch.Tensor,
        source_class: int,
        target_class: int,
    ) -> Dict:
        """
        Generate a counterfactual image for a single input.

        The source image is perturbed iteratively; at each step the gradient
        of the composite loss with respect to the perturbation δ is computed
        and δ is updated.  The perturbation is projected onto the ε-ball of
        the original image after each step.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Single image tensor (3, H, W) on CPU.
        source_class : int
            True / predicted class of the original image.
        target_class : int
            Desired class for the counterfactual.

        Returns
        -------
        Dict with keys:
            cf_tensor      : torch.Tensor – counterfactual image (3, H, W)
            delta          : torch.Tensor – perturbation map    (3, H, W)
            validity       : int   – 1 if CF achieves target class
            final_prob     : float – probability of target class in CF
            proximity_l1   : float
            proximity_l2   : float
            sparsity       : float
            n_iter         : int   – iterations until convergence
        """
        self.model.eval()

        # Move to device; keep original for loss computation
        orig = image_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        cf   = orig.clone().detach().requires_grad_(False)

        # Learnable perturbation initialised to zero
        delta = torch.zeros_like(orig, requires_grad=True, device=self.device)

        target_t = torch.tensor([target_class], dtype=torch.long, device=self.device)
        n_iter   = 0

        with torch.enable_grad():
            for step in range(self.max_iter):
                # Perturbed image (clipped to valid normalised range)
                x_cf   = cf + delta
                logits = self.model(x_cf)
                probs  = torch.softmax(logits, dim=1)

                # Classification loss toward target class
                loss_cls = F.cross_entropy(logits, target_t)

                # Regularisation
                loss_l1 = self.lambda_l1 * delta.abs().mean()
                loss_l2 = self.lambda_l2 * (delta ** 2).mean()

                loss = loss_cls + loss_l1 + loss_l2

                # Manual gradient computation
                loss.backward()

                with torch.no_grad():
                    delta -= self.lr * delta.grad
                    delta.grad.zero_()

                n_iter = step + 1

                # Early stopping: target probability reached
                if float(probs[0, target_class].item()) >= self.confidence_threshold:
                    break

        # Final counterfactual
        with torch.no_grad():
            x_cf_final = (cf + delta).squeeze(0).cpu()
            delta_final = delta.detach().squeeze(0).cpu()

            # Final prediction on counterfactual
            final_logits = self.model((cf + delta).detach())
            final_probs  = torch.softmax(final_logits, dim=1)
            final_prob   = float(final_probs[0, target_class].item())
            final_pred   = int(final_probs.argmax(dim=1).item())

        # Metrics
        validity     = 1 if final_pred == target_class else 0
        proximity_l1 = float(delta_final.abs().mean().item())
        proximity_l2 = float((delta_final ** 2).mean().item())
        sparsity     = float(
            (delta_final.abs() > self.pixel_threshold).float().mean().item()
        )

        return {
            "cf_tensor"   : x_cf_final,
            "delta"       : delta_final,
            "validity"    : validity,
            "final_prob"  : round(final_prob,   4),
            "proximity_l1": round(proximity_l1, 6),
            "proximity_l2": round(proximity_l2, 6),
            "sparsity"    : round(sparsity,     4),
            "n_iter"      : n_iter,
        }


# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────

class CounterfactualExperiment:
    """
    Run ACE-style counterfactual explanations on HAM10000 test images.

    For each source class, a small number of images are taken and
    counterfactuals are generated toward the most clinically relevant
    target class (e.g., nv → mel, mel → nv).

    Parameters
    ----------
    model : SkinLesionClassifier
    test_loader : DataLoader
    device : torch.device
    result_dir : Path
    """

    # Clinically motivated class-transition pairs:
    # Source → Target
    CF_PAIRS: List[Tuple[str, str]] = [
        ("nv",    "mel"),   # Melanocytic Nevi → Melanoma
        ("mel",   "nv"),    # Melanoma → Nevi
        ("bkl",   "mel"),   # Benign Keratosis → Melanoma
        ("akiec", "bcc"),   # Actinic Keratoses → Basal Cell Carcinoma
    ]

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
        self.cf_exp      = CounterfactualExplainer(model, device)
        self.label_map   = {k: i for i, k in enumerate(config.CLASS_LABELS)}

    def _collect_for_class(
        self,
        source_class_idx: int,
        n: int = 3,
    ) -> List[torch.Tensor]:
        """Collect n correctly classified images for a given class index."""
        self.model.eval()
        collected = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                preds  = self.model(images).argmax(dim=1).cpu()
                for img, lbl, prd in zip(images.cpu(), labels, preds):
                    if int(lbl) == source_class_idx and int(prd) == source_class_idx:
                        collected.append(img)
                        if len(collected) >= n:
                            return collected
        return collected

    def _make_panel(
        self,
        orig_tensor: torch.Tensor,
        cf_result: Dict,
        source_name: str,
        target_name: str,
        save_path: Path,
    ) -> None:
        """Save a 4-panel visualisation for one counterfactual."""
        orig_np  = denormalize(orig_tensor)
        cf_np    = denormalize(cf_result["cf_tensor"])
        delta_np = cf_result["delta"].permute(1, 2, 0).numpy()

        # Perturbation map: amplify for visibility
        delta_vis = np.abs(delta_np)
        delta_vis = (delta_vis - delta_vis.min()) / (delta_vis.max() - delta_vis.min() + 1e-8)
        delta_vis = (delta_vis * 255).astype(np.uint8)

        # Difference map (signed, single channel)
        diff_gray = delta_np.mean(axis=2)
        vmax      = max(abs(diff_gray.min()), abs(diff_gray.max()))

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        axes[0].imshow(orig_np)
        axes[0].set_title(f"Original\nClass: {source_name}", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(cf_np)
        validity_str = "✓ Valid" if cf_result["validity"] else "✗ Invalid"
        axes[1].set_title(
            f"Counterfactual → {target_name}\n"
            f"Conf: {cf_result['final_prob']:.2f}  {validity_str}",
            fontsize=9
        )
        axes[1].axis("off")

        im = axes[2].imshow(diff_gray, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[2].set_title(
            f"Signed Difference Map\n"
            f"L1={cf_result['proximity_l1']:.4f}  L2={cf_result['proximity_l2']:.4f}",
            fontsize=9
        )
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(delta_vis)
        axes[3].set_title(
            f"|Perturbation| (amplified)\nSparsity={cf_result['sparsity']:.4f}",
            fontsize=9
        )
        axes[3].axis("off")

        plt.suptitle(
            f"Counterfactual Explanation: {config.CLASS_NAMES[source_name]} → "
            f"{config.CLASS_NAMES[target_name]}",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

    def run(self) -> Dict:
        """
        Execute the full counterfactual experiment for all CF_PAIRS.

        Returns
        -------
        Dict
            Aggregated statistics per transition pair.
        """
        start            = time.time()
        all_stats: Dict  = {}
        valid_counts     = {}
        proximity_l1_all = []
        proximity_l2_all = []
        sparsity_all     = []
        n_iter_all       = []

        for (src_name, tgt_name) in self.CF_PAIRS:
            src_idx = self.label_map[src_name]
            tgt_idx = self.label_map[tgt_name]
            pair_dir = self.result_dir / f"{src_name}_to_{tgt_name}"
            pair_dir.mkdir(exist_ok=True)

            samples = self._collect_for_class(src_idx, n=3)
            if not samples:
                print(f"[CF] No samples found for class '{src_name}', skipping.")
                continue

            pair_validities   = []
            pair_proximity_l1 = []
            pair_proximity_l2 = []
            pair_sparsity     = []
            pair_iters        = []

            for i, img_tensor in enumerate(samples):
                result = self.cf_exp.generate(img_tensor, src_idx, tgt_idx)

                self._make_panel(
                    img_tensor, result, src_name, tgt_name,
                    pair_dir / f"cf_{i:02d}.png",
                )

                pair_validities.append(result["validity"])
                pair_proximity_l1.append(result["proximity_l1"])
                pair_proximity_l2.append(result["proximity_l2"])
                pair_sparsity.append(result["sparsity"])
                pair_iters.append(result["n_iter"])

                proximity_l1_all.append(result["proximity_l1"])
                proximity_l2_all.append(result["proximity_l2"])
                sparsity_all.append(result["sparsity"])
                n_iter_all.append(result["n_iter"])

            pair_key = f"{src_name}_to_{tgt_name}"
            all_stats[pair_key] = {
                "validity_rate"    : round(float(np.mean(pair_validities)),   4),
                "mean_proximity_l1": round(float(np.mean(pair_proximity_l1)), 6),
                "mean_proximity_l2": round(float(np.mean(pair_proximity_l2)), 6),
                "mean_sparsity"    : round(float(np.mean(pair_sparsity)),     4),
                "mean_n_iter"      : round(float(np.mean(pair_iters)),        1),
            }
            valid_counts[pair_key] = int(np.sum(pair_validities))

        elapsed = time.time() - start

        global_stats = {
            "global_validity_rate"    : round(float(np.mean([v["validity_rate"] for v in all_stats.values()])), 4) if all_stats else 0.0,
            "global_mean_proximity_l1": round(float(np.mean(proximity_l1_all)),  6) if proximity_l1_all else 0.0,
            "global_mean_proximity_l2": round(float(np.mean(proximity_l2_all)),  6) if proximity_l2_all else 0.0,
            "global_mean_sparsity"    : round(float(np.mean(sparsity_all)),      4) if sparsity_all else 0.0,
            "global_mean_n_iter"      : round(float(np.mean(n_iter_all)),        1) if n_iter_all else 0.0,
            "total_elapsed_s"         : round(elapsed, 2),
            "per_pair"                : all_stats,
        }

        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="Counterfactual Experiment (ACE-style)",
            conditions={
                "method"              : "Gradient-based ACE (Singla et al., 2023)",
                "max_iter"            : config.CF_MAX_ITER,
                "learning_rate"       : config.CF_LEARNING_RATE,
                "lambda_l1"           : config.CF_LAMBDA_L1,
                "lambda_l2"           : config.CF_LAMBDA_L2,
                "confidence_threshold": config.CF_CONFIDENCE_THRES,
                "pixel_threshold"     : config.CF_PIXEL_THRESHOLD,
                "cf_pairs"            : [f"{s}→{t}" for s, t in self.CF_PAIRS],
                "samples_per_pair"    : 3,
            },
            statistics=global_stats,
        )

        print(
            f"[Counterfactual] Done. "
            f"Global validity: {global_stats['global_validity_rate']:.4f}  "
            f"Proximity L1: {global_stats['global_mean_proximity_l1']:.4f}  "
            f"Sparsity: {global_stats['global_mean_sparsity']:.4f}  "
            f"Elapsed: {elapsed:.1f}s"
        )
        return global_stats
