"""
explainers/lime_explainer.py
-----------------------------
LIME (Local Interpretable Model-Agnostic Explanations) for image classification.

LIME approximates the black-box classifier locally with an interpretable
surrogate model (Ridge regression) trained on perturbed versions of an input
image.  For images, the input space is segmented into super-pixels, and random
subsets of super-pixels are occluded to generate perturbed samples.

Key design choices:
  - QuickSHIFT segmentation (Vedaldi & Soatto, 2008) for superpixel creation,
    which performs well on dermoscopic texture patterns.
  - 1 000 perturbation samples per image (Ribeiro et al., 2016, recommend 1 k).
  - Explanations highlight both supporting (green) and contradicting (red)
    super-pixels relative to the predicted class.

References:
    Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust
    you?" Explaining the predictions of any classifier. KDD 2016.
    https://doi.org/10.1145/2939672.2939778
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

from src import config
from src.model import SkinLesionClassifier
from src.explainers.gradcam import denormalize
from src.utils.result_manager import ResultManager


class LIMEExplainer:
    """
    LIME-based image explainer for skin lesion classification.

    Parameters
    ----------
    model : SkinLesionClassifier
    device : torch.device
    """

    def __init__(self, model: SkinLesionClassifier, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()

        # LIME explainer object (algorithm = QuickSHIFT)
        self.explainer = lime_image.LimeImageExplainer(random_state=config.RANDOM_SEED)
        self.segmenter = SegmentationAlgorithm(
            "quickshift",
            kernel_size=4,
            max_dist=200,
            ratio=0.2,
        )

    def _predict_fn(self, images_np: np.ndarray) -> np.ndarray:
        """
        LIME-compatible batch prediction function.

        Accepts a batch of uint8 numpy images (N, H, W, 3) from LIME's
        perturbation engine, converts to normalised tensors, and returns
        class probability vectors (N, num_classes).

        Parameters
        ----------
        images_np : np.ndarray
            Batch of perturbed images with dtype uint8 in [0, 255].

        Returns
        -------
        np.ndarray
            Softmax probabilities, shape (N, num_classes).
        """
        import torchvision.transforms as T

        normalize = T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        to_tensor = T.ToTensor()

        tensors = []
        for img in images_np:
            pil_img = Image.fromarray(img.astype(np.uint8))
            t       = to_tensor(pil_img)     # [0, 1]
            t       = normalize(t)
            tensors.append(t)

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        num_samples: int = config.LIME_NUM_SAMPLES,
        num_features: int = config.LIME_NUM_SUPERPIXELS,
    ) -> Tuple:
        """
        Generate a LIME explanation for a single image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Single image tensor (3, H, W) normalised as per config.
        target_class : int, optional
            Class for which to generate the explanation.
            Defaults to the predicted class.
        num_samples : int
            Number of perturbation samples.
        num_features : int
            Number of super-pixels to include in the explanation mask.

        Returns
        -------
        Tuple
            (explanation, image_np, target_class, confidence)
            where explanation is a lime_image.ImageExplanation object.
        """
        image_np   = denormalize(image_tensor)      # (H, W, 3) uint8

        # Determine target class from model if not specified
        if target_class is None:
            inp    = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(inp)
                probs  = torch.softmax(logits, dim=1)
            target_class = int(probs.argmax(dim=1).item())
            confidence   = float(probs[0, target_class].item())
        else:
            inp    = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(inp), dim=1)
            confidence = float(probs[0, target_class].item())

        explanation = self.explainer.explain_instance(
            image_np,
            classifier_fn=self._predict_fn,
            top_labels=config.NUM_CLASSES,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=self.segmenter,
        )

        return explanation, image_np, target_class, confidence


# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────

class LIMEExperiment:
    """
    Run LIME explanations on a representative sample of test images.

    Parameters
    ----------
    model : SkinLesionClassifier
    test_loader : DataLoader
    device : torch.device
    result_dir : Path
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
        self.explainer   = LIMEExplainer(model, device)
        self.class_names = config.CLASS_LABELS

    def _collect_samples(self, n_per_class: int = 3) -> List[Tuple]:
        """Collect up to n_per_class correctly predicted images per class."""
        self.model.eval()
        buckets = {i: [] for i in range(config.NUM_CLASSES)}

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                preds  = self.model(images).argmax(dim=1).cpu()
                for img, lbl, prd in zip(images.cpu(), labels, preds):
                    if lbl == prd and len(buckets[int(lbl)]) < n_per_class:
                        buckets[int(lbl)].append((img, int(lbl)))
                if all(len(v) >= n_per_class for v in buckets.values()):
                    break

        return [item for items in buckets.values() for item in items]

    def run(self) -> dict:
        """
        Execute LIME experiments and save visualisations.

        Each panel shows:
          [Original] | [Positive segments (green)] | [All segments]

        Returns
        -------
        dict
            Experiment statistics.
        """
        start   = time.time()
        samples = self._collect_samples(n_per_class=3)

        per_class_dir = self.result_dir / "per_class"
        per_class_dir.mkdir(exist_ok=True)

        explanation_times = []

        for idx, (img_tensor, true_label) in enumerate(samples):
            t0 = time.time()
            exp, img_np, pred_class, confidence = self.explainer.explain(img_tensor)
            explanation_times.append(time.time() - t0)

            class_name = self.class_names[true_label]

            # Positive-only mask
            temp_pos, mask_pos = exp.get_image_and_mask(
                pred_class,
                positive_only=True,
                num_features=config.LIME_NUM_SUPERPIXELS,
                hide_rest=False,
            )

            # All segments (positive + negative)
            temp_all, mask_all = exp.get_image_and_mask(
                pred_class,
                positive_only=False,
                num_features=config.LIME_NUM_SUPERPIXELS,
                hide_rest=False,
            )

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))

            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\nTrue: {class_name}", fontsize=9)
            axes[0].axis("off")

            axes[1].imshow(mark_boundaries(temp_pos / 255.0, mask_pos))
            axes[1].set_title(
                f"LIME – Positive Regions\nPred: {self.class_names[pred_class]} ({confidence:.2f})",
                fontsize=9
            )
            axes[1].axis("off")

            axes[2].imshow(mark_boundaries(temp_all / 255.0, mask_all))
            axes[2].set_title("LIME – All Segments\n(green=for, red=against)", fontsize=9)
            axes[2].axis("off")

            plt.suptitle(
                f"LIME Explanation — {config.CLASS_NAMES[class_name]}",
                fontsize=10, fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(per_class_dir / f"lime_{idx:03d}_{class_name}.png", dpi=120)
            plt.close()

        elapsed = time.time() - start

        stats = {
            "total_samples"            : len(samples),
            "lime_num_samples"         : config.LIME_NUM_SAMPLES,
            "lime_num_superpixels"     : config.LIME_NUM_SUPERPIXELS,
            "mean_explanation_time_s"  : round(float(np.mean(explanation_times)), 3),
            "total_elapsed_s"          : round(elapsed, 2),
        }

        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="LIME Experiment",
            conditions={
                "model"             : config.MODEL_NAME,
                "segmentation_algo" : "QuickSHIFT",
                "num_samples_lime"  : config.LIME_NUM_SAMPLES,
                "num_superpixels"   : config.LIME_NUM_SUPERPIXELS,
                "samples_per_class" : 3,
            },
            statistics=stats,
        )

        print(
            f"[LIME] {len(samples)} explanations saved to {per_class_dir}  "
            f"(avg {np.mean(explanation_times):.1f}s/image)"
        )
        return stats
