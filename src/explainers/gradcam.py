"""
explainers/gradcam.py
---------------------
Gradient-weighted Class Activation Mapping (Grad-CAM) and Grad-CAM++.

Grad-CAM produces a coarse localisation map highlighting the regions that
the network uses to predict a specific class.  It requires no architectural
modifications and is applicable to any CNN.

Grad-CAM++ extends the original method by computing a second-order gradient
to improve localisation of multiple object occurrences.

References:
    Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D.,
    & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks
    via gradient-based localisation. ICCV 2017.

    Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N.
    (2018). Grad-CAM++: Generalised gradient-based visual explanations for
    deep convolutional networks. WACV 2018.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from src import config
from src.model import SkinLesionClassifier
from src.utils.result_manager import ResultManager


class GradCAM:
    """
    Grad-CAM explainer.

    Parameters
    ----------
    model : SkinLesionClassifier
    target_layer : nn.Module
        The convolutional layer from which to extract activations.
    device : torch.device
    """

    def __init__(
        self,
        model: SkinLesionClassifier,
        target_layer: nn.Module,
        device: torch.device,
    ):
        self.model        = model
        self.target_layer = target_layer
        self.device       = device

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _input, output: torch.Tensor) -> None:
        """Forward hook: store feature maps."""
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output: Tuple) -> None:
        """Backward hook: store gradients of the score w.r.t. feature maps."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute the Grad-CAM heatmap for a single image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Single image of shape (1, 3, H, W).
        target_class : int, optional
            Class index for which to generate the map.  If None, the
            predicted class is used.

        Returns
        -------
        Tuple[np.ndarray, int, float]
            (heatmap [H×W float32 in [0,1]], target_class, confidence)
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        # torch.enable_grad() ensures gradients are computed even if this
        # method is called from within a torch.no_grad() context block.
        with torch.enable_grad():
            logits = self.model(image_tensor)
            probs  = torch.softmax(logits, dim=1)

            if target_class is None:
                target_class = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, target_class].item())

            # Backward pass for target class
            self.model.zero_grad()
            logits[0, target_class].backward(retain_graph=False)

        # Global average pooling of gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam).squeeze().detach().cpu().numpy()      # (h, w)

        # Resize to input resolution and normalise
        cam = cv2.resize(cam, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, confidence

    def remove_hooks(self) -> None:
        """Detach all registered hooks from the model."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ explainer.

    Overrides the ``generate`` method to use second-order gradients
    for improved localisation.
    """

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        with torch.enable_grad():
            logits = self.model(image_tensor)
            probs  = torch.softmax(logits, dim=1)

            if target_class is None:
                target_class = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, target_class].item())

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward(retain_graph=True)

        grads   = self._gradients            # (1, C, h, w)
        acts    = self._activations          # (1, C, h, w)
        exp_grads = torch.exp(score) * grads  # element-wise

        # Compute alpha coefficients (Chattopadhay et al., 2018, Eq. 19)
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        alpha    = grads ** 2 / (
            2.0 * grads ** 2 + sum_acts * grads ** 3 + 1e-7
        )
        alpha    = F.relu(alpha)

        weights  = (alpha * F.relu(exp_grads)).mean(dim=(2, 3), keepdim=True)
        cam      = (weights * acts).sum(dim=1, keepdim=True)
        cam      = F.relu(cam).squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, confidence


# ─────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    image : np.ndarray
        Original image (H, W, 3) in [0, 255] uint8.
    heatmap : np.ndarray
        Normalised heatmap (H, W) in [0, 1].
    alpha : float
        Transparency of the heatmap overlay.

    Returns
    -------
    np.ndarray
        Blended image (H, W, 3) uint8.
    """
    heatmap_colour = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_colour + (1 - alpha) * image).astype(np.uint8)
    return overlay


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet/HAM10000 normalisation and convert to uint8 numpy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor (3, H, W) on CPU.

    Returns
    -------
    np.ndarray
        Image array (H, W, 3) in [0, 255] uint8.
    """
    mean = np.array(config.IMAGE_MEAN)
    std  = np.array(config.IMAGE_STD)
    img  = tensor.detach().permute(1, 2, 0).numpy()
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────

class GradCAMExperiment:
    """
    Run Grad-CAM and Grad-CAM++ on a sample of test images and save
    all outputs to the designated result directory.

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
        self.class_names = config.CLASS_LABELS

    def _collect_samples(self, n_per_class: int = 3) -> List[Tuple]:
        """
        Collect up to n_per_class correctly classified images per class.

        Returns
        -------
        List[Tuple]
            Each element: (image_tensor, true_label)
        """
        self.model.eval()
        class_buckets = {i: [] for i in range(config.NUM_CLASSES)}

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                preds  = self.model(images).argmax(dim=1).cpu()
                for i, (img, lbl, prd) in enumerate(
                    zip(images.cpu(), labels, preds)
                ):
                    if lbl == prd and len(class_buckets[int(lbl)]) < n_per_class:
                        class_buckets[int(lbl)].append((img, int(lbl)))

                if all(len(v) >= n_per_class for v in class_buckets.values()):
                    break

        samples = []
        for items in class_buckets.values():
            samples.extend(items)
        return samples

    def run(self) -> dict:
        """
        Execute the Grad-CAM experiment.

        Generates side-by-side panels:
          [Original] | [Grad-CAM] | [Grad-CAM++]
        for a representative sample of each class.

        Returns
        -------
        dict
            Experiment statistics for result.txt.
        """
        start = time.time()
        target_layer = self.model.get_feature_layer()

        gcam    = GradCAM(self.model,       target_layer, self.device)
        gcam_pp = GradCAMPlusPlus(self.model, target_layer, self.device)

        samples = self._collect_samples(n_per_class=3)
        stats   = {
            "total_samples"  : len(samples),
            "samples_per_class": 3,
            "cam_variants"   : ["GradCAM", "GradCAM++"],
        }

        per_class_dir = self.result_dir / "per_class"
        per_class_dir.mkdir(exist_ok=True)

        all_confidences_gcam   = []
        all_confidences_gcampp = []

        for idx, (img_tensor, true_label) in enumerate(samples):
            img_np = denormalize(img_tensor)             # (H, W, 3)
            inp    = img_tensor.unsqueeze(0)             # (1, 3, H, W)

            cam_v1, pred_v1, conf_v1 = gcam.generate(inp)
            cam_v2, pred_v2, conf_v2 = gcam_pp.generate(inp)

            all_confidences_gcam.append(conf_v1)
            all_confidences_gcampp.append(conf_v2)

            overlay_v1 = overlay_heatmap(img_np, cam_v1)
            overlay_v2 = overlay_heatmap(img_np, cam_v2)

            class_name = self.class_names[true_label]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\nTrue: {class_name}", fontsize=9)
            axes[0].axis("off")

            axes[1].imshow(overlay_v1)
            axes[1].set_title(
                f"Grad-CAM\nPred: {self.class_names[pred_v1]} ({conf_v1:.2f})",
                fontsize=9
            )
            axes[1].axis("off")

            axes[2].imshow(overlay_v2)
            axes[2].set_title(
                f"Grad-CAM++\nPred: {self.class_names[pred_v2]} ({conf_v2:.2f})",
                fontsize=9
            )
            axes[2].axis("off")

            plt.suptitle(
                f"Gradient-weighted Class Activation Maps — {config.CLASS_NAMES[class_name]}",
                fontsize=10, fontweight="bold"
            )
            plt.tight_layout()
            save_path = per_class_dir / f"sample_{idx:03d}_{class_name}.png"
            plt.savefig(save_path, dpi=120)
            plt.close()

        gcam.remove_hooks()
        gcam_pp.remove_hooks()

        elapsed = time.time() - start

        stats["mean_confidence_gradcam"]   = round(float(np.mean(all_confidences_gcam)),   4)
        stats["mean_confidence_gradcampp"] = round(float(np.mean(all_confidences_gcampp)), 4)
        stats["elapsed_seconds"]           = round(elapsed, 2)

        # Write result.txt
        rm = ResultManager(self.result_dir)
        rm.write_result(
            experiment_name="Grad-CAM Experiment",
            conditions={
                "model"            : config.MODEL_NAME,
                "target_layer"     : str(self.model.get_feature_layer().__class__.__name__),
                "samples_per_class": 3,
                "num_classes"      : config.NUM_CLASSES,
                "image_size"       : config.IMAGE_SIZE,
            },
            statistics=stats,
        )

        print(f"[GradCAM] {len(samples)} panels saved to {per_class_dir}")
        return stats