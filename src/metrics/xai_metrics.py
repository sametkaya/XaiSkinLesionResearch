"""
metrics/xai_metrics.py
-----------------------
Quantitative evaluation metrics for XAI methods.

Metrics implemented:

1. Faithfulness – Deletion AUC
   The heatmap pixels are removed (set to baseline) in descending order of
   importance.  The area under the model's confidence curve as more pixels
   are deleted measures how faithful the explanation is: a high AUC means
   the explanation correctly identifies the most important pixels.
   (Samek et al., 2017; Petsiuk et al., 2018)

2. Faithfulness – Insertion AUC
   The inverse: pixels are revealed in descending order of importance.
   A fast rise in confidence indicates a faithful explanation.

3. Counterfactual Metrics:
   - Validity Rate   : fraction of CFs that flip the prediction
   - Proximity L1    : mean absolute pixel-level distance
   - Proximity L2    : mean squared pixel-level distance
   - Sparsity        : fraction of pixels perturbed beyond threshold

4. FID (Fréchet Inception Distance)
   Measures the distribution distance between original test images and
   counterfactual images.  Lower FID indicates more realistic CFs.
   Implemented in metrics/fid.py.

References:
    Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K.-R.
    (2017). Evaluating the visualisation of what a Deep Neural Network has
    learned. IEEE Transactions on Neural Networks and Learning Systems.

    Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomised input
    sampling for explanation of black-box models. BMVC 2018.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple


def deletion_auc(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    predict_fn: Callable[[torch.Tensor], float],
    n_steps: int = 10,
    baseline_value: float = 0.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the Deletion AUC for a saliency/Grad-CAM heatmap.

    Pixels are sorted in descending order of attribution magnitude and
    progressively replaced with baseline_value.  The model's confidence on
    the originally predicted class is recorded at each step.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Single image (3, H, W).
    heatmap : np.ndarray
        Attribution heatmap (H, W) in [0, 1], higher = more important.
    predict_fn : Callable
        Function (image_tensor) → float confidence for target class.
    n_steps : int
        Number of deletion steps.
    baseline_value : float
        Value used to replace deleted pixels (0 = black).

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        (auc_score, step_fractions, step_confidences)
    """
    img    = image_tensor.clone()
    H, W   = heatmap.shape
    n_pix  = H * W

    # Flatten heatmap to get pixel indices sorted by importance (descending)
    flat_idx = np.argsort(-heatmap.flatten())

    fractions    = np.linspace(0, 1, n_steps + 1)
    confidences  = []

    for frac in fractions:
        n_del = int(frac * n_pix)
        mask  = torch.ones(H * W, dtype=torch.bool)
        if n_del > 0:
            mask[flat_idx[:n_del]] = False
        mask_2d = mask.view(H, W)

        modified = img.clone()
        modified[:, ~mask_2d] = baseline_value

        conf = predict_fn(modified.unsqueeze(0))
        confidences.append(conf)

    confidences = np.array(confidences)
    auc_score   = float(np.trapz(confidences, fractions))
    return auc_score, fractions, confidences


def insertion_auc(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    predict_fn: Callable[[torch.Tensor], float],
    n_steps: int = 10,
    baseline_value: float = 0.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the Insertion AUC for a saliency/Grad-CAM heatmap.

    Pixels are progressively revealed in descending order of attribution,
    starting from a fully-occluded (baseline) image.

    Parameters
    ----------
    (same as deletion_auc)

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        (auc_score, step_fractions, step_confidences)
    """
    H, W  = heatmap.shape
    n_pix = H * W

    flat_idx = np.argsort(-heatmap.flatten())

    fractions   = np.linspace(0, 1, n_steps + 1)
    confidences = []

    for frac in fractions:
        n_ins    = int(frac * n_pix)
        mask     = torch.zeros(H * W, dtype=torch.bool)
        if n_ins > 0:
            mask[flat_idx[:n_ins]] = True
        mask_2d = mask.view(H, W)

        modified = torch.full_like(image_tensor, baseline_value)
        modified[:, mask_2d] = image_tensor[:, mask_2d]

        conf = predict_fn(modified.unsqueeze(0))
        confidences.append(conf)

    confidences = np.array(confidences)
    auc_score   = float(np.trapz(confidences, fractions))
    return auc_score, fractions, confidences


def compute_faithfulness_metrics(
    model,
    device: torch.device,
    samples: List[Tuple[torch.Tensor, np.ndarray, int]],
    n_steps: int = 10,
) -> dict:
    """
    Compute mean Deletion-AUC and Insertion-AUC over a set of samples.

    Parameters
    ----------
    model : SkinLesionClassifier
    device : torch.device
    samples : List[Tuple[torch.Tensor, np.ndarray, int]]
        Each element is (image_tensor, heatmap, target_class_idx).
    n_steps : int
        Number of deletion/insertion steps.

    Returns
    -------
    dict
        {'mean_deletion_auc', 'mean_insertion_auc',
         'all_deletion_aucs', 'all_insertion_aucs'}
    """
    model.eval()

    def predict_fn(img_t: torch.Tensor) -> float:
        img_t = img_t.to(device)
        with torch.no_grad():
            probs = torch.softmax(model(img_t), dim=1)
        return float(probs[0, target_cls].item())

    del_aucs = []
    ins_aucs = []

    for img, heatmap, target_cls in samples:
        d_auc, _, _ = deletion_auc(img, heatmap, predict_fn, n_steps)
        i_auc, _, _ = insertion_auc(img, heatmap, predict_fn, n_steps)
        del_aucs.append(d_auc)
        ins_aucs.append(i_auc)

    return {
        "mean_deletion_auc" : round(float(np.mean(del_aucs)), 4),
        "mean_insertion_auc": round(float(np.mean(ins_aucs)), 4),
        "all_deletion_aucs" : [round(v, 4) for v in del_aucs],
        "all_insertion_aucs": [round(v, 4) for v in ins_aucs],
    }


def compute_cf_metrics(cf_results: List[dict]) -> dict:
    """
    Aggregate counterfactual metrics across a set of results.

    Parameters
    ----------
    cf_results : List[dict]
        Each dict is the output of CounterfactualExplainer.generate().

    Returns
    -------
    dict
        Aggregated mean metrics.
    """
    if not cf_results:
        return {}

    validity    = [r["validity"]     for r in cf_results]
    prox_l1     = [r["proximity_l1"] for r in cf_results]
    prox_l2     = [r["proximity_l2"] for r in cf_results]
    sparsity    = [r["sparsity"]     for r in cf_results]
    n_iters     = [r["n_iter"]       for r in cf_results]

    return {
        "validity_rate"    : round(float(np.mean(validity)), 4),
        "mean_proximity_l1": round(float(np.mean(prox_l1)),  6),
        "std_proximity_l1" : round(float(np.std(prox_l1)),   6),
        "mean_proximity_l2": round(float(np.mean(prox_l2)),  6),
        "std_proximity_l2" : round(float(np.std(prox_l2)),   6),
        "mean_sparsity"    : round(float(np.mean(sparsity)), 4),
        "std_sparsity"     : round(float(np.std(sparsity)),  4),
        "mean_n_iter"      : round(float(np.mean(n_iters)),  1),
    }
