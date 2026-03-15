"""
train.py
--------
Training loop for the HAM10000 skin lesion classifier.

Features:
  - CUDA GPU acceleration with automatic device detection.
  - Automatic Mixed Precision (AMP) via torch.cuda.amp for ~2x speedup
    and ~50% VRAM reduction on RTX GPUs (Micikevicius et al., 2018).
  - Label-smoothing cross-entropy loss (Szegedy et al., 2016).
  - Cosine annealing LR schedule (Loshchilov & Hutter, 2017).
  - Early stopping on validation macro-F1.
  - Model checkpointing with AMP scaler state.
  - Per-epoch GPU memory logging for reproducibility.

References:
    Micikevicius, P. et al. (2018). Mixed precision training. ICLR 2018.
    Szegedy, C. et al. (2016). Rethinking the inception architecture. CVPR 2016.
    Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic gradient descent
    with warm restarts. ICLR 2017.
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # torch >= 2.0 unified API
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from src import config
from src.model import SkinLesionClassifier


# ─────────────────────────────────────────────
# GPU utility
# ─────────────────────────────────────────────

def get_gpu_memory_mb() -> Optional[float]:
    """Return current GPU memory allocation in MB, or None on CPU."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024 ** 2, 1)
    return None


def print_gpu_info(device: torch.device) -> None:
    """
    Print GPU specs and enable cuDNN benchmark mode.

    cuDNN benchmark mode profles several convolution algorithms on the
    first batch and selects the fastest one for the given input size,
    yielding significant throughput gains on fixed-size inputs.
    """
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_vram = props.total_memory / 1024 ** 3
        print(
            f"[GPU] {props.name}  |  "
            f"VRAM: {total_vram:.1f} GB  |  "
            f"CUDA: {torch.version.cuda}  |  "
            f"cuDNN: {torch.backends.cudnn.version()}"
        )
        torch.backends.cudnn.benchmark = True
        print("[GPU] cuDNN benchmark mode: ON")
    else:
        print("[GPU] CUDA not available — running on CPU.")


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Parameters
    ----------
    smoothing : float
        Label smoothing factor in [0, 1).  A value of 0.1 redistributes
        10 % of the probability mass uniformly over all classes.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Full training and validation lifecycle with GPU + AMP support.

    Automatic Mixed Precision (AMP):
        - Forward pass and loss run in FP16 (autocast).
        - GradScaler prevents gradient underflow.
        - Optimizer step is performed in FP32.
        - ~2x throughput, ~50% VRAM on RTX GPUs.

    Parameters
    ----------
    model : SkinLesionClassifier
        Must already be moved to the target device.
    train_loader : DataLoader
    val_loader : DataLoader
    device : torch.device
    result_dir : Path
    use_amp : bool
        Enable AMP. Automatically disabled on CPU.
    """

    def __init__(
        self,
        model: SkinLesionClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        result_dir: Path,
        use_amp: bool = True,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.result_dir   = result_dir

        # AMP is CUDA-only
        self.use_amp = use_amp and (device.type == "cuda")
        if self.use_amp:
            print("[Trainer] AMP (FP16 forward) — ENABLED")
        else:
            print("[Trainer] AMP — DISABLED (CPU mode)")

        # GradScaler: no-op when use_amp=False
        self.scaler = GradScaler("cuda", enabled=self.use_amp)  # noqa

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6,
        )

        self.history: List[Dict]    = []
        self.best_val_f1: float     = -1.0
        self.best_epoch: int        = 0
        self.epochs_no_improve: int = 0

        self.checkpoint_path = result_dir / "best_model.pth"
        self.log_path        = result_dir / "training_log.csv"

    # ── Internal helpers ───────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, is_train: bool) -> Tuple[float, float]:
        """
        Run one training or validation epoch.

        Training:
            - autocast (FP16) forward pass
            - GradScaler backward + step
            - zero_grad(set_to_none=True) for speed
        Validation:
            - torch.no_grad() + autocast for memory efficiency

        Returns
        -------
        Tuple[float, float]
            (mean loss, macro F1-score)
        """
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss  = 0.0
        all_preds:  List[int] = []
        all_labels: List[int] = []

        desc = "Train" if is_train else "Val  "

        if is_train:
            for images, labels in tqdm(loader, desc=desc, leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", enabled=self.use_amp):
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item() * images.size(0)
                preds = logits.detach().argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().tolist())

        else:
            with torch.no_grad():
                for images, labels in tqdm(loader, desc=desc, leave=False):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    with autocast(device_type="cuda", enabled=self.use_amp):
                        logits = self.model(images)
                        loss   = self.criterion(logits, labels)

                    total_loss += loss.item() * images.size(0)
                    preds = logits.argmax(dim=1).cpu().numpy().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy().tolist())

        mean_loss = total_loss / len(loader.dataset)
        macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return mean_loss, macro_f1

    def _save_checkpoint(self, epoch: int) -> None:
        """Persist model weights, optimizer, scaler, and best metric."""
        torch.save(
            {
                "epoch"            : epoch,
                "model_state_dict" : self.model.state_dict(),
                "optimizer_state"  : self.optimizer.state_dict(),
                "scaler_state"     : self.scaler.state_dict(),
                "best_val_f1"      : self.best_val_f1,
                "amp_enabled"      : self.use_amp,
            },
            self.checkpoint_path,
        )

    def _write_log_header(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch", "train_loss", "train_f1",
                    "val_loss", "val_f1", "lr",
                    "epoch_time_s", "gpu_mem_mb",
                ],
            )
            writer.writeheader()

    def _append_log(self, row: Dict) -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch", "train_loss", "train_f1",
                    "val_loss", "val_f1", "lr",
                    "epoch_time_s", "gpu_mem_mb",
                ],
            )
            writer.writerow(row)

    # ── Public API ─────────────────────────────────────────────────────────

    def train(self) -> List[Dict]:
        """
        Execute the full training loop with GPU acceleration.

        Returns
        -------
        List[Dict]
            Per-epoch: loss, F1, lr, wall-time, GPU memory MB.
        """
        self._write_log_header()
        print_gpu_info(self.device)

        print(f"\n[Trainer] Starting — up to {config.NUM_EPOCHS} epochs")
        print(f"          Batch size  : {config.BATCH_SIZE}")
        print(f"          LR          : {config.LEARNING_RATE}  |  WD: {config.WEIGHT_DECAY}")
        print(f"          Early stop  : patience={config.EARLY_STOP_PATIENCE}")
        print(f"          Checkpoint  : {self.checkpoint_path}\n")

        total_start = time.time()

        for epoch in range(1, config.NUM_EPOCHS + 1):
            epoch_start = time.time()

            train_loss, train_f1 = self._run_epoch(self.train_loader, is_train=True)
            val_loss,   val_f1   = self._run_epoch(self.val_loader,   is_train=False)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            epoch_time = time.time() - epoch_start
            gpu_mem    = get_gpu_memory_mb() or 0.0

            if val_f1 > self.best_val_f1:
                self.best_val_f1       = val_f1
                self.best_epoch        = epoch
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch)
                improved_marker = "  ← best ✓"
            else:
                self.epochs_no_improve += 1
                improved_marker = ""

            row = {
                "epoch"       : epoch,
                "train_loss"  : round(train_loss, 5),
                "train_f1"    : round(train_f1,   5),
                "val_loss"    : round(val_loss,   5),
                "val_f1"      : round(val_f1,     5),
                "lr"          : round(current_lr, 8),
                "epoch_time_s": round(epoch_time, 2),
                "gpu_mem_mb"  : gpu_mem,
            }
            self._append_log(row)
            self.history.append(row)

            print(
                f"Epoch [{epoch:03d}/{config.NUM_EPOCHS}]  "
                f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
                f"F1: {train_f1:.4f}/{val_f1:.4f}  "
                f"LR: {current_lr:.2e}  "
                f"{epoch_time:.1f}s  "
                f"GPU: {gpu_mem:.0f}MB"
                f"{improved_marker}"
            )

            if self.epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                print(
                    f"\n[Trainer] Early stopping at epoch {epoch} "
                    f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)."
                )
                break

        total_time = time.time() - total_start
        print(
            f"\n[Trainer] Done.  "
            f"Best Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}  "
            f"({total_time/60:.1f} min total)"
        )
        return self.history


def load_best_model(
    model: SkinLesionClassifier,
    checkpoint_path: Path,
    device: torch.device,
) -> SkinLesionClassifier:
    """
    Load the best checkpoint into model and move to device.

    Parameters
    ----------
    model : SkinLesionClassifier
    checkpoint_path : Path
    device : torch.device

    Returns
    -------
    SkinLesionClassifier
        In eval mode on device.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    amp_flag = checkpoint.get("amp_enabled", False)
    print(
        f"[Trainer] Loaded checkpoint — epoch {checkpoint['epoch']}  "
        f"Val F1: {checkpoint['best_val_f1']:.4f}  "
        f"AMP: {'on' if amp_flag else 'off'}"
    )
    return model
