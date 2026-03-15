"""
src/abc/abc_trainer.py
----------------------
Training and evaluation loops for the ABC Regressor.

Implements a two-phase training strategy:
  Phase 1 (ABC_FREEZE_EPOCHS): Only the regression head is trained.
  Phase 2 (remaining epochs): Full network fine-tuning.

Metrics tracked per epoch:
  - Loss: Huber (Smooth L1)
  - MAE per criterion (A, B, C) and averaged
  - Pearson correlation per criterion
  - ICC (Intraclass Correlation Coefficient)

All metrics, per-epoch logs, and plots are saved to the experiment
result directory.

References
----------
Koo, T. K., & Li, M. Y. (2016).
    A guideline of selecting and reporting intraclass correlation
    coefficients for reliability research. JCCA, 15(2), 155–163.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.abc.ordinal_loss import build_loss, SORDLoss
from src.abc.color_constancy import build_dermoscopy_transforms
from src.abc.config_abc import (
    ABC_BATCH_SIZE, ABC_NUM_EPOCHS, ABC_LEARNING_RATE,
    ABC_WEIGHT_DECAY, ABC_EARLY_STOP_PATIENCE, ABC_FREEZE_EPOCHS,
    ABC_USE_AMP, ABC_NUM_WORKERS, RANDOM_SEED, ABC_CRITERIA,
)
from src.abc.abc_model import ABCRegressor
from src.utils.result_manager import ResultManager


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def _pearson(preds: np.ndarray, targets: np.ndarray) -> float:
    """Pearson r between prediction and target vectors."""
    if len(preds) < 3:
        return float("nan")
    r, _ = stats.pearsonr(preds, targets)
    return float(r)


def _icc(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Two-way mixed, single measures ICC (ICC 2,1).

    ICC = (MS_r - MS_e) / (MS_r + (k-1)*MS_e + k*(MS_c - MS_e)/n)
    where k=2 (raters), n=number of subjects.
    """
    if len(preds) < 3:
        return float("nan")
    n = len(preds)
    data = np.vstack([preds, targets]).T          # (n, 2)
    row_means = data.mean(axis=1, keepdims=True)
    col_means = data.mean(axis=0, keepdims=True)
    grand     = data.mean()

    SS_row = 2 * np.sum((row_means - grand) ** 2)
    SS_col = 2 * np.sum((col_means - grand) ** 2)
    SS_tot = np.sum((data - grand) ** 2)
    SS_err = SS_tot - SS_row - SS_col

    MS_row = SS_row / (n - 1)
    MS_err = SS_err / ((n - 1) * (2 - 1))
    MS_col = SS_col / (2 - 1)

    denom = MS_row + MS_err + 2 * max(0, MS_col - MS_err) / n
    if denom == 0:
        return float("nan")
    return float((MS_row - MS_err) / denom)


# ─────────────────────────────────────────────
# ABC Trainer
# ─────────────────────────────────────────────

class ABCTrainer:
    """
    Trainer for the ABCRegressor model.

    Parameters
    ----------
    model : ABCRegressor
    train_loader : DataLoader
        Combined PH2 + Derm7pt training set.
    val_loader : DataLoader
        Combined PH2 + Derm7pt validation set.
    device : torch.device
    result_dir : Path
        Directory to save checkpoints, logs, and plots.
    use_amp : bool
        Enable automatic mixed precision.
    """

    def __init__(
        self,
        model: ABCRegressor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        result_dir: Path,
        use_amp: bool = ABC_USE_AMP,
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.device        = device
        self.result_dir    = result_dir
        self.use_amp       = use_amp and device.type == "cuda"

        # OrdinalHuber loss: respects rank structure of ABC annotations.
        # lambda_rank kept small (0.05) to avoid noise with small batches.
        # Reference: Cao et al. (2020), Díaz & Marathe (2019).
        from src.abc.config_abc import ABC_RANK_LAMBDA
        self.criterion  = build_loss("ordinal_huber", beta=0.1, lambda_rank=ABC_RANK_LAMBDA)
        self.sord_loss  = None   # set by trainer if model uses SORD head
        self.use_sord   = hasattr(model, "use_sord") and model.use_sord
        if self.use_sord:
            from src.abc.ordinal_loss import SORDLoss
            self.sord_loss = SORDLoss(num_bins=model.num_bins, sigma=1.5)

        # Only optimise trainable params (head initially)
        self.optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=ABC_LEARNING_RATE,
            weight_decay=ABC_WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=ABC_NUM_EPOCHS, eta_min=1e-6
        )

        self.scaler      = GradScaler(enabled=self.use_amp)
        self.best_val_mae = float("inf")
        self.patience_cnt = 0
        self.history: List[Dict] = []

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict:
        """Run one epoch and return metrics dict."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds  = []   # (N, 3)
        all_targets= []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, _masks, abc_targets, _meta in tqdm(
                loader, leave=False,
                desc="Train" if training else "Val  ",
            ):
                images      = images.to(self.device, non_blocking=True)
                abc_targets = abc_targets.to(self.device, non_blocking=True)

                if training:
                    self.optimiser.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    preds = self.model(images)
                    if self.use_sord and self.sord_loss is not None:
                        # preds: (B, 3, K) logits; decode to scores for metrics
                        loss  = self.sord_loss(preds, abc_targets)
                        preds_scores = self.sord_loss.decode(preds.detach())
                    else:
                        loss  = self.criterion(preds, abc_targets)
                        preds_scores = preds.detach()

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimiser)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.scaler.step(self.optimiser)
                    self.scaler.update()

                total_loss  += loss.item() * images.size(0)
                all_preds.append(preds_scores.cpu().numpy())
                all_targets.append(abc_targets.detach().cpu().numpy())

        n          = sum(len(b) for b in all_preds)
        avg_loss   = total_loss / max(n, 1)
        preds_all  = np.vstack(all_preds)   # (N, 3)
        tgts_all   = np.vstack(all_targets) # (N, 3)

        mae_per = np.abs(preds_all - tgts_all).mean(axis=0)   # (3,)
        mae_avg = float(mae_per.mean())

        metrics = {
            "loss"  : round(avg_loss, 5),
            "mae"   : round(mae_avg, 4),
        }
        for i, crit in enumerate(ABC_CRITERIA):
            metrics[f"mae_{crit}"] = round(float(mae_per[i]), 4)
            metrics[f"r_{crit}"]   = round(
                _pearson(preds_all[:, i], tgts_all[:, i]), 4
            )
            metrics[f"icc_{crit}"] = round(
                _icc(preds_all[:, i], tgts_all[:, i]), 4
            )

        return metrics, preds_all, tgts_all

    def train(self, num_epochs: int = ABC_NUM_EPOCHS) -> List[Dict]:
        """
        Full training loop with two-phase backbone strategy and early stopping.

        Parameters
        ----------
        num_epochs : int

        Returns
        -------
        List[Dict]  — epoch-level metric history
        """
        print(f"\n[ABCTrainer] Starting — up to {num_epochs} epochs")
        print(
            f"             Batch size  : {ABC_BATCH_SIZE}\n"
            f"             LR          : {ABC_LEARNING_RATE}\n"
            f"             Freeze phase: {ABC_FREEZE_EPOCHS} epochs\n"
            f"             Early stop  : patience={ABC_EARLY_STOP_PATIENCE}\n"
            f"             Checkpoint  : {self.result_dir / 'checkpoints'}\n"
        )

        ckpt_dir = self.result_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            # ── Phase transition ──────────────────────
            if epoch == ABC_FREEZE_EPOCHS + 1:
                self.model.set_backbone_trainable(True)
                # Re-create optimiser with full parameter set
                self.optimiser = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=ABC_LEARNING_RATE * 0.1,   # lower LR for fine-tuning
                    weight_decay=ABC_WEIGHT_DECAY,
                )
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimiser,
                    T_max=num_epochs - ABC_FREEZE_EPOCHS,
                    eta_min=1e-7,
                )
                print(f"[ABCTrainer] Phase 2 — full fine-tuning from epoch {epoch}")

            train_metrics, _, _ = self._run_epoch(self.train_loader, training=True)
            val_metrics,   _, _ = self._run_epoch(self.val_loader,   training=False)

            self.scheduler.step()
            elapsed = time.time() - t0

            lr_now = self.optimiser.param_groups[0]["lr"]
            marker = ""

            if val_metrics["mae"] < self.best_val_mae:
                self.best_val_mae = val_metrics["mae"]
                self.patience_cnt = 0
                marker = " ← best ✓"
                # Save checkpoint
                torch.save(
                    {
                        "epoch"              : epoch,
                        "model_state_dict"   : self.model.state_dict(),
                        "optimiser_state_dict": self.optimiser.state_dict(),
                        "best_val_mae"       : self.best_val_mae,
                        "scaler_state_dict"  : self.scaler.state_dict(),
                    },
                    ckpt_dir / "best_abc_model.pth",
                )
            else:
                self.patience_cnt += 1

            row = {
                "epoch"       : epoch,
                "lr"          : lr_now,
                "elapsed_s"   : round(elapsed, 1),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}"  : v for k, v in val_metrics.items()},
            }
            self.history.append(row)

            print(
                f"Epoch [{epoch:03d}/{num_epochs}]  "
                f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}  "
                f"MAE: {train_metrics['mae']:.4f}/{val_metrics['mae']:.4f}  "
                f"LR: {lr_now:.2e}  {elapsed:.1f}s{marker}"
            )

            # Early stopping
            if self.patience_cnt >= ABC_EARLY_STOP_PATIENCE:
                print(
                    f"\n[ABCTrainer] Early stopping at epoch {epoch} "
                    f"(no improvement for {ABC_EARLY_STOP_PATIENCE} epochs)."
                )
                break

        # Save training log
        log_df = pd.DataFrame(self.history)
        log_df.to_csv(self.result_dir / "training_log.csv", index=False)

        print(
            f"\n[ABCTrainer] Done.  "
            f"Best Val MAE: {self.best_val_mae:.4f}"
        )

        # Load best checkpoint
        best_path = ckpt_dir / "best_abc_model.pth"
        if best_path.exists():
            state = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            print(
                f"[ABCTrainer] Loaded best checkpoint — "
                f"epoch {state['epoch']}  Val MAE: {state['best_val_mae']:.4f}"
            )

        return self.history


def build_combined_loaders(
    ph2_train, ph2_val, ph2_test,
    d7_train,  d7_val,  d7_test,
    batch_size: int = ABC_BATCH_SIZE,
    num_workers: int = ABC_NUM_WORKERS,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Combine PH2 and Derm7pt datasets into joint train/val/test loaders.

    Parameters
    ----------
    ph2_train, ph2_val, ph2_test : PH2Dataset
    d7_train, d7_val, d7_test   : Derm7ptDataset
    batch_size, num_workers, seed

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    g = torch.Generator()
    g.manual_seed(seed)

    # PH2 datasets are optional — combine only if available
    train_parts = ([ph2_train] if ph2_train is not None else []) + [d7_train]
    val_parts   = ([ph2_val]   if ph2_val   is not None else []) + [d7_val]
    test_parts  = ([ph2_test]  if ph2_test  is not None else []) + [d7_test]

    train_ds = ConcatDataset(train_parts)
    val_ds   = ConcatDataset(val_parts)
    test_ds  = ConcatDataset(test_parts)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=g, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(
        f"[ABCTrainer] Combined loaders — "
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader