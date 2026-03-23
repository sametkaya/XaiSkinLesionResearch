"""
train_segmenter.py
------------------
Train U-Net lesion segmentation model on HAM10000 expert masks.

Uses the same patient-level train/val/test split as the classifier
to ensure no data leakage.

Usage:
    python3 train_segmenter.py

Output:
    results/run_XX_xai_dermoscopy/12_segmentation/
        best_unet.pth          — best model weights
        training_curves.png    — loss/dice curves
        examples/              — visual predictions on test set
        result.txt             — metrics summary

References:
    Ronneberger et al. (2015). U-Net. MICCAI.
    Codella et al. (2018). ISIC 2018 Challenge.
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.segmentation.segmenter import LesionUNet
from src.data_loader import load_metadata, set_seed, stratified_patient_split
from src import config
from src.abc.config_abc import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD
from src.utils.result_manager import ResultManager


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """
    HAM10000 image + mask pairs for segmentation training.
    
    Parameters
    ----------
    df : DataFrame with 'image_id' and 'path' columns
    mask_dir : Path to segmentation masks
    image_size : int
    augment : bool
    """
    
    def __init__(self, df, mask_dir: Path, image_size: int = 224, augment: bool = False):
        self.records = []
        self.image_size = image_size
        self.augment = augment
        
        for _, row in df.iterrows():
            img_path = Path(row["filepath"])
            mask_path = mask_dir / f"{row['image_id']}_segmentation.png"
            if img_path.exists() and mask_path.exists():
                self.records.append((img_path, mask_path))
        
        print(f"  [SegDataset] {len(self.records)}/{len(df)} pairs found (augment={augment})")
        
        # Normalize transform
        self.normalize = transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.records[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convert to numpy for augmentation
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        # Augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_np = np.fliplr(img_np).copy()
                mask_np = np.fliplr(mask_np).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                img_np = np.flipud(img_np).copy()
                mask_np = np.flipud(mask_np).copy()
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            img_np = np.rot90(img_np, k).copy()
            mask_np = np.rot90(mask_np, k).copy()
            # Random brightness/contrast
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                img_np = np.clip(img_np.astype(float) * factor, 0, 255).astype(np.uint8)
        
        # To tensor
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_t = self.normalize(img_t)
        
        # Mask to binary tensor [0, 1]
        mask_t = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)
        
        return img_t, mask_t


# ─────────────────────────────────────────────
# Loss: BCE + Dice
# ─────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.
    
    BCE handles pixel-level accuracy.
    Dice handles class imbalance (small lesions).
    """
    
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # BCE (on logits)
        bce_loss = self.bce(logits, targets)
        
        # Dice (on probabilities)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Compute IoU and Dice for binary masks."""
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    iou = intersection / max(union, 1)
    dice = 2 * intersection / max(pred.sum() + gt.sum(), 1)
    
    return {"iou": float(iou), "dice": float(dice)}


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    total_dice = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Dice metric
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            smooth = 1e-6
            inter = (pred * masks).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = ((2 * inter + smooth) / (union + smooth)).mean()
        
        total_loss += loss.item()
        total_dice += dice.item()
    
    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        loss = criterion(logits, masks)
        
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        smooth = 1e-6
        
        inter = (pred * masks).sum(dim=(2, 3))
        union_dice = pred.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
        dice = ((2 * inter + smooth) / (union_dice + smooth)).mean()
        
        union_iou = ((pred + masks) > 0).float().sum(dim=(2, 3))
        iou = ((inter + smooth) / (union_iou + smooth)).mean()
        
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
    
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def save_examples(model, loader, device, save_dir: Path, n: int = 20):
    """Save visual prediction examples."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    mean = np.array(IMAGE_MEAN)
    std = np.array(IMAGE_STD)
    count = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images_gpu = images.to(device)
            logits = model(images_gpu)
            probs = torch.sigmoid(logits).cpu()
            
            for i in range(len(images)):
                if count >= n:
                    return
                
                # Denormalize image
                img = images[i].permute(1, 2, 0).numpy()
                img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
                
                gt = masks[i, 0].numpy()
                pred = probs[i, 0].numpy()
                pred_bin = (pred > 0.5).astype(np.uint8) * 255
                
                # 4-panel figure
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                axes[0].imshow(img)
                axes[0].set_title("Input Image", fontsize=9)
                axes[0].axis("off")
                
                axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Ground Truth", fontsize=9)
                axes[1].axis("off")
                
                axes[2].imshow(pred, cmap="hot", vmin=0, vmax=1)
                axes[2].set_title("Prediction (prob)", fontsize=9)
                axes[2].axis("off")
                
                # Overlay: green=TP, red=FP, blue=FN
                overlay = img.copy()
                gt_bool = gt > 0.5
                pred_bool = pred > 0.5
                tp = gt_bool & pred_bool
                fp = ~gt_bool & pred_bool
                fn = gt_bool & ~pred_bool
                overlay[tp] = overlay[tp] * 0.5 + np.array([0, 200, 0]) * 0.5
                overlay[fp] = overlay[fp] * 0.5 + np.array([200, 0, 0]) * 0.5
                overlay[fn] = overlay[fn] * 0.5 + np.array([0, 0, 200]) * 0.5
                axes[3].imshow(overlay.astype(np.uint8))
                
                m = compute_metrics(pred, gt)
                axes[3].set_title(f"Overlay  IoU={m['iou']:.3f}  Dice={m['dice']:.3f}", fontsize=9)
                axes[3].axis("off")
                
                plt.tight_layout()
                plt.savefig(save_dir / f"example_{count+1:03d}.png", dpi=150, bbox_inches="tight")
                plt.close()
                count += 1


def plot_curves(history, save_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCE+Dice)")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history["train_dice"], label="Train")
    axes[1].plot(epochs, history["val_dice"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Dice Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history["val_iou"], label="Val IoU", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("Validation IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train U-Net Lesion Segmenter")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--mask-dir", type=str,
                        default="datas/HAM10000/segmentations")
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Segmenter] Device: {device}")
    
    mask_dir = Path(args.mask_dir)
    print(f"[Segmenter] Mask dir: {mask_dir} ({len(list(mask_dir.glob('*.png')))} masks)")
    
    # ── Output directory ──────────────────────
    # Use latest run folder
    results_dir = Path("results")
    runs = sorted(results_dir.glob("run_*_xai_dermoscopy"))
    if runs:
        run_dir = runs[-1]
    else:
        run_dir = results_dir / "run_01_xai_dermoscopy"
    
    out_dir = run_dir / "12_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_unet.pth"
    
    print(f"[Segmenter] Output: {out_dir}")
    
    # ── Data split (same as classifier) ───────
    print("\n[Segmenter] Loading data...")
    df = load_metadata()
    train_df, val_df, test_df = stratified_patient_split(df)
    
    train_ds = SegmentationDataset(train_df, mask_dir, IMAGE_SIZE, augment=True)
    val_ds = SegmentationDataset(val_df, mask_dir, IMAGE_SIZE, augment=False)
    test_ds = SegmentationDataset(test_df, mask_dir, IMAGE_SIZE, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    
    # ── Model ─────────────────────────────────
    model = LesionUNet(in_channels=3, base_filters=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Segmenter] U-Net params: {n_params:,}")
    
    # ── Training setup ────────────────────────
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # ── Training loop ─────────────────────────
    print(f"\n[Segmenter] Training — {args.epochs} epochs, BS={args.batch_size}, LR={args.lr}")
    print(f"            Patience={args.patience}, Loss=BCE+Dice\n")
    
    history = {"train_loss": [], "val_loss": [], "train_dice": [],
               "val_dice": [], "val_iou": []}
    best_dice = 0.0
    patience_counter = 0
    t0 = time.time()
    
    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, criterion, device)
        
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        
        improved = ""
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            improved = " ← best ✓"
        else:
            patience_counter += 1
        
        elapsed = time.time() - ep_t
        print(f"Epoch [{epoch:03d}/{args.epochs}]  "
              f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
              f"Dice: {train_dice:.4f}/{val_dice:.4f}  "
              f"IoU: {val_iou:.4f}  "
              f"LR: {lr:.2e}  {elapsed:.1f}s{improved}")
        
        if patience_counter >= args.patience:
            print(f"\n[Segmenter] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
    total_time = (time.time() - t0) / 60
    print(f"\n[Segmenter] Done. Best Val Dice: {best_dice:.4f} ({total_time:.1f} min)")
    
    # ── Load best and evaluate on test ────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, device)
    print(f"[Segmenter] Test — Loss: {test_loss:.4f}  Dice: {test_dice:.4f}  IoU: {test_iou:.4f}")
    
    # ── Save outputs ──────────────────────────
    plot_curves(history, out_dir / "training_curves.png")
    save_examples(model, test_loader, device, out_dir / "examples", n=30)
    
    # result.txt
    rm = ResultManager(out_dir)
    rm.write_result(
        experiment_name="U-Net Lesion Segmentation",
        conditions={
            "architecture": "LesionUNet (32-64-128-256)",
            "training_data": f"HAM10000 ({len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test)",
            "loss": "BCE + Dice (0.5 weight each)",
            "optimizer": f"AdamW (lr={args.lr}, wd=1e-4)",
            "scheduler": f"CosineAnnealing (T_max={args.epochs})",
            "augmentation": "flip H/V, rotation 90°, brightness",
            "image_size": IMAGE_SIZE,
            "batch_size": args.batch_size,
        },
        statistics={
            "best_val_dice": round(best_dice, 4),
            "test_dice": round(test_dice, 4),
            "test_iou": round(test_iou, 4),
            "test_loss": round(test_loss, 4),
            "total_epochs": epoch,
            "training_minutes": round(total_time, 1),
        },
    )
    
    print(f"\n{'='*60}")
    print(f"  Segmentation Training Complete")
    print(f"  Best Val Dice : {best_dice:.4f}")
    print(f"  Test Dice     : {test_dice:.4f}")
    print(f"  Test IoU      : {test_iou:.4f}")
    print(f"  Weights       : {ckpt_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
