#!/usr/bin/env python3
"""
apply_patches_v6.py
-------------------
Araştırma-tabanlı counterfactual iyileştirmelerini uygular.

5 kritik değişiklik:
  1. L1 → Negative LPIPS + TV + min-perturbation hinge
  2. Low-pass frequency filtering (her 10 adımda)
  3. Saliency-based δ initialization (sıfır yerine)
  4. Adam + CosineAnnealingWarmRestarts
  5. Geliştirilmiş görselleştirme (5-panel + amplified overlay)

Kullanım:
    cd XaiSkinLesionResearch
    python3 apply_patches_v6.py

Referanslar:
    Jeanneret et al. (2022/2023) — DiME/ACE (ACCV/CVPR)
    Singla et al. (2023) — Progressive exaggeration (MedIA)
    Guo et al. (2019) — Low frequency perturbation (UAI)
    Zhang et al. (2018) — LPIPS perceptual metric (CVPR)
"""

import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
BACKUP_SUFFIX = ".bak_pre_v6"


def backup(path: Path):
    bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  ✓ Yedek: {bak.name}")


def main():
    print("\n" + "=" * 70)
    print("  v6 — Araştırma-Tabanlı Counterfactual İyileştirmeleri")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # 1. cf_losses.py yeni modülü kopyala
    # ═══════════════════════════════════════════
    print("\n[1/3] src/explainers/cf_losses.py")
    src = PROJECT / "v6" / "src" / "explainers" / "cf_losses.py"
    dst = PROJECT / "src" / "explainers" / "cf_losses.py"
    if src.exists():
        shutil.copy2(src, dst)
        print("  ✓ cf_losses.py kopyalandı")
    else:
        print(f"  ⚠ {src} bulunamadı — lütfen dosyayı manuel kopyala")

    # ═══════════════════════════════════════════
    # 2. config_abc.py — yeni CF hiperparametreleri
    # ═══════════════════════════════════════════
    print("\n[2/3] src/abc/config_abc.py")
    config_abc = PROJECT / "src" / "abc" / "config_abc.py"
    backup(config_abc)
    text = config_abc.read_text(encoding="utf-8")

    # Eski CF blok
    old_cf = """# ─────────────────────────────────────────────
# ABC-guided Counterfactual
# ─────────────────────────────────────────────
# Loss = λ_cls * L_cls + λ_A * |ΔA| + λ_B * |ΔB| + λ_C * |ΔC| + λ_l1 * ‖δ‖₁
ABC_CF_MAX_ITER          = 2000
ABC_CF_LEARNING_RATE     = 0.005e-3
ABC_CF_LAMBDA_CLS        = 1.0    # classification loss weight
ABC_CF_LAMBDA_A          = 0.8    # asymmetry preservation weight
ABC_CF_LAMBDA_B          = 0.6    # border preservation weight
ABC_CF_LAMBDA_C          = 0.4    # color preservation weight
ABC_CF_LAMBDA_L1         = 2.0    # pixel sparsity weight
ABC_CF_CONFIDENCE_THRES  = 0.85    # target class probability threshold
ABC_CF_NUM_IMAGES        = 10     # images per class transition pair
ABC_CF_PIXEL_THRESHOLD   = 0.05   # threshold for sparsity mask"""

    new_cf = """# ─────────────────────────────────────────────
# ABC-guided Counterfactual (v6 — research-based)
# ─────────────────────────────────────────────
# Loss = λ_cls·CE + λ_A·|ΔA| + λ_B·|ΔB| + λ_C·|ΔC|
#      + λ_tv·TV(δ) − λ_lpips·LPIPS(x, x+δ) + λ_hinge·max(0, τ−|δ|)
#
# Key changes from v5 (see research report):
#   - L1 sparsity REMOVED — it suppresses visible perturbations
#   - Negative LPIPS ADDED — encourages perceptually visible changes
#   - TV regularization ADDED — spatial smoothness (blobs, not noise)
#   - Min-perturbation hinge ADDED — enforces minimum visibility
#   - Low-pass frequency filter — removes imperceptible HF noise
#   - Saliency initialization — start from meaningful direction
#
# References:
#   Jeanneret et al. (2022). DiME. ACCV 2022.
#   Jeanneret et al. (2023). ACE. CVPR 2023.
#   Singla et al. (2023). Progressive exaggeration. MedIA.
#   Guo et al. (2019). Low frequency perturbation. UAI.
#   Zhang et al. (2018). LPIPS. CVPR 2018.
ABC_CF_MAX_ITER          = 500     # more iterations for visible changes
ABC_CF_LEARNING_RATE     = 0.01    # higher LR with Adam + cosine schedule
ABC_CF_LAMBDA_CLS        = 5.0     # strong classification push
ABC_CF_LAMBDA_A          = 1.0     # asymmetry preservation weight
ABC_CF_LAMBDA_B          = 0.8     # border preservation weight
ABC_CF_LAMBDA_C          = 0.6     # color preservation weight
ABC_CF_LAMBDA_TV         = 5e-3    # total variation — spatial smoothness
ABC_CF_LAMBDA_LPIPS      = 1.0     # negative LPIPS — encourage visibility
ABC_CF_LAMBDA_HINGE      = 10.0    # min-perturbation hinge weight
ABC_CF_HINGE_TAU         = 0.03    # minimum avg |δ| threshold
ABC_CF_CONFIDENCE_THRES  = 0.90    # target class probability threshold
ABC_CF_NUM_IMAGES        = 10      # images per class transition pair
ABC_CF_PIXEL_THRESHOLD   = 0.02    # threshold for sparsity mask
ABC_CF_LOWPASS_RATIO     = 0.30    # frequency cutoff (keep lowest 30%)
ABC_CF_LOWPASS_EVERY     = 10      # apply low-pass every N steps
ABC_CF_SALIENCY_SCALE    = 0.05    # saliency init magnitude"""

    if old_cf in text:
        text = text.replace(old_cf, new_cf, 1)
        print("  ✓ CF hiperparametreleri güncellendi (v6)")
    else:
        # v5 ile güncellenmiş olabilir — alternatif arama
        print("  ⚠ Orijinal CF bloğu bulunamadı — config'i kontrol edin")
        # Yine de devam et
    
    config_abc.write_text(text, encoding="utf-8")

    # ═══════════════════════════════════════════
    # 3. abc_counterfactual.py — generate() güncelle
    # ═══════════════════════════════════════════
    print("\n[3/3] src/explainers/abc_counterfactual.py")
    cf_file = PROJECT / "src" / "explainers" / "abc_counterfactual.py"
    backup(cf_file)
    text = cf_file.read_text(encoding="utf-8")

    # 3a. Import'lara cf_losses ekle
    old_import = """from src.utils.result_manager import ResultManager"""
    new_import = """from src.utils.result_manager import ResultManager
from src.explainers.cf_losses import (
    VGGPerceptualLoss,
    total_variation_loss,
    low_pass_filter,
    saliency_init,
    min_perturbation_hinge,
)"""

    if old_import in text and "cf_losses" not in text:
        text = text.replace(old_import, new_import, 1)
        print("  ✓ cf_losses import'ları eklendi")

    # 3b. Config import'larını güncelle
    old_config_import = """from src.abc.config_abc import (
    ABC_CF_MAX_ITER, ABC_CF_LEARNING_RATE,
    ABC_CF_LAMBDA_CLS, ABC_CF_LAMBDA_A, ABC_CF_LAMBDA_B, ABC_CF_LAMBDA_C,
    ABC_CF_LAMBDA_L1, ABC_CF_CONFIDENCE_THRES,
    ABC_CF_NUM_IMAGES, ABC_CF_PIXEL_THRESHOLD, ABC_CF_PAIRS,
    IMAGE_MEAN, IMAGE_STD, IMAGE_SIZE, ABC_NAMES,
)"""

    new_config_import = """from src.abc.config_abc import (
    ABC_CF_MAX_ITER, ABC_CF_LEARNING_RATE,
    ABC_CF_LAMBDA_CLS, ABC_CF_LAMBDA_A, ABC_CF_LAMBDA_B, ABC_CF_LAMBDA_C,
    ABC_CF_LAMBDA_TV, ABC_CF_LAMBDA_LPIPS, ABC_CF_LAMBDA_HINGE,
    ABC_CF_HINGE_TAU, ABC_CF_CONFIDENCE_THRES,
    ABC_CF_NUM_IMAGES, ABC_CF_PIXEL_THRESHOLD, ABC_CF_PAIRS,
    ABC_CF_LOWPASS_RATIO, ABC_CF_LOWPASS_EVERY, ABC_CF_SALIENCY_SCALE,
    IMAGE_MEAN, IMAGE_STD, IMAGE_SIZE, ABC_NAMES,
)"""

    if old_config_import in text:
        text = text.replace(old_config_import, new_config_import, 1)
        print("  ✓ Config import'ları güncellendi")
    else:
        print("  ⚠ Config import bloğu bulunamadı — zaten güncel olabilir")

    # 3c. __init__'e perceptual_loss ekle
    old_init_end = """        self.clf.eval()
        self.abc_reg.eval()"""
    
    new_init_end = """        self.clf.eval()
        self.abc_reg.eval()
        
        # v6: Perceptual loss for visible counterfactuals
        self.perceptual_loss = VGGPerceptualLoss(device)"""

    if old_init_end in text and "perceptual_loss" not in text:
        text = text.replace(old_init_end, new_init_end, 1)
        print("  ✓ VGGPerceptualLoss __init__'e eklendi")

    # 3d. generate() optimization loop — TAM DEĞİŞİM
    # Adam + v5 versiyonunu bul
    old_generate_loop = """        # v5: Adam optimizer (manual SGD yerine) — daha hızlı yakınsama
        optimizer = torch.optim.Adam([delta], lr=lr)

        n_iter = 0
        with torch.enable_grad():
            for step in range(max_iter):
                optimizer.zero_grad()

                x_cf = cf + delta

                # Classification loss
                logits = self.clf(x_cf)
                probs  = torch.softmax(logits, dim=1)
                loss_cls = F.cross_entropy(logits, target_t) * ABC_CF_LAMBDA_CLS

                # ABC preservation losses
                cf_abc = self.abc_reg(x_cf).squeeze(0)   # (3,)
                loss_A = lambdas["A"] * torch.abs(cf_abc[0] - src_abc[0])
                loss_B = lambdas["B"] * torch.abs(cf_abc[1] - src_abc[1])
                loss_C = lambdas["C"] * torch.abs(cf_abc[2] - src_abc[2])

                # Sparsity
                loss_l1 = ABC_CF_LAMBDA_L1 * delta.abs().mean()

                loss = loss_cls + loss_A + loss_B + loss_C + loss_l1
                loss.backward()

                torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                optimizer.step()

                n_iter = step + 1

                if float(probs[0, target_class].item()) >= confidence_threshold:
                    break"""

    new_generate_loop = """        # v6: Research-based optimization
        # — Saliency init (Singla et al., MedIA 2023)
        # — Adam + CosineAnnealingWarmRestarts
        # — Negative LPIPS + TV + min-perturbation hinge (no L1!)
        # — Low-pass frequency filter (Guo et al., UAI 2019)
        
        # Initialize δ from saliency (not zeros)
        with torch.enable_grad():
            delta_init = saliency_init(
                self.clf, orig, target_class,
                scale=ABC_CF_SALIENCY_SCALE,
            )
        delta = delta_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([delta], lr=lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=lr * 0.01,
        )

        n_iter = 0
        with torch.enable_grad():
            for step in range(max_iter):
                optimizer.zero_grad()

                x_cf = cf + delta

                # ── Classification loss ──────────
                logits = self.clf(x_cf)
                probs  = torch.softmax(logits, dim=1)
                loss_cls = F.cross_entropy(logits, target_t) * ABC_CF_LAMBDA_CLS

                # ── ABC preservation losses ──────
                cf_abc = self.abc_reg(x_cf).squeeze(0)
                loss_A = lambdas["A"] * torch.abs(cf_abc[0] - src_abc[0])
                loss_B = lambdas["B"] * torch.abs(cf_abc[1] - src_abc[1])
                loss_C = lambdas["C"] * torch.abs(cf_abc[2] - src_abc[2])

                # ── TV regularization (spatial smoothness) ──
                loss_tv = ABC_CF_LAMBDA_TV * total_variation_loss(delta.unsqueeze(0)
                          if delta.dim() == 3 else delta)

                # ── Negative LPIPS (encourage perceptual visibility) ──
                loss_lpips = -ABC_CF_LAMBDA_LPIPS * self.perceptual_loss(
                    orig, x_cf,
                )

                # ── Min-perturbation hinge (enforce minimum change) ──
                loss_hinge = ABC_CF_LAMBDA_HINGE * min_perturbation_hinge(
                    delta, tau=ABC_CF_HINGE_TAU,
                )

                loss = (loss_cls + loss_A + loss_B + loss_C
                        + loss_tv + loss_lpips + loss_hinge)
                loss.backward()

                torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # ── Low-pass filter every N steps ──
                if (step + 1) % ABC_CF_LOWPASS_EVERY == 0:
                    with torch.no_grad():
                        d = delta.unsqueeze(0) if delta.dim() == 3 else delta
                        d_filtered = low_pass_filter(d, cutoff_ratio=ABC_CF_LOWPASS_RATIO)
                        delta.data.copy_(d_filtered.squeeze(0) if delta.dim() == 3
                                         else d_filtered)

                n_iter = step + 1

                if float(probs[0, target_class].item()) >= confidence_threshold:
                    break"""

    if old_generate_loop in text:
        text = text.replace(old_generate_loop, new_generate_loop, 1)
        print("  ✓ generate(): v6 optimization loop (LPIPS+TV+hinge+lowpass+saliency)")
    else:
        # v5 öncesi (orijinal) versiyonu dene
        old_original = """        n_iter = 0
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
                    break"""

        if old_original in text:
            text = text.replace(old_original, new_generate_loop, 1)
            print("  ✓ generate(): orijinal → v6 optimization loop")
        else:
            print("  ⚠ generate() loop bulunamadı — manuel güncelleme gerekebilir")

    # 3e. _save_panels görselleştirme — 5-panel + amplified overlay
    old_panels_title = '''        plt.suptitle(title, fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()'''

    new_panels_title = '''        plt.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()'''

    if old_panels_title in text:
        text = text.replace(old_panels_title, new_panels_title, 1)
        print("  ✓ _save_panels: dpi=150 + bbox_inches")

    cf_file.write_text(text, encoding="utf-8")

    # ═══════════════════════════════════════════
    # Özet
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  v6 yamaları uygulandı!")
    print("=" * 70)
    print("""
  Değişiklikler:
    1. cf_losses.py — Yeni modül: LPIPS, TV, low-pass, saliency init, hinge
    2. config_abc.py — L1 kaldırıldı, LPIPS/TV/hinge/lowpass eklendi
    3. abc_counterfactual.py — generate() tamamen yeniden yazıldı

  Çalıştırma:
    python3 train_abc_counterfactual.py \\
        --ham-checkpoint results/run_01_xai_dermoscopy/02_classifier_training/best_model.pth \\
        --abc-checkpoint results/run_01_xai_dermoscopy/09_abc_regressor/checkpoints/best_abc_model.pth \\
        --mask-dir datas/HAM10000/segmentations \\
        --experiment-dir results/experiment_03_v6_visible_cf \\
        --n-images 10

  Beklenen iyileşme:
    L1: 0.016 → 0.05-0.10  (3-6x artış)
    SSIM: 0.998 → 0.95-0.98  (görünür fark)
    Sparsity: 21% → 30-50%  (daha geniş etki alanı)
    Pertürbasyon: Düşük frekanslı, uzamsal olarak tutarlı, renkli değişimler
""")


if __name__ == "__main__":
    main()
