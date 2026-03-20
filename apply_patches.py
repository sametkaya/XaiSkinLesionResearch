#!/usr/bin/env python3
"""
apply_patches.py
----------------
Tüm pipeline iyileştirmelerini otomatik uygular.

Kullanım:
    cd /mnt/c/Users/samet/Documents/PycharmProjects/XaiSkinLesionResearch
    python3 apply_patches.py

Bu script şu dosyaları yerinde (in-place) günceller:
  1. src/config.py              — classifier eğitim ayarları + dizin yapısı
  2. src/abc/config_abc.py      — renk aralıkları + CF threshold
  3. src/abc/abc_ip_scorer.py   — h_wrap desteği
  4. src/abc/ham10000_scorer.py  — mask_source takibi + detaylı result.txt
  5. src/explainers/abc_counterfactual.py — sample filtre + görselleştirme + Adam

Her değişiklik öncesi yedek alınır (.bak uzantısı).
"""

import re
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
BACKUP_SUFFIX = ".bak_pre_v5"


def backup(path: Path):
    """Yedek al (varsa atla)."""
    bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  ✓ Yedek: {bak.name}")


def patch_file(path: Path, replacements: list):
    """Dosyada (old, new) çiftlerini uygula."""
    backup(path)
    text = path.read_text(encoding="utf-8")
    for old, new, desc in replacements:
        if old in text:
            text = text.replace(old, new, 1)
            print(f"  ✓ {desc}")
        else:
            print(f"  ⚠ Bulunamadı: {desc}")
    path.write_text(text, encoding="utf-8")


def main():
    print("\n" + "=" * 70)
    print("  Pipeline v5 — Tüm Yamaları Uygula")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # 1. src/config.py
    # ═══════════════════════════════════════════
    print("\n[1/5] src/config.py")
    config_py = PROJECT / "src" / "config.py"

    patch_file(config_py, [
        # 1a. Patience artır
        (
            "EARLY_STOP_PATIENCE= 30",
            "EARLY_STOP_PATIENCE= 50    # v5: SGDR T0=40 cycle'ını tamamlasın",
            "EARLY_STOP_PATIENCE 30→50"
        ),
        # 1b. Epoch artır
        (
            "NUM_EPOCHS         = 200",
            "NUM_EPOCHS         = 300   # v5: daha fazla SGDR restart döngüsü",
            "NUM_EPOCHS 200→300"
        ),
        # 1c. IMAGE_DIRS — tek klasör yapısı ekle
        (
            '    HAM10000_DIR / "HAM10000_images_part_2",\n]',
            '    HAM10000_DIR / "HAM10000_images_part_2",\n'
            '    HAM10000_DIR / "HAM10000",                  # v5: tek klasör yapısı\n]',
            "IMAGE_DIRS += HAM10000/HAM10000"
        ),
        # 1d. SEGMENTATION_DIR ekle
        (
            'METADATA_CSV = HAM10000_DIR / "HAM10000_metadata.csv"',
            'METADATA_CSV      = HAM10000_DIR / "HAM10000_metadata.csv"\n'
            'SEGMENTATION_DIR  = HAM10000_DIR / "segmentations"          # v5: tam maskeler',
            "SEGMENTATION_DIR eklendi"
        ),
    ])

    # ═══════════════════════════════════════════
    # 2. src/abc/config_abc.py
    # ═══════════════════════════════════════════
    print("\n[2/5] src/abc/config_abc.py")
    config_abc = PROJECT / "src" / "abc" / "config_abc.py"

    patch_file(config_abc, [
        # 2a. Color threshold
        (
            "IP_COLOR_THRESHOLD      = 0.05",
            "IP_COLOR_THRESHOLD      = 0.03  # v5: nadir renkler için daha hassas",
            "IP_COLOR_THRESHOLD 0.05→0.03"
        ),
        # 2b. DERMOSCOPIC_COLORS — tam blok değişimi
        (
            '''DERMOSCOPIC_COLORS = {
    "black"       : {"h": (0,   20),  "s": (0,   80),  "v": (0,   60)},
    "dark_brown"  : {"h": (10,  25),  "s": (50,  255), "v": (30,  100)},
    "light_brown" : {"h": (15,  35),  "s": (30,  180), "v": (100, 200)},
    "red"         : {"h": (0,   15),  "s": (80,  255), "v": (80,  255)},
    "blue_gray"   : {"h": (100, 140), "s": (20,  150), "v": (50,  180)},
    "white"       : {"h": (0,   180), "s": (0,   40),  "v": (200, 255)},
}''',
            '''DERMOSCOPIC_COLORS = {
    # v5: HSV aralıkları düzeltildi — önceki aralıklar C_ip ≈ 0 üretiyordu
    # Argenziano et al. (1998) + HAM10000 piksel dağılımı doğrulaması
    "black"       : {"h": (0,   180), "s": (0,   255), "v": (0,    80)},
    "dark_brown"  : {"h": (5,    25), "s": (30,  255), "v": (30,  160)},
    "light_brown" : {"h": (8,    35), "s": (15,  255), "v": (140, 230)},
    "red"         : {"h": (0,    15), "s": (50,  255), "v": (60,  255),
                     "h_wrap": (160, 180)},
    "blue_gray"   : {"h": (85,  150), "s": (10,  180), "v": (50,  210)},
    "white"       : {"h": (0,   180), "s": (0,    50), "v": (180, 255)},
}''',
            "DERMOSCOPIC_COLORS aralıkları düzeltildi"
        ),
        # 2c. CF confidence threshold
        (
            "ABC_CF_CONFIDENCE_THRES  = 0.85",
            "ABC_CF_CONFIDENCE_THRES  = 0.90    # v5: daha yüksek hedef güven",
            "ABC_CF_CONFIDENCE_THRES 0.85→0.90"
        ),
    ])

    # ═══════════════════════════════════════════
    # 3. src/abc/abc_ip_scorer.py — h_wrap desteği
    # ═══════════════════════════════════════════
    print("\n[3/5] src/abc/abc_ip_scorer.py")
    ip_scorer = PROJECT / "src" / "abc" / "abc_ip_scorer.py"
    backup(ip_scorer)
    text = ip_scorer.read_text(encoding="utf-8")

    # compute_color içindeki döngüyü güncelle
    old_loop = """    colors_detected = 0
    for color_name, ranges in DERMOSCOPIC_COLORS.items():
        h_lo, h_hi = ranges["h"]
        s_lo, s_hi = ranges["s"]
        v_lo, v_hi = ranges["v"]

        h = hsv_pixels[:, 0].astype(int)
        s = hsv_pixels[:, 1].astype(int)
        v = hsv_pixels[:, 2].astype(int)

        in_h = (h >= h_lo) & (h <= h_hi)
        in_s = (s >= s_lo) & (s <= s_hi)
        in_v = (v >= v_lo) & (v <= v_hi)

        frac = (in_h & in_s & in_v).sum() / n_pixels
        if frac >= min_frac:
            colors_detected += 1"""

    new_loop = """    # v5: h, s, v döngü dışına taşındı + h_wrap desteği eklendi
    h = hsv_pixels[:, 0].astype(int)
    s = hsv_pixels[:, 1].astype(int)
    v = hsv_pixels[:, 2].astype(int)

    colors_detected = 0
    for color_name, ranges in DERMOSCOPIC_COLORS.items():
        h_lo, h_hi = ranges["h"]
        s_lo, s_hi = ranges["s"]
        v_lo, v_hi = ranges["v"]

        in_h = (h >= h_lo) & (h <= h_hi)

        # Handle hue wrap-around for red/pink (H near 0 and near 180)
        if "h_wrap" in ranges:
            hw_lo, hw_hi = ranges["h_wrap"]
            in_h = in_h | ((h >= hw_lo) & (h <= hw_hi))

        in_s = (s >= s_lo) & (s <= s_hi)
        in_v = (v >= v_lo) & (v <= v_hi)

        frac = (in_h & in_s & in_v).sum() / n_pixels
        if frac >= min_frac:
            colors_detected += 1"""

    if old_loop in text:
        text = text.replace(old_loop, new_loop, 1)
        print("  ✓ compute_color: h_wrap desteği eklendi")
    else:
        print("  ⚠ compute_color döngüsü bulunamadı — zaten güncel olabilir")
    ip_scorer.write_text(text, encoding="utf-8")

    # ═══════════════════════════════════════════
    # 4. src/abc/ham10000_scorer.py — mask_source
    # ═══════════════════════════════════════════
    print("\n[4/5] src/abc/ham10000_scorer.py")
    scorer = PROJECT / "src" / "abc" / "ham10000_scorer.py"
    backup(scorer)
    text = scorer.read_text(encoding="utf-8")

    # 4a. run() içinde mask_source takibi ekle
    old_record = '''                records.append({
                    "image_id"            : str(image_ids[i]),
                    "dx"                  : str(dxs[i]),
                    "A_dl"                : round(A_dl, 4),
                    "B_dl"                : round(B_dl, 4),
                    "C_dl"                : round(C_dl, 4),
                    "A_ip"                : round(A_ip, 4),
                    "B_ip"                : round(B_ip, 4),
                    "C_ip"                : round(C_ip, 4),
                    "A_mean"              : round((A_dl + A_ip) / 2, 4),
                    "B_mean"              : round((B_dl + B_ip) / 2, 4),
                    "C_mean"              : round((C_dl + C_ip) / 2, 4),
                })'''

    new_record = '''                # v5: mask_source takibi
                mask_src = "expert" if mask is not None else "otsu"

                records.append({
                    "image_id"            : str(image_ids[i]),
                    "dx"                  : str(dxs[i]),
                    "A_dl"                : round(A_dl, 4),
                    "B_dl"                : round(B_dl, 4),
                    "C_dl"                : round(C_dl, 4),
                    "A_ip"                : round(A_ip, 4),
                    "B_ip"                : round(B_ip, 4),
                    "C_ip"                : round(C_ip, 4),
                    "A_mean"              : round((A_dl + A_ip) / 2, 4),
                    "B_mean"              : round((B_dl + B_ip) / 2, 4),
                    "C_mean"              : round((C_dl + C_ip) / 2, 4),
                    "mask_source"         : mask_src,
                })'''

    if old_record in text:
        text = text.replace(old_record, new_record, 1)
        print("  ✓ run(): mask_source sütunu eklendi")
    else:
        print("  ⚠ records.append bloğu bulunamadı — zaten güncel olabilir")

    # 4b. segmentations string güncelle
    old_seg = '"segmentations"  : "ISIC 2018 Task 1 masks (Otsu fallback)"'
    new_seg = '"segmentations"  : "Full expert masks + Otsu fallback"'
    if old_seg in text:
        text = text.replace(old_seg, new_seg, 1)
        print("  ✓ result.txt segmentation açıklaması güncellendi")

    scorer.write_text(text, encoding="utf-8")

    # ═══════════════════════════════════════════
    # 5. src/explainers/abc_counterfactual.py
    # ═══════════════════════════════════════════
    print("\n[5/5] src/explainers/abc_counterfactual.py")
    cf_file = PROJECT / "src" / "explainers" / "abc_counterfactual.py"
    backup(cf_file)
    text = cf_file.read_text(encoding="utf-8")

    # 5a. _collect_samples — min_source_prob filtresi
    old_collect = """        samples = []
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
        return samples"""

    new_collect = """        # v5: min_source_prob filtresi — yüksek güvenli örnekler
        min_source_prob = 0.80
        samples = []
        self.explainer.clf.eval()
        with torch.no_grad():
            for images, labels in self.loader:
                imgs   = images.to(self.device)
                logits = self.explainer.clf(imgs)
                probs  = torch.softmax(logits, dim=1).cpu()
                preds  = probs.argmax(dim=1)
                for i, (img, lbl, prd) in enumerate(
                    zip(images, labels, preds)
                ):
                    if (int(lbl) == src_class
                            and int(prd) == src_class
                            and float(probs[i, src_class]) >= min_source_prob):
                        samples.append((img, int(lbl), None))
                    if len(samples) >= n:
                        break
                if len(samples) >= n:
                    break
        if len(samples) < n:
            print(f"  ⚠ {len(samples)}/{n} samples with prob≥{min_source_prob} for class {src_class}")
        return samples"""

    if old_collect in text:
        text = text.replace(old_collect, new_collect, 1)
        print("  ✓ _collect_samples: min_source_prob=0.80 filtresi eklendi")
    else:
        print("  ⚠ _collect_samples bloğu bulunamadı — zaten güncel olabilir")

    # 5b. generate() — manual SGD → Adam
    old_optim = """        n_iter = 0
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

    new_optim = """        # v5: Adam optimizer (manual SGD yerine) — daha hızlı yakınsama
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

    if old_optim in text:
        text = text.replace(old_optim, new_optim, 1)
        print("  ✓ generate(): manual SGD → Adam optimizer")
    else:
        print("  ⚠ generate() optim bloğu bulunamadı — zaten güncel olabilir")

    # 5c. _save_panels — iyileştirilmiş görselleştirme
    old_heatmap = '''            axes[row, 3].imshow(
                abs_np / (abs_np.max() + 1e-8),
                cmap="hot",
            )'''

    new_heatmap = '''            # v5: percentile-based scaling + inferno colormap
            abs_delta = np.abs(diff_np).max(axis=2)
            p99 = np.percentile(abs_delta[abs_delta > 0], 99) if (abs_delta > 0).any() else 0.01
            axes[row, 3].imshow(
                abs_delta,
                cmap="inferno",
                vmin=0, vmax=max(p99, 0.005),
            )'''

    if old_heatmap in text:
        text = text.replace(old_heatmap, new_heatmap, 1)
        print("  ✓ _save_panels: inferno + percentile scaling")
    else:
        print("  ⚠ _save_panels heatmap bloğu bulunamadı — zaten güncel olabilir")

    # 5d. Diff panel: min-max → RdBu_r symmetric
    old_diff = '''            axes[row, 2].imshow(
                (diff_np - diff_np.min()) /
                (diff_np.max() - diff_np.min() + 1e-8)
            )'''

    new_diff = '''            # v5: signed difference with RdBu_r colormap
            diff_gray = diff_np.mean(axis=2)
            vmax_d = max(abs(diff_gray.min()), abs(diff_gray.max()), 0.01)
            im_d = axes[row, 2].imshow(
                diff_gray, cmap="RdBu_r",
                vmin=-vmax_d, vmax=vmax_d,
            )
            plt.colorbar(im_d, ax=axes[row, 2], fraction=0.046, pad=0.04)'''

    if old_diff in text:
        text = text.replace(old_diff, new_diff, 1)
        print("  ✓ _save_panels: RdBu_r + colorbar")
    else:
        print("  ⚠ _save_panels diff bloğu bulunamadı — zaten güncel olabilir")

    cf_file.write_text(text, encoding="utf-8")

    # ═══════════════════════════════════════════
    # Özet
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Tüm yamalar uygulandı!")
    print("=" * 70)
    print("""
  Sonraki adımlar:

  ┌─────────────────────────────────────────────────┐
  │  Faz 1: Full classifier pipeline                │
  │  python3 main.py                                │
  │  (Tahmini: 3-6 saat — RTX 3080)                │
  └─────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────┐
  │  Faz 2: ABC skorlama                            │
  │  python3 score_ham10000.py \\                     │
  │    --abc-checkpoint <yeni_run>/09_.../best_...  │
  │    --mask-dir datas/HAM10000/segmentations \\    │
  │    --experiment-dir results/experiment_01_v5 \\  │
  │    --no-hdf5                                    │
  └─────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────┐
  │  Faz 3: ABC counterfactual                      │
  │  python3 train_abc_counterfactual.py \\           │
  │    --ham-checkpoint <yeni_run>/02_.../best_...  │
  │    --abc-checkpoint <yeni_run>/09_.../best_...  │
  │    --mask-dir datas/HAM10000/segmentations \\    │
  │    --experiment-dir results/experiment_02_v5 \\  │
  │    --n-images 10                                │
  └─────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
