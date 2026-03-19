#!/usr/bin/env python3
"""
test_color_fix.py
-----------------
HSV renk aralığı düzeltmesini doğrulama scripti.
Tam pipeline'ı çalıştırmadan önce 20 örnek görüntüde test eder.

Kullanım:
    python3 test_color_fix.py
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# ── Eski aralıklar (hatalı) ──────────────────
OLD_COLORS = {
    "black"       : {"h": (0,   20),  "s": (0,   80),  "v": (0,   60)},
    "dark_brown"  : {"h": (10,  25),  "s": (50,  255), "v": (30,  100)},
    "light_brown" : {"h": (15,  35),  "s": (30,  180), "v": (100, 200)},
    "red"         : {"h": (0,   15),  "s": (80,  255), "v": (80,  255)},
    "blue_gray"   : {"h": (100, 140), "s": (20,  150), "v": (50,  180)},
    "white"       : {"h": (0,   180), "s": (0,   40),  "v": (200, 255)},
}
OLD_THRESHOLD = 0.05

# ── Yeni aralıklar (düzeltilmiş) ─────────────
NEW_COLORS = {
    "black"       : {"h": (0,   180), "s": (0,   255), "v": (0,    80)},
    "dark_brown"  : {"h": (5,    25), "s": (30,  255), "v": (30,  160)},
    "light_brown" : {"h": (8,    35), "s": (15,  255), "v": (140, 230)},
    "red"         : {"h": (0,    15), "s": (50,  255), "v": (60,  255),
                     "h_wrap": (160, 180)},
    "blue_gray"   : {"h": (85,  150), "s": (10,  180), "v": (50,  210)},
    "white"       : {"h": (0,   180), "s": (0,    50), "v": (180, 255)},
}
NEW_THRESHOLD = 0.03


def count_colors(hsv_pixels, colors_dict, threshold):
    """Verilen HSV aralıklarıyla renk sayısını hesapla."""
    n = len(hsv_pixels)
    h = hsv_pixels[:, 0].astype(int)
    s = hsv_pixels[:, 1].astype(int)
    v = hsv_pixels[:, 2].astype(int)

    detected = {}
    for name, ranges in colors_dict.items():
        h_lo, h_hi = ranges["h"]
        s_lo, s_hi = ranges["s"]
        v_lo, v_hi = ranges["v"]

        in_h = (h >= h_lo) & (h <= h_hi)
        if "h_wrap" in ranges:
            hw_lo, hw_hi = ranges["h_wrap"]
            in_h = in_h | ((h >= hw_lo) & (h <= hw_hi))

        in_s = (s >= s_lo) & (s <= s_hi)
        in_v = (v >= v_lo) & (v <= v_hi)

        frac = (in_h & in_s & in_v).sum() / n
        detected[name] = frac

    n_colors = sum(1 for f in detected.values() if f >= threshold)
    C = max(0.0, (n_colors - 1) / 5.0)
    return n_colors, C, detected


def main():
    img_dir  = Path("datas/HAM10000/HAM10000")
    mask_dir = Path("datas/HAM10000/segmentations")

    # Alternatif dizinler
    if not img_dir.exists():
        for alt in [Path("datas/HAM10000/HAM10000_images_part_1"),
                    Path("datas/HAM10000/HAM10000_images_part_2")]:
            if alt.exists():
                img_dir = alt
                break

    if not img_dir.exists():
        print(f"HATA: Görüntü dizini bulunamadı")
        return

    samples = sorted(img_dir.glob("ISIC_*.jpg"))[:20]

    print("=" * 90)
    print("  HSV Renk Aralığı Düzeltme Testi — Eski vs Yeni")
    print("=" * 90)
    print(f"\n  {'Image':<16} {'Eski':>18}         {'Yeni':>18}         Tespit Edilen Renkler (yeni)")
    print(f"  {'':16} {'#renk':>6} {'C_ip':>6}     {'#renk':>6} {'C_ip':>6}")
    print("  " + "─" * 85)

    old_c_list = []
    new_c_list = []

    for img_path in samples:
        iid = img_path.stem
        mask_path = mask_dir / f"{iid}_segmentation.png"
        if not mask_path.exists():
            continue

        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask_bool = (mask > 127)

        if mask_bool.sum() < 50:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv[mask_bool]

        old_n, old_c, _ = count_colors(hsv_pixels, OLD_COLORS, OLD_THRESHOLD)
        new_n, new_c, new_det = count_colors(hsv_pixels, NEW_COLORS, NEW_THRESHOLD)

        old_c_list.append(old_c)
        new_c_list.append(new_c)

        # Yeni ile tespit edilen renkler
        detected_names = [n for n, f in new_det.items() if f >= NEW_THRESHOLD]

        print(f"  {iid:<16} {old_n:>6} {old_c:>6.2f}     {new_n:>6} {new_c:>6.2f}     {', '.join(detected_names)}")

    print("\n" + "─" * 90)
    if old_c_list:
        print(f"  ESKİ C_ip ortalaması : {np.mean(old_c_list):.4f}")
        print(f"  YENİ C_ip ortalaması : {np.mean(new_c_list):.4f}")
        print(f"  İyileşme             : +{np.mean(new_c_list) - np.mean(old_c_list):.4f}")
        print(f"  Eski C_ip == 0 oranı : {sum(1 for c in old_c_list if c == 0) / len(old_c_list) * 100:.0f}%")
        print(f"  Yeni C_ip == 0 oranı : {sum(1 for c in new_c_list if c == 0) / len(new_c_list) * 100:.0f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
