#!/usr/bin/env python3
"""
apply_8panel_patch.py
---------------------
Integrates the 8-panel publication-quality visualization into
abc_counterfactual.py and sets n_images=10.

Changes:
  1. Copy cf_visualizer.py to src/explainers/
  2. Add import in abc_counterfactual.py
  3. Add save_8panel_figure call after _save_panels
  4. Store mask in results dict for contour drawing
  5. Set mode_records[:10] for 10 examples per panel

Usage:
    cd XaiSkinLesionResearch
    python3 apply_8panel_patch.py
"""

import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def main():
    print("\n" + "=" * 60)
    print("  8-Panel Visualization Patch")
    print("=" * 60)

    # ═══════════════════════════════════════════
    # 1. Copy cf_visualizer.py
    # ═══════════════════════════════════════════
    src = PROJECT / "cf_visualizer.py"
    dst = PROJECT / "src" / "explainers" / "cf_visualizer.py"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  ✓ {dst.name} copied")
    else:
        print(f"  ⚠ {src} not found — copy manually")
        return

    # ═══════════════════════════════════════════
    # 2. Patch abc_counterfactual.py
    # ═══════════════════════════════════════════
    cf_path = PROJECT / "src" / "explainers" / "abc_counterfactual.py"
    text = cf_path.read_text(encoding="utf-8")
    changed = False

    # 2a. Add import (after existing imports)
    import_marker = "from src.utils.result_manager import ResultManager"
    import_new = """from src.utils.result_manager import ResultManager
from src.explainers.cf_visualizer import save_8panel_figure"""

    if "cf_visualizer" not in text:
        text = text.replace(import_marker, import_new, 1)
        print("  ✓ cf_visualizer import added")
        changed = True

    # 2b. Store mask in generate() result dict
    # Find where result dict is built and add mask
    old_result_mask = '"ssim"              : ssim_val,'
    new_result_mask = '''"ssim"              : ssim_val,
                "mask"              : soft_mask,    # for contour overlay'''
    
    if '"mask"' not in text and old_result_mask in text:
        text = text.replace(old_result_mask, new_result_mask, 1)
        print("  ✓ mask added to result dict")
        changed = True

    # 2c. Add 8-panel call after each _save_panels call
    # Find the pattern where _save_panels is called with mode_records
    old_save = '''                    self._save_panels(
                    mode_records[:10],'''
    
    # Check what the current slice is
    for slice_val in [":10", ":5", ":3"]:
        pattern = f"mode_records[{slice_val}]"
        if pattern in text:
            # Ensure it's [:10]
            if slice_val != ":10":
                text = text.replace(f"mode_records[{slice_val}]", "mode_records[:10]")
                print(f"  ✓ mode_records[{slice_val}] → mode_records[:10]")
                changed = True
            break

    # Add 8-panel call after _save_panels calls
    # Look for the pattern of _save_panels being called
    old_panel_block = '''                self._save_panels(
                    mode_records[:10],
                    per_class_dir / f"{src_name}_to_{tgt_name}_{mode}.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                )'''
    
    new_panel_block = '''                self._save_panels(
                    mode_records[:10],
                    per_class_dir / f"{src_name}_to_{tgt_name}_{mode}.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                )
                # v7: 8-panel publication-quality figure
                save_8panel_figure(
                    mode_records[:10],
                    self.explainer.clf,
                    self.device,
                    per_class_dir / f"{src_name}_to_{tgt_name}_{mode}_8panel.png",
                    f"{src_name} → {tgt_name} | mode={mode}",
                    max_rows=10,
                )'''

    if "8panel" not in text and old_panel_block in text:
        text = text.replace(old_panel_block, new_panel_block, 1)
        print("  ✓ save_8panel_figure call added (first location)")
        changed = True

    # Also add for the second _save_panels call (per_class_dir / all modes)
    # There are typically 2 calls — one per mode, one summary
    # Check for the summary panel call pattern
    old_summary = '''                self._save_panels(
                    mode_records[:10],
                    per_class_dir / f"{src_name}_to_{tgt_name}_{mode}_summary.png",'''
    
    # If there's a second call, add 8panel there too
    count_save_panels = text.count("self._save_panels(")
    count_8panel = text.count("save_8panel_figure(")
    
    if count_save_panels > count_8panel:
        # There are more _save_panels calls than 8panel calls
        # Find ALL _save_panels calls and add 8panel after each
        # (We already handled the first one above)
        pass  # The first call covers the main per-mode panels

    # 2d. Set n_images = 10 in config
    config_path = PROJECT / "src" / "abc" / "config_abc.py"
    config_text = config_path.read_text(encoding="utf-8")
    
    old_n = "ABC_CF_NUM_IMAGES        = 10"
    if old_n in config_text:
        print("  ✓ ABC_CF_NUM_IMAGES already 10")
    else:
        for old_val in ["= 5", "= 3", "= 15", "= 20"]:
            pattern = f"ABC_CF_NUM_IMAGES        {old_val}"
            if pattern in config_text:
                config_text = config_text.replace(pattern,
                    "ABC_CF_NUM_IMAGES        = 10")
                config_path.write_text(config_text, encoding="utf-8")
                print(f"  ✓ ABC_CF_NUM_IMAGES → 10")
                break

    if changed:
        cf_path.write_text(text, encoding="utf-8")
        print("  ✓ abc_counterfactual.py updated")

    # ═══════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Patch applied!")
    print("=" * 60)
    print("""
  Her sınıf çifti için 2 dosya üretilecek:
    {pair}_{mode}.png      — Mevcut 4-panel (eski format)
    {pair}_{mode}_8panel.png — 8-panel yayın kalitesi (YENİ)

  8 Panel Sütunları:
    1. Original + yeşil lezyon konturu
    2. Grad-CAM (turbo, α=0.4)
    3. Counterfactual + kontru
    4. Grad-CAM (CF üzerinde)
    5. Dikkat Farkı (GradCAM_orig − GradCAM_cf)
    6. İşaretli δ (RdBu_r)
    7. 10× Büyütülmüş δ
    8. |δ| Heatmap (inferno)

  Çalıştır:
    python3 train_abc_counterfactual.py \\
        --ham-checkpoint results/run_03_xai_dermoscopy/02_classifier_training/best_model.pth \\
        --abc-checkpoint results/run_03_xai_dermoscopy/09_abc_regressor/checkpoints/best_abc_model.pth \\
        --mask-dir datas/HAM10000/segmentations \\
        --n-images 10
""")


if __name__ == "__main__":
    main()
