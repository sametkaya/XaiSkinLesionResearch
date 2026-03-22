#!/usr/bin/env python3
"""
Integrate individual_panels.py into the pipeline.
Run from project root.
"""
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

def main():
    # 1. Copy module
    src = PROJECT / "individual_panels.py"
    dst = PROJECT / "src" / "explainers" / "individual_panels.py"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"OK: {dst}")
    else:
        print(f"NOT FOUND: {src}")
        return

    # 2. Patch abc_counterfactual.py
    cf = PROJECT / "src" / "explainers" / "abc_counterfactual.py"
    text = cf.read_text(encoding="utf-8")

    # Add import
    if "individual_panels" not in text:
        text = text.replace(
            "from src.explainers.abc_visualizer import save_abc_panel",
            "from src.explainers.abc_visualizer import save_abc_panel\nfrom src.explainers.individual_panels import generate_individual_panels",
            1
        )
        print("OK: import added")

    # Add call after save_abc_panel
    if "generate_individual_panels" not in text:
        old = '''                    save_abc_panel(
                        mode_records[:5],
                        self.explainer.abc_reg,
                        self.device,
                        pairs_dir / f"{src_name}_to_{tgt_name}_abc_clinical.png",
                        f"{src_name} → {tgt_name} | ABC Clinical Analysis",
                        max_rows=5,
                    )'''
        new = '''                    save_abc_panel(
                        mode_records[:5],
                        self.explainer.abc_reg,
                        self.device,
                        pairs_dir / f"{src_name}_to_{tgt_name}_abc_clinical.png",
                        f"{src_name} → {tgt_name} | ABC Clinical Analysis",
                        max_rows=5,
                    )
                    # v8: Individual per-image panels (20+ examples)
                    indiv_dir = pairs_dir / f"{src_name}_to_{tgt_name}_individual"
                    generate_individual_panels(
                        mode_records,
                        self.explainer.clf,
                        self.explainer.abc_reg,
                        self.device,
                        indiv_dir,
                        src_name, tgt_name,
                        mode="ABC",
                    )'''
        if old in text:
            text = text.replace(old, new, 1)
            print("OK: generate_individual_panels call added")
        else:
            print("WARN: save_abc_panel block not found")

    # Set n_images = 20
    cfg = PROJECT / "src" / "abc" / "config_abc.py"
    ct = cfg.read_text(encoding="utf-8")
    ct = ct.replace("ABC_CF_NUM_IMAGES        = 10", "ABC_CF_NUM_IMAGES        = 20")
    cfg.write_text(ct, encoding="utf-8")
    print("OK: ABC_CF_NUM_IMAGES = 20")

    cf.write_text(text, encoding="utf-8")
    print("\nDone! Run:")
    print("  python3 train_abc_counterfactual.py \\")
    print("    --ham-checkpoint results/run_03_xai_dermoscopy/02_classifier_training/best_model.pth \\")
    print("    --abc-checkpoint results/run_03_xai_dermoscopy/09_abc_regressor/checkpoints/best_abc_model.pth \\")
    print("    --mask-dir datas/HAM10000/segmentations \\")
    print("    --n-images 20")

if __name__ == "__main__":
    main()
