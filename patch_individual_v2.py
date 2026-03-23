#!/usr/bin/env python3
"""
patch_individual_v2.py — Update abc_counterfactual.py to call
generate_individual_panels with all 4 modes' results.
"""
from pathlib import Path

def main():
    cf = Path("src/explainers/abc_counterfactual.py")
    text = cf.read_text(encoding="utf-8")

    # 1. Remove old per-mode call
    old_call = """                    # v8: Individual per-image panels
                    indiv_dir = pairs_dir / f"{src_name}_to_{tgt_name}_individual"
                    generate_individual_panels(
                        mode_records,
                        self.explainer.clf,
                        self.explainer.abc_reg,
                        self.device,
                        indiv_dir,
                        src_name, tgt_name,
                        mode="ABC",
                    )"""

    if old_call in text:
        text = text.replace(old_call, "")
        print("OK: Removed old per-mode call")
    else:
        print("WARN: Old call not found (may already be removed)")

    # 2. Find the place after all modes loop to add new call
    # Look for the line after the modes loop that starts summary
    # The pattern: after "for mode in ABLATION_MODES:" loop ends,
    # there should be a summary section.
    # We need to add the call AFTER the for-mode loop but BEFORE summary.

    # Find "# ── Per-pair summary" or the line with pair_stats
    # Actually, let's find where pair_records dict is built
    # Better: add after the for-mode loop. Find the pattern:

    marker = """            # ── Per-pair summary"""
    if marker not in text:
        # Try alternative markers
        marker = """            # Per-pair summary"""
    if marker not in text:
        # Look for pair_stats or similar
        marker = """            pair_stats"""

    # Let's use a different approach: add a dict to collect mode records,
    # then call after the loop

    # Add mode_records_all dict before the for-mode loop
    old_loop_start = """            for mode in ABLATION_MODES:
                mode_records = []"""

    new_loop_start = """            mode_records_all = {}
            for mode in ABLATION_MODES:
                mode_records = []"""

    if old_loop_start in text:
        text = text.replace(old_loop_start, new_loop_start, 1)
        print("OK: Added mode_records_all dict")
    else:
        print("WARN: Loop start not found")

    # After mode_records is populated (before ablation row print),
    # store in dict
    old_ablation_row = """                # Ablation row
                print("""

    new_ablation_row = """                mode_records_all[mode] = mode_records
                # Ablation row
                print("""

    if old_ablation_row in text:
        text = text.replace(old_ablation_row, new_ablation_row, 1)
        print("OK: Added mode_records_all[mode] = mode_records")
    else:
        print("WARN: Ablation row marker not found")

    # Find end of the for-mode loop to add individual panels call
    # The for-mode loop ends, then there's pair timing or summary
    # Look for pattern after the loop

    # Find the indentation change after the for-mode block
    # After "for mode in ABLATION_MODES:" block ends with pair summary
    # Let's find "pair_time" or next "# ──" at pair level

    # Search for pattern that comes right after the modes loop
    import re
    # Find all lines with "            # ──" (pair-level comments)
    # The one after the modes loop should be pair summary

    # Add the call before pair timing
    old_pair_end = """            pair_time = time.time() - pair_t0"""
    new_pair_end = """            # v8: Individual panels (all modes combined)
            indiv_dir = pairs_dir / f"{src_name}_to_{tgt_name}_individual"
            generate_individual_panels(
                mode_records_all,
                self.explainer.clf,
                self.explainer.abc_reg,
                self.device,
                indiv_dir,
                src_name, tgt_name,
                n_images=len(mode_records_all.get("ABC", [])),
            )

            pair_time = time.time() - pair_t0"""

    if old_pair_end in text:
        text = text.replace(old_pair_end, new_pair_end, 1)
        print("OK: Added new individual panels call after all modes")
    else:
        print("WARN: pair_time marker not found")
        # Try alternative
        old_alt = """            pair_time"""
        if old_alt in text:
            print("  Found 'pair_time' without full match — manual edit needed")

    cf.write_text(text, encoding="utf-8")
    print("\nDone! Now:")
    print("  1. cp ~/Downloads/individual_panels.py src/explainers/individual_panels.py")
    print("  2. python3 train_abc_counterfactual.py ...")

if __name__ == "__main__":
    main()
