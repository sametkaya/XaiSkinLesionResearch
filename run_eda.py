"""
run_eda.py
----------
Exploratory Data Analysis (EDA) for HAM10000.

Generates:
  - Class distribution plot
  - Split distribution plot
  - Sample image grid
  - Metadata statistics (age, sex, localization)
  - Summary statistics written to result.txt

Run this script first, before model training, to verify the data
is loaded correctly and to understand the dataset characteristics.

Usage:
    python run_eda.py
"""

import time
import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_metadata, set_seed, stratified_patient_split
from src.utils.visualization import (
    plot_class_distribution,
    plot_split_distribution,
    plot_sample_images,
    plot_metadata_stats,
)
from src.utils.result_manager import ResultManager


def run_eda() -> None:
    """Execute the full EDA pipeline."""
    set_seed(config.RANDOM_SEED)
    config.create_result_dirs()

    print("\n" + "=" * 60)
    print("  Exploratory Data Analysis — HAM10000")
    print("=" * 60)

    start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_metadata()
    train_df, val_df, test_df = stratified_patient_split(df)

    # ── Basic statistics ──────────────────────────────────────────────────
    class_counts = df["dx"].value_counts().to_dict()
    class_pcts   = (df["dx"].value_counts(normalize=True) * 100).round(2).to_dict()

    print("\n[EDA] Class distribution:")
    for cls, count in class_counts.items():
        full_name = config.CLASS_NAMES[cls]
        print(f"  {cls:8s} ({full_name:30s}): {count:5d}  ({class_pcts[cls]:.1f}%)")

    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    print(f"\n  Imbalance ratio (max/min): {imbalance_ratio:.1f}×")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[EDA] Generating plots …")

    plot_class_distribution(
        df, config.RESULT_EDA / "class_distribution.png"
    )
    plot_split_distribution(
        train_df, val_df, test_df,
        config.RESULT_EDA / "split_distribution.png"
    )
    plot_sample_images(
        df, config.RESULT_EDA / "sample_images.png", n_per_class=4
    )
    plot_metadata_stats(
        df, config.RESULT_EDA / "metadata_stats.png"
    )

    elapsed = time.time() - start

    # ── result.txt ─────────────────────────────────────────────────────────
    rm = ResultManager(config.RESULT_EDA)
    rm.write_result(
        experiment_name="Exploratory Data Analysis",
        conditions={
            "dataset"          : "HAM10000",
            "metadata_csv"     : str(config.METADATA_CSV),
            "image_directories": [str(d) for d in config.IMAGE_DIRS],
            "random_seed"      : config.RANDOM_SEED,
            "train_ratio"      : config.TRAIN_RATIO,
            "val_ratio"        : config.VAL_RATIO,
            "test_ratio"       : config.TEST_RATIO,
        },
        statistics={
            "total_images"      : len(df),
            "num_classes"       : config.NUM_CLASSES,
            "class_counts"      : class_counts,
            "class_percentages" : class_pcts,
            "imbalance_ratio"   : round(imbalance_ratio, 2),
            "train_size"        : len(train_df),
            "val_size"          : len(val_df),
            "test_size"         : len(test_df),
            "unique_patients"   : df["lesion_id"].nunique(),
            "age_mean"          : round(float(df["age"].mean()),   1),
            "age_std"           : round(float(df["age"].std()),    1),
            "age_min"           : float(df["age"].min()),
            "age_max"           : float(df["age"].max()),
            "sex_distribution"  : df["sex"].value_counts().to_dict(),
            "top_localizations" : df["localization"].value_counts().head(5).to_dict(),
            "elapsed_seconds"   : round(elapsed, 2),
        },
    )

    print(f"\n[EDA] Completed in {elapsed:.1f}s.  Outputs → {config.RESULT_EDA}")


if __name__ == "__main__":
    run_eda()
