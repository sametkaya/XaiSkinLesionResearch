"""
utils/result_manager.py
------------------------
Utility class for writing structured result.txt files.

Every experiment writes a single result.txt to its dedicated result folder.
The file records:
  - Experiment name
  - Timestamp
  - Experiment conditions (hyperparameters, dataset settings)
  - All statistics (scalars, lists, nested dicts)

This ensures every result is self-describing and reproducible.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class ResultManager:
    """
    Write experiment metadata and results to a structured result.txt file.

    Parameters
    ----------
    result_dir : Path
        Directory to write result.txt into.
    """

    def __init__(self, result_dir: Path):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def write_result(
        self,
        experiment_name: str,
        conditions: Dict[str, Any],
        statistics: Dict[str, Any],
    ) -> None:
        """
        Write a result.txt file for the given experiment.

        Parameters
        ----------
        experiment_name : str
            Human-readable name of the experiment.
        conditions : Dict[str, Any]
            Key-value pairs describing the experimental setup.
        statistics : Dict[str, Any]
            Key-value pairs of all result statistics.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines     = []

        # ── Header ────────────────────────────────────────────────────────
        lines.append("=" * 70)
        lines.append(f"  EXPERIMENT : {experiment_name}")
        lines.append(f"  TIMESTAMP  : {timestamp}")
        lines.append("=" * 70)
        lines.append("")

        # ── Experiment Conditions ─────────────────────────────────────────
        lines.append("EXPERIMENT CONDITIONS")
        lines.append("-" * 40)
        for key, val in conditions.items():
            lines.append(f"  {key:<30s}: {val}")
        lines.append("")

        # ── Statistics ────────────────────────────────────────────────────
        lines.append("RESULTS / STATISTICS")
        lines.append("-" * 40)
        self._format_dict(statistics, lines, indent=2)
        lines.append("")
        lines.append("=" * 70)

        out_path = self.result_dir / "result.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[ResultManager] Wrote {out_path}")

    def _format_dict(
        self,
        d: Dict[str, Any],
        lines: list,
        indent: int,
    ) -> None:
        """Recursively format a nested dictionary into result lines."""
        prefix = " " * indent
        for key, val in d.items():
            if isinstance(val, dict):
                lines.append(f"{prefix}{key}:")
                self._format_dict(val, lines, indent + 4)
            elif isinstance(val, (list, tuple)):
                lines.append(f"{prefix}{key:<30s}: {val}")
            else:
                lines.append(f"{prefix}{key:<30s}: {val}")
