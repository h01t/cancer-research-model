"""Tests for supervised summary tooling."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def _write_run(run_dir: Path, config: dict, val_auc: float, test_auc: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "val_metrics.yaml").write_text(yaml.safe_dump({"auc": val_auc}))
    (run_dir / "test_metrics.yaml").write_text(
        yaml.safe_dump(
            {
                "auc": test_auc,
                "sensitivity": 0.7,
                "specificity": 0.6,
                "decision_threshold": 0.5,
            }
        )
    )


def test_summarize_supervised_script(tmp_path, base_config):
    results_dir = tmp_path / "results"
    config_a = base_config | {
        "experiment": {"method": "supervised", "seed": 42},
        "model": base_config["model"] | {"name": "efficientnet-b0"},
        "dataset": base_config["dataset"] | {"image_size": 512, "labeled_subset_size": 500},
    }
    config_b = base_config | {
        "experiment": {"method": "supervised", "seed": 43},
        "model": base_config["model"] | {"name": "efficientnet-b0"},
        "dataset": base_config["dataset"] | {"image_size": 512, "labeled_subset_size": 500},
    }

    _write_run(results_dir / "run_a", config_a, val_auc=0.80, test_auc=0.75)
    _write_run(results_dir / "run_b", config_b, val_auc=0.82, test_auc=0.77)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_supervised.py",
            "--results_dir",
            str(results_dir),
            "--output_dir",
            str(results_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    summary_path = results_dir / "supervised_summary.txt"
    grouped_path = results_dir / "supervised_summary_grouped.csv"
    assert summary_path.exists()
    assert grouped_path.exists()
    summary_text = summary_path.read_text()
    assert "Best grouped candidate so far" in summary_text
