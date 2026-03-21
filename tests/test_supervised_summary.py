"""Tests for supervised summary tooling."""

from __future__ import annotations

import csv
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


def test_summarize_supervised_script_separates_config_families(tmp_path, base_config):
    results_dir = tmp_path / "results"
    config_res512 = base_config | {
        "experiment": {"method": "supervised", "seed": 42},
        "model": base_config["model"] | {"name": "efficientnet-b0"},
        "dataset": base_config["dataset"] | {"image_size": 512, "labeled_subset_size": 500},
        "augmentation": {
            "weak": {
                "random_horizontal_flip": True,
                "random_vertical_flip": True,
                "random_rotation": 5,
                "color_jitter": 0.1,
            }
        },
    }
    config_aug_safe = base_config | {
        "experiment": {"method": "supervised", "seed": 43},
        "model": base_config["model"] | {"name": "efficientnet-b0"},
        "dataset": base_config["dataset"] | {"image_size": 512, "labeled_subset_size": 500},
        "augmentation": {
            "weak": {
                "random_horizontal_flip": True,
                "random_vertical_flip": False,
                "random_rotation": 3,
                "color_jitter": 0.0,
            }
        },
    }
    config_ls = base_config | {
        "experiment": {"method": "supervised", "seed": 44},
        "model": base_config["model"] | {"name": "efficientnet-b0"},
        "dataset": base_config["dataset"] | {"image_size": 512, "labeled_subset_size": 500},
        "training": base_config["training"] | {"label_smoothing": 0.1},
    }

    _write_run(results_dir / "default_nofreeze_res512_seed42", config_res512, val_auc=0.80, test_auc=0.75)
    _write_run(results_dir / "default_nofreeze_aug_safe_seed43", config_aug_safe, val_auc=0.82, test_auc=0.77)
    _write_run(results_dir / "default_nofreeze_ls_seed44", config_ls, val_auc=0.84, test_auc=0.76)

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
    candidate_path = results_dir / "candidate_comparison.csv"
    assert summary_path.exists()
    assert grouped_path.exists()
    assert candidate_path.exists()

    summary_text = summary_path.read_text()
    assert "Candidate Comparison" in summary_text
    assert "default_nofreeze_res512" in summary_text
    assert "default_nofreeze_aug_safe" in summary_text

    with open(grouped_path, newline="") as f:
        rows = list(csv.DictReader(f))
    config_names = {row["config_name"] for row in rows}
    assert "default_nofreeze_res512" in config_names
    assert "default_nofreeze_aug_safe" in config_names
    assert len(rows) == 3


def test_aug_safe_ls_config_combines_safe_aug_and_label_smoothing():
    config_path = Path("configs/default_nofreeze_aug_safe_ls.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["training"]["label_smoothing"] == 0.1
    assert config["model"]["name"] == "efficientnet-b0"
    assert config["dataset"]["image_size"] == 512
    assert config["training"]["optimizer"] == "adam"
    assert config["augmentation"]["weak"]["random_vertical_flip"] is False
    assert config["augmentation"]["weak"]["random_rotation"] == 3
    assert config["augmentation"]["weak"]["color_jitter"] == 0.0
