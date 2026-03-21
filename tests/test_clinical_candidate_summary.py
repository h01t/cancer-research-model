"""Tests for clinical candidate summary tooling."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml


def _write_clinical_summary(run_dir: Path, run_name: str, config_name: str, seed: int, val_auc: float, test_auc: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_dir": f"/tmp/{run_name}",
        "run_name": run_name,
        "config_name": config_name,
        "checkpoint": f"/tmp/{run_name}/best_model.pth",
        "backbone": "efficientnet-b0",
        "image_size": 512,
        "optimizer": "adam",
        "label_smoothing": 0.1 if "ls" in config_name else 0.0,
        "seed": seed,
        "decision_threshold": 0.5,
        "val_metrics": {
            "auc": val_auc,
            "pr_auc": 0.7,
            "brier_score": 0.2,
            "ece": 0.05,
        },
        "test_metrics": {
            "auc": test_auc,
            "pr_auc": 0.68,
            "brier_score": 0.21,
            "ece": 0.06,
            "sensitivity": 0.7,
            "specificity": 0.65,
        },
        "grouped_metrics": [
            {"group_name": "exam_id", "auc": test_auc - 0.02, "pr_auc": 0.66}
        ],
    }
    (run_dir / "clinical_summary.yaml").write_text(yaml.safe_dump(summary))


def test_summarize_clinical_candidates(tmp_path):
    results_dir = tmp_path / "clinical"
    _write_clinical_summary(
        results_dir / "default_nofreeze_res512_seed42",
        "default_nofreeze_res512_seed42",
        "default_nofreeze_res512",
        42,
        val_auc=0.84,
        test_auc=0.75,
    )
    _write_clinical_summary(
        results_dir / "default_nofreeze_aug_safe_seed43",
        "default_nofreeze_aug_safe_seed43",
        "default_nofreeze_aug_safe",
        43,
        val_auc=0.85,
        test_auc=0.77,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_clinical_candidates.py",
            "--results_dir",
            str(results_dir),
            "--output_dir",
            str(results_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    summary_path = results_dir / "clinical_candidates_summary.txt"
    grouped_path = results_dir / "clinical_candidates_grouped.csv"
    assert summary_path.exists()
    assert grouped_path.exists()
    assert "Clinical Candidate Summary" in summary_path.read_text()

    with open(grouped_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["config_name"] for row in rows} == {
        "default_nofreeze_res512",
        "default_nofreeze_aug_safe",
    }
