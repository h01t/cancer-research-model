"""Tests for the final SSL-closure summary and no-freeze baseline sweep."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def write_metrics(exp_dir: Path, val_auc: float, test_auc: float, threshold: float = 0.5) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "val_metrics.yaml").write_text(
        yaml.safe_dump({"auc": val_auc, "accuracy": 0.7, "decision_threshold": threshold})
    )
    (exp_dir / "test_metrics.yaml").write_text(
        yaml.safe_dump(
            {
                "auc": test_auc,
                "accuracy": 0.7,
                "sensitivity": 0.75,
                "specificity": 0.65,
                "decision_threshold": threshold,
            }
        )
    )


class TestSSLClosureSummary:
    def test_closure_summary_prefers_nofreeze_supervised(self, tmp_path):
        rescue_dir = tmp_path / "results_rescue"
        followup_dir = tmp_path / "results_followup"
        nofreeze_dir = tmp_path / "results_nofreeze_baseline"
        output_dir = tmp_path / "results_closure"

        for seed, val_auc in [(42, 0.61), (43, 0.62), (44, 0.60)]:
            write_metrics(rescue_dir / f"supervised_100_seed{seed}", val_auc, 0.58)
            write_metrics(rescue_dir / f"fixmatch_100_seed{seed}", 0.66, 0.55)
            write_metrics(rescue_dir / f"supervised_250_seed{seed}", 0.71, 0.69)
            write_metrics(rescue_dir / f"fixmatch_250_seed{seed}", 0.68, 0.58)

        for seed, val_auc in [(42, 0.75), (43, 0.76)]:
            write_metrics(followup_dir / f"supervised_nofreeze_100_seed{seed}", val_auc, 0.66)
        write_metrics(nofreeze_dir / "supervised_nofreeze_100_seed44", 0.74, 0.66)

        for seed, val_auc in [(42, 0.82), (43, 0.87)]:
            write_metrics(followup_dir / f"supervised_nofreeze_250_seed{seed}", val_auc, 0.74)
        write_metrics(nofreeze_dir / "supervised_nofreeze_250_seed44", 0.81, 0.74)

        for seed, val_auc in [(42, 0.87), (43, 0.84), (44, 0.85)]:
            write_metrics(nofreeze_dir / f"supervised_nofreeze_500_seed{seed}", val_auc, 0.75)

        for seed, val_auc in [(42, 0.63), (43, 0.62), (44, 0.61)]:
            write_metrics(followup_dir / f"mean_teacher_100_seed{seed}", val_auc, 0.53)

        for seed, val_auc in [(42, 0.60), (43, 0.77), (44, 0.60)]:
            write_metrics(followup_dir / f"mean_teacher_250_seed{seed}", val_auc, 0.63)

        result = subprocess.run(
            [
                sys.executable,
                "scripts/summarize_ssl_closure.py",
                "--rescue_dir",
                str(rescue_dir),
                "--followup_dir",
                str(followup_dir),
                "--nofreeze_dir",
                str(nofreeze_dir),
                "--output_dir",
                str(output_dir),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )

        summary_text = (output_dir / "ssl_closure_summary.txt").read_text()
        assert "Official baseline: supervised no-freeze." in summary_text
        assert "The official 500-label no-freeze baseline currently has mean val AUC" in summary_text
        assert "close the vanilla SSL chapter for now" in summary_text
        assert "supervised_nofreeze" in summary_text
        assert "ssl_closure_summary.csv" not in result.stdout


class TestNoFreezeBaselineSweep:
    def test_nofreeze_baseline_sweep_dry_run_and_skip_completed(self, tmp_path):
        rescue_dir = tmp_path / "results_rescue"
        followup_dir = tmp_path / "results_followup"
        output_dir = tmp_path / "results_nofreeze_baseline"

        write_metrics(rescue_dir / "supervised_100_seed42", 0.61, 0.58)
        write_metrics(rescue_dir / "fixmatch_100_seed42", 0.66, 0.55)
        write_metrics(followup_dir / "supervised_nofreeze_100_seed42", 0.75, 0.66)
        write_metrics(output_dir / "supervised_nofreeze_500_seed42", 0.82, 0.74)

        result = subprocess.run(
            [
                "bash",
                "run_nofreeze_baseline_sweep.sh",
                "--rescue_dir",
                str(rescue_dir),
                "--followup_dir",
                str(followup_dir),
                "--output",
                str(output_dir),
                "--dry_run",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )

        log_text = (output_dir / "nofreeze_baseline.log").read_text()
        assert "SKIP: supervised_nofreeze_500_seed42 already completed" in log_text
        assert "DRY RUN: not executing supervised_nofreeze_100_seed44" in log_text
        assert "DRY RUN: not executing supervised_nofreeze_250_seed44" in log_text
        assert "SSL Chapter Closure Summary" in result.stdout
