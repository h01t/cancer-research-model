"""Tests for rescue and follow-up sweep tooling."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.summarize_rescue import build_rescue_summary


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


def write_history(exp_dir: Path, headers: list[str], rows: list[list[float]]) -> None:
    with open(exp_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


class TestRescueSummary:
    def test_build_rescue_summary_keep_fixmatch(self, tmp_path):
        rescue_dir = tmp_path / "results_rescue"
        for seed, sup_auc, fix_auc in [(42, 0.61, 0.74), (43, 0.62, 0.76), (44, 0.60, 0.75)]:
            write_metrics(rescue_dir / f"supervised_100_seed{seed}", sup_auc, sup_auc - 0.02)
            write_metrics(rescue_dir / f"fixmatch_100_seed{seed}", fix_auc, fix_auc - 0.03)
            write_metrics(rescue_dir / f"supervised_250_seed{seed}", 0.70, 0.68)
            write_metrics(rescue_dir / f"fixmatch_250_seed{seed}", 0.70, 0.69)

        summary = build_rescue_summary(rescue_dir, [42, 43, 44])

        assert summary["promotion_gate"]["ready"] is True
        assert summary["promotion_gate"]["decision"] == "keep_fixmatch"


class TestFollowupSummary:
    def test_followup_summary_handles_partial_runs_and_recommends_mean_teacher(self, tmp_path):
        rescue_dir = tmp_path / "results_rescue"
        followup_dir = tmp_path / "results_followup"

        for seed, sup_auc in [(42, 0.61), (43, 0.62), (44, 0.60)]:
            write_metrics(rescue_dir / f"supervised_100_seed{seed}", sup_auc, 0.6)
            write_metrics(rescue_dir / f"supervised_250_seed{seed}", 0.70, 0.69)
            write_metrics(rescue_dir / f"fixmatch_100_seed{seed}", 0.74, 0.70)
            write_metrics(rescue_dir / f"fixmatch_250_seed{seed}", 0.67, 0.64)

        write_metrics(followup_dir / "mean_teacher_100_seed42", 0.66, 0.64)
        write_history(
            followup_dir / "mean_teacher_100_seed42",
            ["consistency_loss", "consistency_weight"],
            [[0.5, 0.4], [0.3, 1.0]],
        )
        write_metrics(followup_dir / "mean_teacher_100_seed43", 0.65, 0.63)
        write_history(
            followup_dir / "mean_teacher_100_seed43",
            ["consistency_loss", "consistency_weight"],
            [[0.6, 0.4], [0.3, 1.0]],
        )
        write_metrics(followup_dir / "mean_teacher_250_seed42", 0.73, 0.71)
        write_history(
            followup_dir / "mean_teacher_250_seed42",
            ["consistency_loss", "consistency_weight"],
            [[0.4, 0.4], [0.2, 1.0]],
        )
        write_metrics(followup_dir / "mean_teacher_250_seed43", 0.74, 0.72)
        write_history(
            followup_dir / "mean_teacher_250_seed43",
            ["consistency_loss", "consistency_weight"],
            [[0.5, 0.4], [0.2, 1.0]],
        )

        write_metrics(followup_dir / "supervised_nofreeze_100_seed42", 0.66, 0.65)
        write_history(
            followup_dir / "supervised_nofreeze_100_seed42",
            ["train_loss"],
            [[1.0], [0.7]],
        )
        write_metrics(followup_dir / "supervised_nofreeze_250_seed42", 0.75, 0.73)
        write_history(
            followup_dir / "supervised_nofreeze_250_seed42",
            ["train_loss"],
            [[1.0], [0.7]],
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/summarize_followup.py",
                "--output_dir",
                str(followup_dir),
                "--rescue_dir",
                str(rescue_dir),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )

        summary_path = followup_dir / "followup_summary.txt"
        csv_path = followup_dir / "followup_summary.csv"
        assert summary_path.exists()
        assert csv_path.exists()
        summary_text = summary_path.read_text()
        assert "Promote Mean Teacher as the main next SSL path." in summary_text
        assert "Mean Teacher Comparison" in summary_text
        assert "Freeze Sanity Comparison" in summary_text
        assert "Future SSL comparisons should include a matched no-freeze baseline" not in summary_text
        assert "followup_summary.csv" not in result.stdout


class TestConfigIsolation:
    def test_default_nofreeze_diff_is_freeze_only(self):
        base = yaml.safe_load(Path("configs/default.yaml").read_text())
        probe = yaml.safe_load(Path("configs/default_nofreeze.yaml").read_text())

        probe_dataset = {k: v for k, v in probe["dataset"].items() if k not in {"name"}}
        base_dataset = {k: v for k, v in base["dataset"].items() if k not in {"name"}}
        assert probe_dataset == base_dataset

        probe_training = {k: v for k, v in probe["training"].items() if k not in {"label_smoothing"}}
        base_training = {k: v for k, v in base["training"].items() if k not in {"label_smoothing"}}
        assert probe_training == base_training

        assert probe["augmentation"] == base["augmentation"]
        assert probe["logging"] == base["logging"]
        assert probe["wandb"] == base["wandb"]
        assert probe["model"]["freeze_backbone"] is False
        assert probe["model"]["freeze_backbone_epochs"] == 0
        comparable_model = {k: v for k, v in probe["model"].items() if k not in {"freeze_backbone", "freeze_backbone_epochs"}}
        base_model = {k: v for k, v in base["model"].items() if k not in {"freeze_backbone", "freeze_backbone_epochs"}}
        assert comparable_model == base_model

    def test_fixmatch_static_diff_is_schedule_only(self):
        base = yaml.safe_load(Path("configs/fixmatch.yaml").read_text())
        probe = yaml.safe_load(Path("configs/fixmatch_static.yaml").read_text())

        assert probe["augmentation"] == base["augmentation"]
        assert probe["model"] == base["model"]
        assert probe["training"] == base["training"]
        assert probe["ssl"]["confidence_threshold_start"] == 0.95
        assert probe["ssl"]["confidence_threshold_end"] == 0.95
        assert probe["ssl"]["confidence_threshold_ramp_epochs"] == 0
        assert probe["ssl"]["lambda_u_start"] == 1.0
        assert probe["ssl"]["lambda_u_end"] == 1.0
        assert probe["ssl"]["lambda_u_ramp_epochs"] == 0

    def test_fixmatch_legacy_aug_diff_is_augmentation_only(self):
        base = yaml.safe_load(Path("configs/fixmatch.yaml").read_text())
        probe = yaml.safe_load(Path("configs/fixmatch_legacy_aug.yaml").read_text())

        comparable_ssl_keys = {
            key: value for key, value in probe["ssl"].items() if key not in {"method"}
        }
        base_ssl_keys = {key: value for key, value in base["ssl"].items() if key not in {"method"}}
        assert comparable_ssl_keys == base_ssl_keys
        assert probe["augmentation"] != base["augmentation"]
        assert probe["augmentation"]["weak"]["random_vertical_flip"] is True
        assert probe["augmentation"]["weak"]["color_jitter"] == 0.1


class TestFollowupRunner:
    def test_followup_runner_dry_run_uses_mean_teacher_first_profile(self, tmp_path):
        rescue_dir = tmp_path / "results_rescue"
        output_dir = tmp_path / "results_followup"

        for seed, sup_auc, fix_auc in [(42, 0.61, 0.74), (43, 0.62, 0.76), (44, 0.60, 0.75)]:
            write_metrics(rescue_dir / f"supervised_100_seed{seed}", sup_auc, 0.6)
            write_metrics(rescue_dir / f"fixmatch_100_seed{seed}", fix_auc, 0.7)
            write_metrics(rescue_dir / f"supervised_250_seed{seed}", 0.70, 0.69)
            write_metrics(rescue_dir / f"fixmatch_250_seed{seed}", 0.71, 0.70)

        write_metrics(output_dir / "supervised_500_seed42", 0.80, 0.79)

        result = subprocess.run(
            [
                "bash",
                "run_followup_overnight.sh",
                "--rescue_dir",
                str(rescue_dir),
                "--output",
                str(output_dir),
                "--profile",
                "mean_teacher_first",
                "--dry_run",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )

        log_text = (output_dir / "followup.log").read_text()
        rescue_json = json.loads((output_dir / "rescue_summary.json").read_text())
        assert rescue_json["promotion_gate"]["decision"] == "keep_fixmatch"
        assert "Profile selected: Mean Teacher first" in log_text
        assert "DRY RUN: not executing mean_teacher_100_seed42" in log_text
        assert "DRY RUN: not executing supervised_nofreeze_100_seed42" in log_text
        assert "SKIP: supervised_500_seed42 already completed" in log_text
        assert "Follow-Up Overnight Summary" in result.stdout
