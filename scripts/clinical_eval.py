#!/usr/bin/env python3
"""Create a compact clinical-style evidence bundle for a supervised run."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import (
    build_experiment_context,
    build_supervised_experiment,
    collect_loader_predictions,
    create_model,
    create_trainer,
)
from src.training.metrics import (
    aggregate_group_predictions,
    bootstrap_metric_ci,
    calibration_table,
    compute_calibration_metrics,
    compute_metrics,
    subgroup_metrics_table,
    threshold_for_target_sensitivity,
    threshold_table,
)


SEED_SUFFIX_PATTERN = re.compile(r"_seed\d+$")


def derive_config_name(run_name: str) -> str:
    return SEED_SUFFIX_PATTERN.sub("", run_name)


def summarize_frame(
    frame: pd.DataFrame,
    decision_threshold: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_prob = frame["y_prob"].to_numpy(dtype=float)
    y_pred = (y_prob >= decision_threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics.update(compute_calibration_metrics(y_true, y_prob))
    metrics["decision_threshold"] = float(decision_threshold)

    ci_specs = {
        "auc_ci": lambda yt, yp: compute_metrics(yt, (yp >= decision_threshold).astype(int), yp)["auc"],
        "pr_auc_ci": lambda yt, yp: compute_metrics(yt, (yp >= decision_threshold).astype(int), yp)["pr_auc"],
        "sensitivity_ci": lambda yt, yp: compute_metrics(yt, (yp >= decision_threshold).astype(int), yp)["sensitivity"],
        "specificity_ci": lambda yt, yp: compute_metrics(yt, (yp >= decision_threshold).astype(int), yp)["specificity"],
    }
    for key, fn in ci_specs.items():
        low, high = bootstrap_metric_ci(
            fn,
            y_true,
            y_prob,
            num_bootstrap=bootstrap_samples,
            seed=bootstrap_seed,
        )
        metrics[f"{key}_low"] = low
        metrics[f"{key}_high"] = high

    return metrics


def build_group_rows(frame: pd.DataFrame, decision_threshold: float) -> list[dict]:
    rows = []
    for group_name in ("patient_id", "exam_id"):
        if group_name not in frame.columns:
            continue
        y_true_group, y_prob_group = aggregate_group_predictions(
            frame["y_true"].to_numpy(dtype=int),
            frame["y_prob"].to_numpy(dtype=float),
            frame[group_name].astype(str).tolist(),
            reducer="max",
        )
        y_pred_group = (y_prob_group >= decision_threshold).astype(int)
        metrics = compute_metrics(y_true_group, y_pred_group, y_prob_group)
        metrics.update(compute_calibration_metrics(y_true_group, y_prob_group))
        metrics["group_name"] = group_name
        metrics["decision_threshold"] = float(decision_threshold)
        rows.append(metrics)
    return rows


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clinical-style evaluation bundle")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory with config + checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: run_dir/best_model.pth)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: run_dir/clinical_eval)")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pth"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "clinical_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        saved_config = yaml.safe_load(f)
    seed = saved_config.get("experiment", {}).get("seed", 42)

    context = build_experiment_context(
        config_path=str(config_path),
        output_dir=str(output_dir),
        seed=seed,
        device_override=args.device,
        overrides={"experiment": {"method": "supervised"}},
    )

    bundle = build_supervised_experiment(
        context.config,
        context.device,
        seed=context.seed,
        labeled_subset_size=context.config["dataset"].get("labeled_subset_size"),
    )

    model = create_model(context.config)
    trainer = create_trainer("supervised", model, context, class_weights=None)
    trainer.load_checkpoint(checkpoint_path)

    _, val_frame = collect_loader_predictions(trainer, bundle.loaders.val_loader)
    decision_threshold, _ = trainer.tune_decision_threshold(bundle.loaders.val_loader)
    _, test_frame = collect_loader_predictions(trainer, bundle.loaders.test_loader)

    val_metrics = summarize_frame(
        val_frame,
        decision_threshold=decision_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    test_metrics = summarize_frame(
        test_frame,
        decision_threshold=decision_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    fixed_sensitivity_rows = []
    for target in (0.85, 0.90, 0.95):
        threshold, _ = threshold_for_target_sensitivity(
            val_frame["y_true"].to_numpy(dtype=int),
            val_frame["y_prob"].to_numpy(dtype=float),
            target_sensitivity=target,
        )
        metrics = summarize_frame(
            test_frame,
            decision_threshold=threshold,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        fixed_sensitivity_rows.append(
            {
                "target_sensitivity": target,
                "threshold": threshold,
                "test_specificity": metrics["specificity"],
                "test_sensitivity": metrics["sensitivity"],
                "test_auc": metrics["auc"],
                "test_pr_auc": metrics["pr_auc"],
            }
        )

    subgroup_df = subgroup_metrics_table(
        test_frame,
        test_frame["y_true"].to_numpy(dtype=int),
        test_frame["y_prob"].to_numpy(dtype=float),
        decision_threshold=decision_threshold,
        columns=["laterality", "view", "abnormality_type", "source_id"],
    )
    group_rows = build_group_rows(test_frame, decision_threshold)

    failures = test_frame.copy()
    failures["y_pred"] = (failures["y_prob"].to_numpy(dtype=float) >= decision_threshold).astype(int)
    fp_df = failures[(failures["y_true"] == 0) & (failures["y_pred"] == 1)].sort_values(
        "y_prob", ascending=False
    )
    fn_df = failures[(failures["y_true"] == 1) & (failures["y_pred"] == 0)].sort_values(
        "y_prob", ascending=True
    )

    val_metrics["split"] = "val"
    test_metrics["split"] = "test"
    summary = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "config_name": derive_config_name(run_dir.name),
        "checkpoint": str(checkpoint_path),
        "backbone": context.config["model"]["name"],
        "image_size": int(context.config["dataset"]["image_size"]),
        "optimizer": context.config["training"]["optimizer"],
        "label_smoothing": float(context.config["training"].get("label_smoothing", 0.0)),
        "seed": int(context.seed),
        "decision_threshold": float(decision_threshold),
        "val_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()},
        "grouped_metrics": group_rows,
        "fixed_sensitivity": fixed_sensitivity_rows,
    }
    write_yaml(output_dir / "clinical_summary.yaml", summary)

    pd.DataFrame([val_metrics, test_metrics]).to_csv(output_dir / "clinical_summary.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "grouped_metrics.csv", index=False)
    pd.DataFrame(fixed_sensitivity_rows).to_csv(output_dir / "fixed_sensitivity_metrics.csv", index=False)
    threshold_table(
        test_frame["y_true"].to_numpy(dtype=int),
        test_frame["y_prob"].to_numpy(dtype=float),
    ).to_csv(output_dir / "threshold_table.csv", index=False)
    calibration_table(
        test_frame["y_true"].to_numpy(dtype=int),
        test_frame["y_prob"].to_numpy(dtype=float),
    ).to_csv(output_dir / "calibration_bins.csv", index=False)
    subgroup_df.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    fp_df.head(25).to_csv(output_dir / "top_false_positives.csv", index=False)
    fn_df.head(25).to_csv(output_dir / "top_false_negatives.csv", index=False)

    report = "\n".join(
        [
            "# Candidate Baseline Report",
            "",
            f"- Backbone: `{context.config['model']['name']}`",
            f"- Image size: `{context.config['dataset']['image_size']}`",
            f"- Optimizer: `{context.config['training']['optimizer']}`",
            f"- Seed: `{context.seed}`",
            f"- Decision threshold: `{decision_threshold:.4f}`",
            "",
            "## Validation",
            f"- ROC AUC: `{val_metrics['auc']:.4f}`",
            f"- PR AUC: `{val_metrics['pr_auc']:.4f}`",
            f"- Sensitivity: `{val_metrics['sensitivity']:.4f}`",
            f"- Specificity: `{val_metrics['specificity']:.4f}`",
            f"- Brier score: `{val_metrics['brier_score']:.4f}`",
            f"- ECE: `{val_metrics['ece']:.4f}`",
            "",
            "## Test",
            f"- ROC AUC: `{test_metrics['auc']:.4f}`",
            f"- PR AUC: `{test_metrics['pr_auc']:.4f}`",
            f"- Sensitivity: `{test_metrics['sensitivity']:.4f}`",
            f"- Specificity: `{test_metrics['specificity']:.4f}`",
            f"- Brier score: `{test_metrics['brier_score']:.4f}`",
            f"- ECE: `{test_metrics['ece']:.4f}`",
            "",
            "## Artifacts",
            "- `clinical_summary.yaml`",
            "- `clinical_summary.csv`",
            "- `grouped_metrics.csv`",
            "- `fixed_sensitivity_metrics.csv`",
            "- `threshold_table.csv`",
            "- `calibration_bins.csv`",
            "- `subgroup_metrics.csv`",
            "- `top_false_positives.csv`",
            "- `top_false_negatives.csv`",
        ]
    )
    (output_dir / "candidate_baseline_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
