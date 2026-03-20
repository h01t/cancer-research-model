#!/usr/bin/env python3
"""Summarize supervised experiment directories into grouped comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.yaml"
        val_path = run_dir / "val_metrics.yaml"
        test_path = run_dir / "test_metrics.yaml"
        if not (config_path.exists() and val_path.exists() and test_path.exists()):
            continue

        config = load_yaml(config_path)
        if config.get("experiment", {}).get("method", "supervised") != "supervised":
            continue

        val_metrics = load_yaml(val_path)
        test_metrics = load_yaml(test_path)
        rows.append(
            {
                "run_name": run_dir.name,
                "backbone": config["model"]["name"],
                "image_size": int(config["dataset"]["image_size"]),
                "optimizer": config["training"]["optimizer"],
                "label_smoothing": float(config["training"].get("label_smoothing", 0.0)),
                "batch_size": int(config["training"]["batch_size"]),
                "labeled_subset_size": config["dataset"].get("labeled_subset_size"),
                "seed": config.get("experiment", {}).get("seed", -1),
                "val_auc": float(val_metrics["auc"]),
                "test_auc": float(test_metrics["auc"]),
                "sensitivity": float(test_metrics["sensitivity"]),
                "specificity": float(test_metrics["specificity"]),
                "decision_threshold": float(test_metrics.get("decision_threshold", 0.5)),
            }
        )
    return rows


def grouped_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["backbone"],
            row["image_size"],
            row["optimizer"],
            row["label_smoothing"],
            row["labeled_subset_size"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, members in sorted(grouped.items()):
        backbone, image_size, optimizer, label_smoothing, labeled_subset_size = key
        summary.append(
            {
                "backbone": backbone,
                "image_size": image_size,
                "optimizer": optimizer,
                "label_smoothing": label_smoothing,
                "labeled_subset_size": labeled_subset_size,
                "runs": len(members),
                "mean_val_auc": mean([m["val_auc"] for m in members]),
                "mean_test_auc": mean([m["test_auc"] for m in members]),
                "mean_sensitivity": mean([m["sensitivity"] for m in members]),
                "mean_specificity": mean([m["specificity"] for m in members]),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_summary(rows: list[dict], grouped: list[dict]) -> str:
    lines = ["Supervised Sweep Summary", "========================", ""]
    lines.append(
        "run_name                       backbone        size  opt     ls     seed  val_auc  test_auc  sens     spec"
    )
    lines.append(
        "-----------------------------  --------------  ----  ------  -----  ----  -------  --------  -------  -------"
    )
    for row in rows:
        lines.append(
            f"{row['run_name']:<29} {row['backbone']:<14} {row['image_size']:>4}  "
            f"{row['optimizer']:<6}  {row['label_smoothing']:>5.2f}  {row['seed']:>4}  "
            f"{row['val_auc']:>7.4f}  {row['test_auc']:>8.4f}  "
            f"{row['sensitivity']:>7.4f}  {row['specificity']:>7.4f}"
        )

    lines.extend(["", "Grouped Means", "-------------"])
    lines.append(
        "backbone        size  opt     ls     labeled  runs  mean_val_auc  mean_test_auc  mean_sens  mean_spec"
    )
    lines.append(
        "--------------  ----  ------  -----  -------  ----  ------------  -------------  ---------  ---------"
    )
    for row in grouped:
        lines.append(
            f"{row['backbone']:<14} {row['image_size']:>4}  {row['optimizer']:<6}  "
            f"{row['label_smoothing']:>5.2f}  {str(row['labeled_subset_size']):>7}  {row['runs']:>4}  "
            f"{row['mean_val_auc']:>12.4f}  {row['mean_test_auc']:>13.4f}  "
            f"{row['mean_sensitivity']:>9.4f}  {row['mean_specificity']:>9.4f}"
        )

    if grouped:
        best = max(grouped, key=lambda row: row["mean_val_auc"])
        lines.extend(
            [
                "",
                "Recommendation",
                "--------------",
                (
                    "Best grouped candidate so far: "
                    f"{best['backbone']} at {best['image_size']}px with {best['optimizer']} "
                    f"(label_smoothing={best['label_smoothing']:.2f}), "
                    f"mean val AUC {best['mean_val_auc']:.4f}."
                ),
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize supervised experiment runs")
    parser.add_argument("--results_dir", type=str, default="results_supervised_sweep")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(results_dir)
    grouped = grouped_rows(rows)
    summary = format_summary(rows, grouped)

    (output_dir / "supervised_summary.txt").write_text(summary)
    write_csv(
        output_dir / "supervised_summary.csv",
        rows,
        [
            "run_name",
            "backbone",
            "image_size",
            "optimizer",
            "label_smoothing",
            "batch_size",
            "labeled_subset_size",
            "seed",
            "val_auc",
            "test_auc",
            "sensitivity",
            "specificity",
            "decision_threshold",
        ],
    )
    write_csv(
        output_dir / "supervised_summary_grouped.csv",
        grouped,
        [
            "backbone",
            "image_size",
            "optimizer",
            "label_smoothing",
            "labeled_subset_size",
            "runs",
            "mean_val_auc",
            "mean_test_auc",
            "mean_sensitivity",
            "mean_specificity",
        ],
    )
    print(summary)


if __name__ == "__main__":
    main()
