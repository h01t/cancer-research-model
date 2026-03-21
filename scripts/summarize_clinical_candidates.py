#!/usr/bin/env python3
"""Summarize clinical evaluation bundles across candidate runs."""

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

    for summary_path in sorted(root.glob("*/clinical_summary.yaml")):
        data = load_yaml(summary_path)
        val_metrics = data["val_metrics"]
        test_metrics = data["test_metrics"]

        exam_metrics = {}
        for grouped in data.get("grouped_metrics", []):
            if grouped.get("group_name") == "exam_id":
                exam_metrics = grouped
                break

        rows.append(
            {
                "run_name": data.get("run_name", summary_path.parent.name),
                "config_name": data["config_name"],
                "seed": int(data["seed"]),
                "backbone": data["backbone"],
                "image_size": int(data["image_size"]),
                "optimizer": data["optimizer"],
                "label_smoothing": float(data.get("label_smoothing", 0.0)),
                "val_auc": float(val_metrics["auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
                "val_brier_score": float(val_metrics["brier_score"]),
                "val_ece": float(val_metrics["ece"]),
                "test_auc": float(test_metrics["auc"]),
                "test_pr_auc": float(test_metrics["pr_auc"]),
                "test_brier_score": float(test_metrics["brier_score"]),
                "test_ece": float(test_metrics["ece"]),
                "test_sensitivity": float(test_metrics["sensitivity"]),
                "test_specificity": float(test_metrics["specificity"]),
                "exam_auc": float(exam_metrics.get("auc", 0.0)),
                "exam_pr_auc": float(exam_metrics.get("pr_auc", 0.0)),
            }
        )
    return rows


def grouped_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["config_name"], []).append(row)

    summary = []
    for config_name, members in sorted(grouped.items()):
        summary.append(
            {
                "config_name": config_name,
                "runs": len(members),
                "backbone": members[0]["backbone"],
                "image_size": members[0]["image_size"],
                "optimizer": members[0]["optimizer"],
                "label_smoothing": members[0]["label_smoothing"],
                "mean_val_auc": mean([m["val_auc"] for m in members]),
                "mean_val_pr_auc": mean([m["val_pr_auc"] for m in members]),
                "mean_val_brier_score": mean([m["val_brier_score"] for m in members]),
                "mean_val_ece": mean([m["val_ece"] for m in members]),
                "mean_test_auc": mean([m["test_auc"] for m in members]),
                "mean_test_pr_auc": mean([m["test_pr_auc"] for m in members]),
                "mean_test_brier_score": mean([m["test_brier_score"] for m in members]),
                "mean_test_ece": mean([m["test_ece"] for m in members]),
                "mean_test_sensitivity": mean([m["test_sensitivity"] for m in members]),
                "mean_test_specificity": mean([m["test_specificity"] for m in members]),
                "mean_exam_auc": mean([m["exam_auc"] for m in members]),
                "mean_exam_pr_auc": mean([m["exam_pr_auc"] for m in members]),
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
    lines = ["Clinical Candidate Summary", "==========================", ""]
    lines.append(
        "run_name                       config                        val_auc  test_auc  test_pr  brier    ece      sens     spec"
    )
    lines.append(
        "-----------------------------  ----------------------------  -------  --------  -------  -------  -------  -------  -------"
    )
    for row in rows:
        lines.append(
            f"{row['run_name']:<29} {row['config_name']:<28} "
            f"{row['val_auc']:>7.4f}  {row['test_auc']:>8.4f}  {row['test_pr_auc']:>7.4f}  "
            f"{row['test_brier_score']:>7.4f}  {row['test_ece']:>7.4f}  "
            f"{row['test_sensitivity']:>7.4f}  {row['test_specificity']:>7.4f}"
        )

    lines.extend(["", "Grouped Means", "-------------"])
    lines.append(
        "config                        mean_val_auc  mean_test_auc  mean_test_pr  mean_brier  mean_ece  mean_sens  mean_spec  mean_exam_auc"
    )
    lines.append(
        "----------------------------  ------------  -------------  ------------  ----------  --------  ---------  ---------  -------------"
    )
    for row in grouped:
        lines.append(
            f"{row['config_name']:<28} {row['mean_val_auc']:>12.4f}  {row['mean_test_auc']:>13.4f}  "
            f"{row['mean_test_pr_auc']:>12.4f}  {row['mean_test_brier_score']:>10.4f}  "
            f"{row['mean_test_ece']:>8.4f}  {row['mean_test_sensitivity']:>9.4f}  "
            f"{row['mean_test_specificity']:>9.4f}  {row['mean_exam_auc']:>13.4f}"
        )

    if grouped:
        best = max(grouped, key=lambda row: row["mean_val_auc"])
        lines.extend(
            [
                "",
                "Recommendation",
                "--------------",
                (
                    "Best clinical candidate by mean validation AUC so far: "
                    f"{best['config_name']} "
                    f"(val_auc={best['mean_val_auc']:.4f}, test_auc={best['mean_test_auc']:.4f}, "
                    f"test_brier={best['mean_test_brier_score']:.4f}, test_ece={best['mean_test_ece']:.4f})."
                ),
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize clinical evaluation outputs")
    parser.add_argument("--results_dir", type=str, default="results_supervised_sweep/clinical_candidates")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(results_dir)
    grouped = grouped_rows(rows)
    summary = format_summary(rows, grouped)

    (output_dir / "clinical_candidates_summary.txt").write_text(summary)
    write_csv(
        output_dir / "clinical_candidates_summary.csv",
        rows,
        [
            "run_name",
            "config_name",
            "seed",
            "backbone",
            "image_size",
            "optimizer",
            "label_smoothing",
            "val_auc",
            "val_pr_auc",
            "val_brier_score",
            "val_ece",
            "test_auc",
            "test_pr_auc",
            "test_brier_score",
            "test_ece",
            "test_sensitivity",
            "test_specificity",
            "exam_auc",
            "exam_pr_auc",
        ],
    )
    write_csv(
        output_dir / "clinical_candidates_grouped.csv",
        grouped,
        [
            "config_name",
            "runs",
            "backbone",
            "image_size",
            "optimizer",
            "label_smoothing",
            "mean_val_auc",
            "mean_val_pr_auc",
            "mean_val_brier_score",
            "mean_val_ece",
            "mean_test_auc",
            "mean_test_pr_auc",
            "mean_test_brier_score",
            "mean_test_ece",
            "mean_test_sensitivity",
            "mean_test_specificity",
            "mean_exam_auc",
            "mean_exam_pr_auc",
        ],
    )
    print(summary)


if __name__ == "__main__":
    main()
