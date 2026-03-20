#!/usr/bin/env python3
"""Build a compact final comparison across rescue and follow-up SSL experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import yaml


METHOD_PATTERN = re.compile(
    r"^(?P<method>supervised_nofreeze|supervised|fixmatch|mean_teacher)"
    r"_(?P<subset>full|\d+)_seed(?P<seed>\d+)$"
)


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def parse_experiment_name(name: str) -> dict | None:
    match = METHOD_PATTERN.match(name)
    if not match:
        return None
    subset_text = match.group("subset")
    return {
        "method": match.group("method"),
        "subset": subset_text if subset_text == "full" else int(subset_text),
        "seed": int(match.group("seed")),
    }


def collect_rows(root: Path, source: str) -> list[dict]:
    rows = []
    if not root.exists():
        return rows

    for exp_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        parsed = parse_experiment_name(exp_dir.name)
        if parsed is None:
            continue
        val_metrics_path = exp_dir / "val_metrics.yaml"
        test_metrics_path = exp_dir / "test_metrics.yaml"
        if not val_metrics_path.exists() or not test_metrics_path.exists():
            continue

        val_metrics = load_yaml(val_metrics_path)
        test_metrics = load_yaml(test_metrics_path)
        rows.append(
            {
                "source": source,
                "experiment": exp_dir.name,
                "method": parsed["method"],
                "subset": parsed["subset"],
                "seed": parsed["seed"],
                "val_auc": float(val_metrics["auc"]),
                "test_auc": float(test_metrics["auc"]),
                "sensitivity": float(test_metrics["sensitivity"]),
                "specificity": float(test_metrics["specificity"]),
                "decision_threshold": float(test_metrics.get("decision_threshold", 0.5)),
            }
        )
    return rows


def group_means(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["method"], str(row["subset"]))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (method, subset), members in sorted(grouped.items()):
        summary_rows.append(
            {
                "method": method,
                "subset": subset,
                "num_runs": len(members),
                "mean_val_auc": mean([row["val_auc"] for row in members]),
                "mean_test_auc": mean([row["test_auc"] for row in members]),
                "mean_sensitivity": mean([row["sensitivity"] for row in members]),
                "mean_specificity": mean([row["specificity"] for row in members]),
            }
        )
    return summary_rows


def lookup(grouped: list[dict], method: str, subset: int | str, metric: str) -> float | None:
    subset_text = str(subset)
    for row in grouped:
        if row["method"] == method and row["subset"] == subset_text:
            return float(row[metric])
    return None


def build_recommendation(grouped: list[dict]) -> str:
    parts = []
    nofreeze_100 = lookup(grouped, "supervised_nofreeze", 100, "mean_val_auc")
    nofreeze_250 = lookup(grouped, "supervised_nofreeze", 250, "mean_val_auc")
    nofreeze_500 = lookup(grouped, "supervised_nofreeze", 500, "mean_val_auc")
    sup_100 = lookup(grouped, "supervised", 100, "mean_val_auc")
    sup_250 = lookup(grouped, "supervised", 250, "mean_val_auc")
    mt_100 = lookup(grouped, "mean_teacher", 100, "mean_val_auc")
    mt_250 = lookup(grouped, "mean_teacher", 250, "mean_val_auc")
    fix_100 = lookup(grouped, "fixmatch", 100, "mean_val_auc")
    fix_250 = lookup(grouped, "fixmatch", 250, "mean_val_auc")

    parts.append("Official baseline: supervised no-freeze.")

    if None not in (nofreeze_100, sup_100):
        parts.append(f"At 100 labels, no-freeze supervised beats frozen supervised by {nofreeze_100 - sup_100:+.4f} val AUC.")
    if None not in (nofreeze_250, sup_250):
        parts.append(f"At 250 labels, no-freeze supervised beats frozen supervised by {nofreeze_250 - sup_250:+.4f} val AUC.")
    if None not in (nofreeze_100, fix_100):
        parts.append(f"Versus FixMatch at 100, the no-freeze baseline is {nofreeze_100 - fix_100:+.4f} val AUC ahead.")
    if None not in (nofreeze_250, fix_250):
        parts.append(f"Versus FixMatch at 250, the no-freeze baseline is {nofreeze_250 - fix_250:+.4f} val AUC ahead.")
    if None not in (nofreeze_100, mt_100):
        parts.append(f"Versus Mean Teacher at 100, the no-freeze baseline is {nofreeze_100 - mt_100:+.4f} val AUC ahead.")
    if None not in (nofreeze_250, mt_250):
        parts.append(f"Versus Mean Teacher at 250, the no-freeze baseline is {nofreeze_250 - mt_250:+.4f} val AUC ahead.")
    if nofreeze_500 is not None:
        parts.append(f"The official 500-label no-freeze baseline currently has mean val AUC {nofreeze_500:.4f}.")

    parts.append("Conclusion: close the vanilla SSL chapter for now and focus on stronger supervised baselines or pretraining.")
    return " ".join(parts)


def format_summary(rows: list[dict], grouped: list[dict], recommendation: str) -> str:
    lines = ["SSL Chapter Closure Summary", "===========================", ""]
    lines.append(
        "source   experiment                      val_auc  test_auc  sens     spec     threshold"
    )
    lines.append(
        "-------  ------------------------------  -------  --------  -------  -------  ---------"
    )
    for row in sorted(rows, key=lambda item: (item["source"], str(item["subset"]), item["method"], item["seed"])):
        lines.append(
            f"{row['source']:<7} {row['experiment']:<30} {row['val_auc']:>7.4f} "
            f"{row['test_auc']:>8.4f} {row['sensitivity']:>7.4f} "
            f"{row['specificity']:>7.4f} {row['decision_threshold']:>9.4f}"
        )

    lines.extend(["", "Grouped Means", "-------------"])
    lines.append("method              subset  runs  mean_val_auc  mean_test_auc  mean_sens  mean_spec")
    lines.append("------------------  -----  ----  ------------  -------------  ---------  ---------")
    for row in grouped:
        lines.append(
            f"{row['method']:<18} {str(row['subset']):>5} {row['num_runs']:>5} "
            f"{row['mean_val_auc']:>12.4f} {row['mean_test_auc']:>13.4f} "
            f"{row['mean_sensitivity']:>10.4f} {row['mean_specificity']:>10.4f}"
        )

    lines.extend(["", "Recommendation", "--------------", recommendation])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source",
        "experiment",
        "method",
        "subset",
        "seed",
        "val_auc",
        "test_auc",
        "sensitivity",
        "specificity",
        "decision_threshold",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize the final supervised-vs-SSL comparison")
    parser.add_argument("--rescue_dir", type=str, default="results_rescue")
    parser.add_argument("--followup_dir", type=str, default="results_followup")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    rescue_dir = Path(args.rescue_dir)
    followup_dir = Path(args.followup_dir)
    output_dir = Path(args.output_dir) if args.output_dir else followup_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(rescue_dir, "rescue")
    rows.extend(collect_rows(followup_dir, "followup"))
    grouped = group_means(rows)
    recommendation = build_recommendation(grouped)
    summary_text = format_summary(rows, grouped, recommendation)

    (output_dir / "ssl_closure_summary.txt").write_text(summary_text)
    write_csv(output_dir / "ssl_closure_summary.csv", rows)
    print(summary_text)


if __name__ == "__main__":
    main()
