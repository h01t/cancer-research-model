#!/usr/bin/env python3
"""Summarize follow-up overnight experiments and emit a recommendation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import pandas as pd
import yaml


METHOD_PATTERN = re.compile(
    r"^(?P<method>supervised_nofreeze|supervised|fixmatch_static|fixmatch_legacy_aug|fixmatch|mean_teacher)"
    r"_(?P<subset>full|\d+)_seed(?P<seed>\d+)$"
)
MATERIAL_DROP = 0.02
NOFREEZE_ALERT_DELTA = 0.03


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


def load_health_metrics(history_path: Path, method: str) -> dict:
    if not history_path.exists():
        return {}

    history = pd.read_csv(history_path)
    if history.empty:
        return {}

    if method.startswith("fixmatch"):
        return {
            "peak_mask_ratio": _last_or_nan(history.get("mask_ratio"), reducer="max"),
            "final_unsup_loss": _last_or_nan(history.get("unsup_loss")),
            "final_lambda_u": _last_or_nan(history.get("lambda_u")),
        }
    if method == "mean_teacher":
        return {
            "final_consistency_loss": _last_or_nan(history.get("consistency_loss")),
            "final_consistency_weight": _last_or_nan(history.get("consistency_weight")),
        }
    return {}


def _last_or_nan(series, reducer: str = "last"):
    if series is None or len(series) == 0:
        return ""
    clean = series.dropna()
    if clean.empty:
        return ""
    if reducer == "max":
        return float(clean.max())
    return float(clean.iloc[-1])


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
        row = {
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
        row.update(load_health_metrics(exp_dir / "training_history.csv", parsed["method"]))
        rows.append(row)
    return rows


def group_means(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["method"], str(row["subset"]))
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (method, subset), members in sorted(grouped.items()):
        summaries.append(
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
    return summaries


def lookup_group_metric(grouped: list[dict], method: str, subset: str | int, metric: str) -> float | None:
    subset_text = str(subset)
    for row in grouped:
        if row["method"] == method and row["subset"] == subset_text:
            return float(row[metric])
    return None


def build_recommendation(grouped: list[dict]) -> str:
    mt_100 = lookup_group_metric(grouped, "mean_teacher", 100, "mean_val_auc")
    mt_250 = lookup_group_metric(grouped, "mean_teacher", 250, "mean_val_auc")
    sup_100 = lookup_group_metric(grouped, "supervised", 100, "mean_val_auc")
    sup_250 = lookup_group_metric(grouped, "supervised", 250, "mean_val_auc")
    nofreeze_100 = lookup_group_metric(grouped, "supervised_nofreeze", 100, "mean_val_auc")
    nofreeze_250 = lookup_group_metric(grouped, "supervised_nofreeze", 250, "mean_val_auc")

    nofreeze_flags = []
    if nofreeze_100 is not None and sup_100 is not None:
        delta = nofreeze_100 - sup_100
        if delta >= NOFREEZE_ALERT_DELTA:
            nofreeze_flags.append(
                f"The no-freeze supervised 100 baseline improves over rescue supervised by {delta:+.4f}."
            )
    if nofreeze_250 is not None and sup_250 is not None:
        delta = nofreeze_250 - sup_250
        if delta >= NOFREEZE_ALERT_DELTA:
            nofreeze_flags.append(
                f"The no-freeze supervised 250 baseline improves over rescue supervised by {delta:+.4f}."
            )

    if None not in (mt_100, mt_250, sup_100, sup_250):
        delta_100 = mt_100 - sup_100
        delta_250 = mt_250 - sup_250
        if delta_100 >= 0.02 and delta_250 >= 0.02:
            verdict = "Promote Mean Teacher as the main next SSL path."
        elif delta_100 >= 0.02:
            verdict = (
                "Mean Teacher looks like another low-label-only signal; prioritize baseline sanity checks "
                "and representation-learning alternatives next."
            )
        else:
            verdict = (
                "Deprioritize new consistency-based SSL methods for now and focus on stronger supervised "
                "baselines or pretraining."
            )

        details = (
            f"Mean Teacher vs rescue supervised mean val AUC deltas: "
            f"100={delta_100:+.4f}, 250={delta_250:+.4f}."
        )
        if nofreeze_flags:
            details = f"{details} {' '.join(nofreeze_flags)}"
        return f"{verdict} {details}"

    if nofreeze_flags:
        return (
            "Mean Teacher results are still incomplete. "
            + " ".join(nofreeze_flags)
            + " Future SSL comparisons should include a matched no-freeze baseline."
        )

    return "Follow-up results are incomplete; wait for more Mean Teacher and no-freeze runs before making a method decision."


def build_comparison_sections(grouped: list[dict]) -> list[str]:
    lines = []

    mt_100 = lookup_group_metric(grouped, "mean_teacher", 100, "mean_val_auc")
    mt_250 = lookup_group_metric(grouped, "mean_teacher", 250, "mean_val_auc")
    sup_100 = lookup_group_metric(grouped, "supervised", 100, "mean_val_auc")
    sup_250 = lookup_group_metric(grouped, "supervised", 250, "mean_val_auc")
    if any(value is not None for value in (mt_100, mt_250)):
        lines.extend(["", "Mean Teacher Comparison", "-----------------------"])
        if mt_100 is not None and sup_100 is not None:
            lines.append(
                f"Subset 100: mean_teacher={mt_100:.4f}, rescue_supervised={sup_100:.4f}, delta={mt_100 - sup_100:+.4f}"
            )
        else:
            lines.append("Subset 100: incomplete")
        if mt_250 is not None and sup_250 is not None:
            lines.append(
                f"Subset 250: mean_teacher={mt_250:.4f}, rescue_supervised={sup_250:.4f}, delta={mt_250 - sup_250:+.4f}"
            )
        else:
            lines.append("Subset 250: incomplete")

    nofreeze_100 = lookup_group_metric(grouped, "supervised_nofreeze", 100, "mean_val_auc")
    nofreeze_250 = lookup_group_metric(grouped, "supervised_nofreeze", 250, "mean_val_auc")
    if any(value is not None for value in (nofreeze_100, nofreeze_250)):
        lines.extend(["", "Freeze Sanity Comparison", "------------------------"])
        if nofreeze_100 is not None and sup_100 is not None:
            lines.append(
                f"Subset 100: nofreeze={nofreeze_100:.4f}, rescue_supervised={sup_100:.4f}, delta={nofreeze_100 - sup_100:+.4f}"
            )
        else:
            lines.append("Subset 100: incomplete")
        if nofreeze_250 is not None and sup_250 is not None:
            lines.append(
                f"Subset 250: nofreeze={nofreeze_250:.4f}, rescue_supervised={sup_250:.4f}, delta={nofreeze_250 - sup_250:+.4f}"
            )
        else:
            lines.append("Subset 250: incomplete")

    return lines


def format_summary(rows: list[dict], grouped: list[dict], recommendation: str) -> str:
    lines = ["Follow-Up Overnight Summary", "==========================", ""]
    lines.append(
        "source   experiment                     val_auc  test_auc  sens     spec     threshold"
    )
    lines.append(
        "-------  -----------------------------  -------  --------  -------  -------  ---------"
    )
    for row in sorted(rows, key=lambda item: (item["source"], str(item["subset"]), item["method"], item["seed"])):
        lines.append(
            f"{row['source']:<7} {row['experiment']:<29} {row['val_auc']:>7.4f} "
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

    lines.extend(build_comparison_sections(grouped))

    fixmatch_rows = [row for row in rows if row["method"].startswith("fixmatch")]
    if fixmatch_rows:
        lines.extend(["", "FixMatch Health", "---------------"])
        for row in fixmatch_rows:
            if "peak_mask_ratio" not in row:
                continue
            lines.append(
                f"{row['experiment']}: peak_mask_ratio={_fmt(row.get('peak_mask_ratio'))}, "
                f"final_unsup_loss={_fmt(row.get('final_unsup_loss'))}, "
                f"final_lambda_u={_fmt(row.get('final_lambda_u'))}"
            )

    mean_teacher_rows = [row for row in rows if row["method"] == "mean_teacher"]
    if mean_teacher_rows:
        lines.extend(["", "Mean Teacher Health", "-------------------"])
        for row in mean_teacher_rows:
            if "final_consistency_loss" not in row:
                continue
            lines.append(
                f"{row['experiment']}: final_consistency_loss={_fmt(row.get('final_consistency_loss'))}, "
                f"final_consistency_weight={_fmt(row.get('final_consistency_weight'))}"
            )

    lines.extend(["", "Recommendation", "--------------", recommendation])
    return "\n".join(lines)


def _fmt(value) -> str:
    if value == "":
        return "n/a"
    return f"{float(value):.4f}"


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
        "peak_mask_ratio",
        "final_unsup_loss",
        "final_lambda_u",
        "final_consistency_loss",
        "final_consistency_weight",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize follow-up overnight experiments")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rescue_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rescue_dir = Path(args.rescue_dir) if args.rescue_dir else None

    rows = collect_rows(output_dir, source="followup")
    if rescue_dir is not None:
        rows.extend(collect_rows(rescue_dir, source="rescue"))
    grouped = group_means(rows)
    recommendation = build_recommendation(grouped)
    summary_text = format_summary(rows, grouped, recommendation)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "followup_summary.txt").write_text(summary_text)
    write_csv(output_dir / "followup_summary.csv", rows)
    print(summary_text)


if __name__ == "__main__":
    main()
