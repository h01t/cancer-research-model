#!/usr/bin/env python3
"""Summarize the targeted FixMatch rescue sweep and apply the promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_metric(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def collect_method_metrics(output_dir: Path, method: str, subset: int, seeds: list[int]) -> list[dict]:
    metrics = []
    for seed in seeds:
        exp_dir = output_dir / f"{method}_{subset}_seed{seed}"
        val_metrics = exp_dir / "val_metrics.yaml"
        test_metrics = exp_dir / "test_metrics.yaml"
        if val_metrics.exists() and test_metrics.exists():
            metrics.append(
                {
                    "seed": seed,
                    "val": load_metric(val_metrics),
                    "test": load_metric(test_metrics),
                }
            )
    return metrics


def build_rescue_summary(output_dir: Path, seeds: list[int]) -> dict:
    subsets = [100, 250]
    subset_rows: list[dict] = []
    means: dict[tuple[str, int], float] = {}

    for subset in subsets:
        method_rows = []
        for method in ["supervised", "fixmatch"]:
            metrics = collect_method_metrics(output_dir, method, subset, seeds)
            if not metrics:
                method_rows.append({"method": method, "status": "missing"})
                continue

            val_aucs = [float(m["val"]["auc"]) for m in metrics]
            test_aucs = [float(m["test"]["auc"]) for m in metrics]
            sens = [float(m["test"]["sensitivity"]) for m in metrics]
            spec = [float(m["test"]["specificity"]) for m in metrics]
            means[(method, subset)] = mean(val_aucs)
            method_rows.append(
                {
                    "method": method,
                    "status": "ok",
                    "completed_seeds": [int(m["seed"]) for m in metrics],
                    "mean_val_auc": mean(val_aucs),
                    "mean_test_auc": mean(test_aucs),
                    "mean_sensitivity": mean(sens),
                    "mean_specificity": mean(spec),
                }
            )
        subset_rows.append({"subset": subset, "methods": method_rows})

    fix_100 = means.get(("fixmatch", 100))
    sup_100 = means.get(("supervised", 100))
    fix_250 = means.get(("fixmatch", 250))
    sup_250 = means.get(("supervised", 250))

    promotion_gate = {
        "ready": None not in (fix_100, sup_100, fix_250, sup_250),
        "delta_100": None if fix_100 is None or sup_100 is None else fix_100 - sup_100,
        "delta_250": None if fix_250 is None or sup_250 is None else fix_250 - sup_250,
        "decision": "insufficient_data",
        "decision_label": "Insufficient completed runs to evaluate the gate.",
    }

    if promotion_gate["ready"]:
        keep_fixmatch = (
            promotion_gate["delta_100"] >= 0.02 and promotion_gate["delta_250"] >= -0.01
        )
        promotion_gate["decision"] = "keep_fixmatch" if keep_fixmatch else "pivot_to_mean_teacher"
        promotion_gate["decision_label"] = (
            "KEEP FIXMATCH" if keep_fixmatch else "PIVOT TO MEAN TEACHER"
        )

    return {
        "output_dir": str(output_dir),
        "seeds": [int(seed) for seed in seeds],
        "subsets": subset_rows,
        "promotion_gate": promotion_gate,
    }


def format_rescue_summary(summary: dict) -> str:
    lines = ["Targeted Rescue Summary", "======================"]

    for subset_summary in summary["subsets"]:
        lines.append(f"\nSubset {subset_summary['subset']}")
        lines.append("method      mean_val_auc  mean_test_auc  mean_sens  mean_spec")
        lines.append("----------  ------------  -------------  ---------  ---------")
        for method_summary in subset_summary["methods"]:
            method = method_summary["method"]
            if method_summary["status"] != "ok":
                lines.append(f"{method:<10} MISSING")
                continue

            lines.append(
                f"{method:<10} {method_summary['mean_val_auc']:>12.4f} "
                f"{method_summary['mean_test_auc']:>14.4f} "
                f"{method_summary['mean_sensitivity']:>10.4f} "
                f"{method_summary['mean_specificity']:>10.4f}"
            )

    gate = summary["promotion_gate"]
    lines.append("\nPromotion Gate")
    lines.append("--------------")
    if not gate["ready"]:
        lines.append(gate["decision_label"])
        return "\n".join(lines)

    lines.append(f"Subset 100 delta (FixMatch - Supervised): {gate['delta_100']:+.4f}")
    lines.append(f"Subset 250 delta (FixMatch - Supervised): {gate['delta_250']:+.4f}")
    lines.append(f"Decision: {gate['decision_label']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize rescue sweep results")
    parser.add_argument("--output_dir", type=str, default="results_rescue")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON summary instead of text output",
    )
    args = parser.parse_args()

    summary = build_rescue_summary(Path(args.output_dir), args.seeds)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_rescue_summary(summary))


if __name__ == "__main__":
    main()
