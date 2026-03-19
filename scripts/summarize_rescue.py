#!/usr/bin/env python3
"""
Summarize the targeted FixMatch rescue sweep and apply the promotion gate.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Summarize rescue sweep results")
    parser.add_argument("--output_dir", type=str, default="results_rescue")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    seeds = args.seeds
    subsets = [100, 250]

    print("Targeted Rescue Summary")
    print("======================")

    means: dict[tuple[str, int], float] = {}
    for subset in subsets:
        print(f"\nSubset {subset}")
        print("method      mean_val_auc  mean_test_auc  mean_sens  mean_spec")
        print("----------  ------------  -------------  ---------  ---------")
        for method in ["supervised", "fixmatch"]:
            metrics = collect_method_metrics(output_dir, method, subset, seeds)
            if not metrics:
                print(f"{method:<10} MISSING")
                continue

            val_aucs = [float(m["val"]["auc"]) for m in metrics]
            test_aucs = [float(m["test"]["auc"]) for m in metrics]
            sens = [float(m["test"]["sensitivity"]) for m in metrics]
            spec = [float(m["test"]["specificity"]) for m in metrics]
            means[(method, subset)] = mean(val_aucs)

            print(
                f"{method:<10} {mean(val_aucs):>12.4f} {mean(test_aucs):>14.4f} "
                f"{mean(sens):>10.4f} {mean(spec):>10.4f}"
            )

    fix_100 = means.get(("fixmatch", 100))
    sup_100 = means.get(("supervised", 100))
    fix_250 = means.get(("fixmatch", 250))
    sup_250 = means.get(("supervised", 250))

    print("\nPromotion Gate")
    print("--------------")
    if None in (fix_100, sup_100, fix_250, sup_250):
        print("Insufficient completed runs to evaluate the gate.")
        return

    delta_100 = fix_100 - sup_100
    delta_250 = fix_250 - sup_250
    keep_fixmatch = delta_100 >= 0.02 and delta_250 >= -0.01

    print(f"Subset 100 delta (FixMatch - Supervised): {delta_100:+.4f}")
    print(f"Subset 250 delta (FixMatch - Supervised): {delta_250:+.4f}")
    print(f"Decision: {'KEEP FIXMATCH' if keep_fixmatch else 'PIVOT TO MEAN TEACHER'}")


if __name__ == "__main__":
    main()
