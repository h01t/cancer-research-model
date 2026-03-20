#!/usr/bin/env python3
"""
Train FixMatch semi-supervised learning on CBIS-DDSM dataset.

Usage:
    python scripts/train_fixmatch.py --config configs/default.yaml --labeled 100
    python scripts/train_fixmatch.py --config configs/default.yaml --labeled 250 --output_dir results/fixmatch_250
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import (
    build_experiment_context,
    build_fixmatch_experiment,
    compute_class_weights,
    create_model,
    create_trainer,
    evaluate_and_persist_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train FixMatch SSL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--labeled",
        type=int,
        required=True,
        help="Number of labeled samples to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fixmatch",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max epochs from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible data splits and training",
    )
    args = parser.parse_args()

    overrides = {
        "dataset": {"labeled_subset_size": args.labeled},
        "experiment": {"method": "fixmatch"},
    }
    if args.max_epochs is not None:
        overrides.setdefault("training", {})["num_epochs"] = args.max_epochs

    context = build_experiment_context(
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        device_override=args.device,
        overrides=overrides,
    )
    logger.info(f"Using device: {context.device}")

    bundle = build_fixmatch_experiment(
        context.config,
        context.device,
        seed=context.seed,
        num_labeled=args.labeled,
    )
    logger.info(
        "Labeled train: %s, Val: %s, Unlabeled: %s, Test: %s",
        len(bundle.datasets.labeled_dataset),
        len(bundle.datasets.val_dataset),
        len(bundle.datasets.unlabeled_dataset),
        len(bundle.datasets.test_dataset),
    )

    logger.info("Creating model...")
    model = create_model(context.config)
    use_class_weights = context.config["training"].get("class_weighted_loss", True)
    class_weights = (
        compute_class_weights(bundle.datasets.raw_dataset) if use_class_weights else None
    )
    trainer = create_trainer(
        "fixmatch",
        model,
        context,
        class_weights,
    )

    logger.info("Starting FixMatch training...")
    trainer.train(
        bundle.loaders.labeled_loader,
        bundle.loaders.unlabeled_loader,
        bundle.loaders.val_loader,
    )
    logger.info("Evaluating on test set...")
    threshold, val_metrics, test_metrics = evaluate_and_persist_results(
        trainer,
        context.output_dir,
        bundle.loaders.val_loader,
        bundle.loaders.test_loader,
        ema=trainer.ema,
    )

    print(f"\nValidation threshold: {threshold:.4f}")
    print("\nValidation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    logger.info(f"Results saved to {context.output_dir}")


if __name__ == "__main__":
    main()
