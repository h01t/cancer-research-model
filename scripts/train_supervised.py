#!/usr/bin/env python3
"""
Train supervised baseline on CBIS-DDSM dataset.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CBISDDSMDataset
from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.trainer import BaseTrainer


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config, labeled_subset_size=None):
    """Create train, validation, and test datasets."""
    image_size = config["dataset"]["image_size"]

    # Transforms
    train_transform = get_transforms("weak", image_size=image_size)
    val_transform = get_transforms("test", image_size=image_size)
    test_transform = get_transforms("test", image_size=image_size)

    # Training dataset (full training split)
    train_dataset = CBISDDSMDataset(
        split="train",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=labeled_subset_size,
        transform=train_transform,
        data_dir=config["dataset"]["data_dir"],
    )

    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Test dataset
    test_dataset = CBISDDSMDataset(
        split="test",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,  # use all test samples
        transform=test_transform,
        data_dir=config["dataset"]["data_dir"],
    )

    print(f"Train subset: {len(train_subset)} samples")
    print(f"Val subset: {len(val_subset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    return train_subset, val_subset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Train supervised baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--labeled_subset",
        type=int,
        default=None,
        help="Number of labeled samples to use (for ablation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/supervised",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override labeled subset size if specified
    if args.labeled_subset is not None:
        config["dataset"]["labeled_subset_size"] = args.labeled_subset

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        config, labeled_subset_size=config["dataset"].get("labeled_subset_size")
    )

    # Create data loaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    model = EfficientNetClassifier(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout_rate"],
    )

    # Create trainer
    trainer = BaseTrainer(model, config)

    # Train
    print("Starting training...")
    trainer.train(train_loader, val_loader)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save test metrics
    with open(output_dir / "test_metrics.yaml", "w") as f:
        yaml.dump(test_metrics, f)

    # Save training history
    import pandas as pd

    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
