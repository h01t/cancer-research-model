#!/usr/bin/env python3
"""
Train FixMatch semi-supervised learning on CBIS-DDSM dataset.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CBISDDSMDataset, split_labeled_unlabeled, UnlabeledWrapper
from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.fixmatch_trainer import FixMatchTrainer


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config, num_labeled):
    """Create labeled, unlabeled, validation, and test datasets."""
    image_size = config["dataset"]["image_size"]

    # Transforms
    weak_transform = get_transforms("weak", image_size=image_size)
    strong_transform = get_transforms(
        "strong",
        image_size=image_size,
        n=config["ssl"].get("randaugment_n", 2),
        m=config["ssl"].get("randaugment_m", 10),
    )
    test_transform = get_transforms("test", image_size=image_size)

    # Full training dataset (without labeled subset restriction)
    train_dataset = CBISDDSMDataset(
        split="train",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,  # use all training samples
        transform=weak_transform,  # will be overridden for unlabeled
        data_dir=config["dataset"]["data_dir"],
    )

    # Split into labeled and unlabeled indices
    labeled_indices, unlabeled_indices = split_labeled_unlabeled(
        train_dataset, num_labeled, seed=42
    )

    # Create labeled subset (with weak transform)
    labeled_subset = Subset(train_dataset, labeled_indices)
    # Create unlabeled subset (with wrapper that strips labels)
    unlabeled_subset = UnlabeledWrapper(Subset(train_dataset, unlabeled_indices))

    # Split labeled subset further into train/val (80/20)
    labeled_size = len(labeled_subset)
    val_size = int(0.2 * labeled_size)
    train_size = labeled_size - val_size

    # Random split
    generator = torch.Generator().manual_seed(42)
    labeled_train_subset, labeled_val_subset = torch.utils.data.random_split(
        labeled_subset, [train_size, val_size], generator=generator
    )

    # Validation dataset uses test transform
    # We need to apply test transform to validation images
    # Create a wrapper that applies test transform
    class TransformSubset:
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            return self.transform(img), label

    val_subset = TransformSubset(labeled_val_subset, test_transform)

    # Test dataset
    test_dataset = CBISDDSMDataset(
        split="test",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,
        transform=test_transform,
        data_dir=config["dataset"]["data_dir"],
    )

    print(f"Labeled training samples: {len(labeled_train_subset)}")
    print(f"Labeled validation samples: {len(val_subset)}")
    print(f"Unlabeled samples: {len(unlabeled_subset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Return datasets with appropriate transforms
    # For labeled training we keep weak transform (already applied in dataset)
    # For unlabeled training we need both weak and strong transforms
    # We'll handle augmentations inside the trainer
    return labeled_train_subset, unlabeled_subset, val_subset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Train FixMatch SSL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--labeled", type=int, required=True, help="Number of labeled samples to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fixmatch",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override labeled subset size
    config["dataset"]["labeled_subset_size"] = args.labeled

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create datasets
    print("Creating datasets...")
    labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = create_datasets(
        config, args.labeled
    )

    # Create data loaders
    batch_size = config["training"]["batch_size"]
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size * config["ssl"]["unlabeled_batch_ratio"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
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
    trainer = FixMatchTrainer(model, config)

    # Train
    print("Starting FixMatch training...")
    trainer.train(labeled_loader, unlabeled_loader, val_loader)

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
