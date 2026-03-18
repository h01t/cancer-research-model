#!/usr/bin/env python3
"""
Train supervised baseline on CBIS-DDSM dataset.

Usage:
    python scripts/train_supervised.py --config configs/default.yaml --labeled_subset 100
    python scripts/train_supervised.py --config configs/test.yaml --output_dir results/test
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Add project root to path (removed when pyproject.toml is set up in Phase 3)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CBISDDSMDataset
from src.data.ssl_dataset import TransformSubset
from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.trainer import BaseTrainer, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset: CBISDDSMDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced training."""
    counts = dataset.get_class_counts()
    total = counts["benign"] + counts["malignant"]
    # Inverse frequency, normalized
    w_benign = total / (2.0 * max(counts["benign"], 1))
    w_malignant = total / (2.0 * max(counts["malignant"], 1))
    weights = torch.tensor([w_benign, w_malignant], dtype=torch.float32)
    logger.info(f"Class weights: benign={w_benign:.3f}, malignant={w_malignant:.3f}")
    return weights


def create_datasets(config: dict, labeled_subset_size: int | None = None):
    """Create train, validation, and test datasets.

    Returns raw PIL datasets split into train/val, plus a test dataset.
    Transforms are applied via wrapper datasets to ensure val gets test transforms.
    """
    image_size = config["dataset"]["image_size"]
    aug_config = config.get("augmentation", {}).get("weak", {})

    train_transform = get_transforms("weak", image_size=image_size, config=aug_config)
    test_transform = get_transforms("test", image_size=image_size)

    # Load raw dataset (no transform — returns PIL images)
    raw_dataset = CBISDDSMDataset(
        split="train",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=labeled_subset_size,
        transform=None,  # Raw PIL images
        data_dir=config["dataset"]["data_dir"],
    )

    # Split into train/val (80/20) using indices
    total = len(raw_dataset)
    train_size = int(0.8 * total)
    indices = list(range(total))

    # Deterministic split
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(total, generator=generator).tolist()
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    # Wrap with appropriate transforms
    train_subset = TransformSubset(Subset(raw_dataset, train_indices), train_transform)
    val_subset = TransformSubset(Subset(raw_dataset, val_indices), test_transform)

    # Test dataset (with test transform applied directly)
    test_dataset = CBISDDSMDataset(
        split="test",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,
        transform=test_transform,
        data_dir=config["dataset"]["data_dir"],
    )

    logger.info(
        f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}"
    )
    return raw_dataset, train_subset, val_subset, test_dataset


def get_num_workers(config: dict) -> int:
    """Get num_workers from config, respecting platform."""
    n = config.get("training", {}).get("num_workers", 4)
    # MPS/CPU on macOS can have issues with multiprocessing
    if not torch.cuda.is_available():
        n = min(n, 4)
    return n


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
        help="Number of labeled samples (None = all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/supervised",
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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.labeled_subset is not None:
        config["dataset"]["labeled_subset_size"] = args.labeled_subset
    if args.max_epochs is not None:
        config["training"]["num_epochs"] = args.max_epochs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Datasets
    logger.info("Creating datasets...")
    raw_dataset, train_dataset, val_dataset, test_dataset = create_datasets(
        config, labeled_subset_size=config["dataset"].get("labeled_subset_size")
    )

    # Class weights
    use_class_weights = config["training"].get("class_weighted_loss", True)
    class_weights = compute_class_weights(raw_dataset) if use_class_weights else None

    # Data loaders
    num_workers = get_num_workers(config)
    pin_memory = device.type == "cuda"
    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Model
    logger.info("Creating model...")
    model = EfficientNetClassifier(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout_rate"],
    )

    # Trainer
    trainer = BaseTrainer(
        model,
        config,
        device=device,
        output_dir=output_dir,
        class_weights=class_weights,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    with open(output_dir / "test_metrics.yaml", "w") as f:
        yaml.dump({k: float(v) for k, v in test_metrics.items()}, f)

    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
