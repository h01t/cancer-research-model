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

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Add project root to path (removed when pyproject.toml is set up in Phase 3)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CBISDDSMDataset, split_labeled_unlabeled
from src.data.ssl_dataset import (
    FixMatchLabeledDataset,
    FixMatchUnlabeledDataset,
    TransformSubset,
)
from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.fixmatch_trainer import FixMatchTrainer
from src.training.trainer import get_device

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
    """Compute inverse-frequency class weights."""
    counts = dataset.get_class_counts()
    total = counts["benign"] + counts["malignant"]
    w_benign = total / (2.0 * max(counts["benign"], 1))
    w_malignant = total / (2.0 * max(counts["malignant"], 1))
    return torch.tensor([w_benign, w_malignant], dtype=torch.float32)


def create_datasets(config: dict, num_labeled: int):
    """Create labeled, unlabeled, validation, and test datasets.

    Data flow:
    1. Load raw dataset (no transforms) — returns PIL images
    2. Split into labeled/unlabeled indices (class-balanced)
    3. Split labeled into train/val
    4. Wrap with appropriate transforms:
       - Labeled train: weak augmentation
       - Unlabeled: weak + strong augmentation (same image, two views)
       - Validation: test transforms
    """
    image_size = config["dataset"]["image_size"]
    ssl_config = config["ssl"]
    aug_config = config.get("augmentation", {}).get("weak", {})

    # Build transforms
    weak_transform = get_transforms("weak", image_size=image_size, config=aug_config)
    strong_transform = get_transforms(
        "strong",
        image_size=image_size,
        n=ssl_config.get("randaugment_n", 2),
        m=ssl_config.get("randaugment_m", 10),
    )
    test_transform = get_transforms("test", image_size=image_size)

    # Load raw dataset (PIL images, no transform)
    raw_dataset = CBISDDSMDataset(
        split="train",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,  # Use all training samples
        transform=None,  # Raw PIL images
        data_dir=config["dataset"]["data_dir"],
    )

    # Split into labeled and unlabeled indices
    labeled_indices, unlabeled_indices = split_labeled_unlabeled(
        raw_dataset, num_labeled, seed=42
    )

    # Further split labeled into train/val (80/20)
    generator = torch.Generator().manual_seed(42)
    n_labeled = len(labeled_indices)
    n_val = max(1, int(0.2 * n_labeled))
    n_train = n_labeled - n_val

    perm = torch.randperm(n_labeled, generator=generator).tolist()
    train_labeled_indices = [labeled_indices[i] for i in perm[:n_train]]
    val_labeled_indices = [labeled_indices[i] for i in perm[n_train:]]

    # Build dataset wrappers
    labeled_train = FixMatchLabeledDataset(
        Subset(raw_dataset, train_labeled_indices),
        weak_transform=weak_transform,
    )

    unlabeled_train = FixMatchUnlabeledDataset(
        Subset(raw_dataset, unlabeled_indices),
        weak_transform=weak_transform,
        strong_transform=strong_transform,
    )

    val_dataset = TransformSubset(
        Subset(raw_dataset, val_labeled_indices),
        transform=test_transform,
    )

    # Test dataset
    test_dataset = CBISDDSMDataset(
        split="test",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,
        transform=test_transform,
        data_dir=config["dataset"]["data_dir"],
    )

    logger.info(
        f"Labeled train: {len(labeled_train)}, Val: {len(val_dataset)}, "
        f"Unlabeled: {len(unlabeled_train)}, Test: {len(test_dataset)}"
    )

    return raw_dataset, labeled_train, unlabeled_train, val_dataset, test_dataset


def get_num_workers(config: dict) -> int:
    """Get num_workers from config, respecting platform."""
    n = config.get("training", {}).get("num_workers", 4)
    if not torch.cuda.is_available():
        n = min(n, 4)
    return n


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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config["dataset"]["labeled_subset_size"] = args.labeled

    if args.max_epochs is not None:
        config["training"]["num_epochs"] = args.max_epochs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Datasets
    logger.info("Creating datasets...")
    raw_dataset, labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = (
        create_datasets(config, args.labeled)
    )

    # Class weights
    use_class_weights = config["training"].get("class_weighted_loss", True)
    class_weights = compute_class_weights(raw_dataset) if use_class_weights else None

    # Data loaders
    num_workers = get_num_workers(config)
    pin_memory = device.type == "cuda"
    batch_size = config["training"]["batch_size"]
    unlabeled_batch_ratio = config["ssl"]["unlabeled_batch_ratio"]

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size * unlabeled_batch_ratio,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
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
    trainer = FixMatchTrainer(
        model,
        config,
        device=device,
        output_dir=output_dir,
        class_weights=class_weights,
    )

    # Train
    logger.info("Starting FixMatch training...")
    trainer.train(labeled_loader, unlabeled_loader, val_loader)

    # Evaluate on test set (with EMA if available)
    logger.info("Evaluating on test set...")
    if trainer.ema is not None:
        trainer.ema.apply(trainer.model)
    test_metrics = trainer.evaluate(test_loader)
    if trainer.ema is not None:
        trainer.ema.restore(trainer.model)

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
