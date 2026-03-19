#!/usr/bin/env python3
"""
Train Mean Teacher semi-supervised learning on CBIS-DDSM.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CBISDDSMDataset, patient_aware_split
from src.data.sampling import sample_balanced_labeled_indices
from src.data.ssl_dataset import (
    FixMatchLabeledDataset,
    TeacherStudentUnlabeledDataset,
    TransformSubset,
)
from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.mean_teacher_trainer import MeanTeacherTrainer
from src.training.trainer import get_device
from src.training.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset: CBISDDSMDataset) -> torch.Tensor:
    counts = dataset.get_class_counts()
    total = counts["benign"] + counts["malignant"]
    w_benign = total / (2.0 * max(counts["benign"], 1))
    w_malignant = total / (2.0 * max(counts["malignant"], 1))
    return torch.tensor([w_benign, w_malignant], dtype=torch.float32)


def create_datasets(config: dict, num_labeled: int, seed: int = 42):
    image_size = config["dataset"]["image_size"]
    aug_cfg = config.get("augmentation", {})
    student_cfg = aug_cfg.get("mild_strong", {})
    teacher_aug_name = config["ssl"].get("teacher_augmentation", "weak")
    student_aug_name = config["ssl"].get("student_augmentation", "mild_strong")

    def build_transform(aug_name: str):
        if aug_name == "weak":
            return get_transforms(
                "weak",
                image_size=image_size,
                config=aug_cfg.get("weak", {}),
            )
        if aug_name == "mild_strong":
            return get_transforms(
                "mild_strong",
                image_size=image_size,
                **aug_cfg.get("mild_strong", {}),
            )
        return get_transforms(aug_name, image_size=image_size)

    weak_transform = build_transform("weak")
    teacher_transform = build_transform(teacher_aug_name)
    student_transform = build_transform(student_aug_name)
    test_transform = get_transforms("test", image_size=image_size)

    raw_dataset = CBISDDSMDataset(
        split="train",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,
        transform=None,
        data_dir=config["dataset"]["data_dir"],
    )

    val_fraction = config.get("training", {}).get("val_split_ratio", 0.15)
    train_pool_indices, val_indices = patient_aware_split(
        raw_dataset, val_fraction=val_fraction, seed=seed
    )

    import random as _random

    rng = _random.Random(seed)
    labeled_indices = sample_balanced_labeled_indices(
        train_pool_indices,
        raw_dataset.labels,
        num_labeled,
        seed,
    )
    rng.shuffle(labeled_indices)

    labeled_set = set(labeled_indices)
    unlabeled_indices = [i for i in train_pool_indices if i not in labeled_set]
    rng.shuffle(unlabeled_indices)

    labeled_train = FixMatchLabeledDataset(Subset(raw_dataset, labeled_indices), weak_transform)
    unlabeled_train = TeacherStudentUnlabeledDataset(
        Subset(raw_dataset, unlabeled_indices),
        teacher_transform=teacher_transform,
        student_transform=student_transform,
    )
    val_dataset = TransformSubset(Subset(raw_dataset, val_indices), test_transform)

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
    n = config.get("training", {}).get("num_workers", 4)
    if not torch.cuda.is_available():
        n = min(n, 4)
    return n


def main():
    parser = argparse.ArgumentParser(description="Train Mean Teacher SSL")
    parser.add_argument("--config", type=str, default="configs/mean_teacher.yaml")
    parser.add_argument("--labeled", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="results/mean_teacher")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    config["dataset"]["labeled_subset_size"] = args.labeled
    config.setdefault("experiment", {})["seed"] = args.seed
    if args.max_epochs is not None:
        config["training"]["num_epochs"] = args.max_epochs

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    raw_dataset, labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = create_datasets(
        config, args.labeled, seed=args.seed
    )

    use_class_weights = config["training"].get("class_weighted_loss", True)
    class_weights = compute_class_weights(raw_dataset) if use_class_weights else None

    num_workers = get_num_workers(config)
    pin_memory = device.type == "cuda"
    batch_size = config["training"]["batch_size"]
    unlabeled_batch_ratio = config["ssl"].get("unlabeled_batch_ratio", 2)

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

    model = EfficientNetClassifier(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout_rate"],
        freeze_backbone=config["model"].get("freeze_backbone", False),
    )

    trainer = MeanTeacherTrainer(
        model,
        config,
        device=device,
        output_dir=output_dir,
        class_weights=class_weights,
    )

    logger.info("Starting Mean Teacher training...")
    trainer.train(labeled_loader, unlabeled_loader, val_loader)

    best_checkpoint = output_dir / "best_model.pth"
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)

    trainer.ema.apply(trainer.model)
    try:
        threshold, val_metrics = trainer.tune_decision_threshold(val_loader)
        test_metrics = trainer.evaluate(test_loader, decision_threshold=threshold)
    finally:
        trainer.ema.restore(trainer.model)

    print(f"\nValidation threshold: {threshold:.4f}")
    print("\nValidation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    with open(output_dir / "val_metrics.yaml", "w") as f:
        yaml.dump({k: float(v) for k, v in val_metrics.items()}, f)
    with open(output_dir / "test_metrics.yaml", "w") as f:
        yaml.dump({k: float(v) for k, v in test_metrics.items()}, f)

    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
