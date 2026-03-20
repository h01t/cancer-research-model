"""Dataset and dataloader builders for supervised and SSL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import (
    CBISDDSMDataset,
    FixMatchLabeledDataset,
    FixMatchUnlabeledDataset,
    TeacherStudentUnlabeledDataset,
    TransformSubset,
    get_transforms,
    patient_aware_split,
    sample_balanced_labeled_indices,
)


@dataclass
class DatasetBundle:
    raw_dataset: CBISDDSMDataset
    train_dataset: Dataset | None
    labeled_dataset: Dataset | None
    unlabeled_dataset: Dataset | None
    val_dataset: Dataset
    test_dataset: Dataset


@dataclass
class LoaderBundle:
    train_loader: DataLoader | None
    labeled_loader: DataLoader | None
    unlabeled_loader: DataLoader | None
    val_loader: DataLoader
    test_loader: DataLoader


@dataclass
class ExperimentBundle:
    datasets: DatasetBundle
    loaders: LoaderBundle


def _get_num_workers(config: dict, device: torch.device) -> int:
    n = config.get("training", {}).get("num_workers", 4)
    if device.type != "cuda":
        n = min(n, 4)
    return n


def _get_common_loader_kwargs(config: dict, device: torch.device) -> dict[str, Any]:
    return {
        "num_workers": _get_num_workers(config, device),
        "pin_memory": device.type == "cuda",
    }


def _build_raw_train_dataset(config: dict) -> CBISDDSMDataset:
    return CBISDDSMDataset(
        split="train",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,
        transform=None,
        data_dir=config["dataset"]["data_dir"],
    )


def _build_test_dataset(config: dict, test_transform: Any) -> CBISDDSMDataset:
    return CBISDDSMDataset(
        split="test",
        abnormality_type=config["dataset"]["abnormality_type"],
        labeled_subset_size=None,
        transform=test_transform,
        data_dir=config["dataset"]["data_dir"],
    )


def _split_train_pool(config: dict, raw_dataset: CBISDDSMDataset, seed: int) -> tuple[list[int], list[int]]:
    val_fraction = config.get("training", {}).get("val_split_ratio", 0.15)
    return patient_aware_split(raw_dataset, val_fraction=val_fraction, seed=seed)


def _build_common_transforms(config: dict) -> tuple[Any, Any, dict]:
    image_size = config["dataset"]["image_size"]
    augmentation_cfg = config.get("augmentation", {})
    weak_transform = get_transforms(
        "weak",
        image_size=image_size,
        config=augmentation_cfg.get("weak", {}),
    )
    test_transform = get_transforms("test", image_size=image_size)
    return weak_transform, test_transform, augmentation_cfg


def build_supervised_experiment(
    config: dict,
    device: torch.device,
    seed: int,
    labeled_subset_size: int | None = None,
) -> ExperimentBundle:
    weak_transform, test_transform, _ = _build_common_transforms(config)
    raw_dataset = _build_raw_train_dataset(config)
    train_pool_indices, val_indices = _split_train_pool(config, raw_dataset, seed)

    train_indices = (
        sample_balanced_labeled_indices(train_pool_indices, raw_dataset.labels, labeled_subset_size, seed)
        if labeled_subset_size is not None
        else train_pool_indices
    )

    datasets = DatasetBundle(
        raw_dataset=raw_dataset,
        train_dataset=TransformSubset(Subset(raw_dataset, train_indices), weak_transform),
        labeled_dataset=None,
        unlabeled_dataset=None,
        val_dataset=TransformSubset(Subset(raw_dataset, val_indices), test_transform),
        test_dataset=_build_test_dataset(config, test_transform),
    )

    loader_kwargs = _get_common_loader_kwargs(config, device)
    batch_size = config["training"]["batch_size"]
    loaders = LoaderBundle(
        train_loader=DataLoader(
            datasets.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        ),
        labeled_loader=None,
        unlabeled_loader=None,
        val_loader=DataLoader(
            datasets.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        test_loader=DataLoader(
            datasets.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    )
    return ExperimentBundle(datasets=datasets, loaders=loaders)


def build_fixmatch_experiment(
    config: dict,
    device: torch.device,
    seed: int,
    num_labeled: int,
) -> ExperimentBundle:
    weak_transform, test_transform, augmentation_cfg = _build_common_transforms(config)
    image_size = config["dataset"]["image_size"]
    ssl_cfg = config["ssl"]
    raw_dataset = _build_raw_train_dataset(config)
    train_pool_indices, val_indices = _split_train_pool(config, raw_dataset, seed)

    labeled_indices = sample_balanced_labeled_indices(
        train_pool_indices, raw_dataset.labels, num_labeled, seed
    )
    labeled_set = set(labeled_indices)
    unlabeled_indices = [idx for idx in train_pool_indices if idx not in labeled_set]

    strong_transform = get_transforms(
        "mild_strong",
        image_size=image_size,
        n=ssl_cfg.get("randaugment_n", 2),
        m=ssl_cfg.get("randaugment_m", 10),
        **augmentation_cfg.get("mild_strong", {}),
    )

    datasets = DatasetBundle(
        raw_dataset=raw_dataset,
        train_dataset=None,
        labeled_dataset=FixMatchLabeledDataset(Subset(raw_dataset, labeled_indices), weak_transform),
        unlabeled_dataset=FixMatchUnlabeledDataset(
            Subset(raw_dataset, unlabeled_indices),
            weak_transform=weak_transform,
            strong_transform=strong_transform,
        ),
        val_dataset=TransformSubset(Subset(raw_dataset, val_indices), test_transform),
        test_dataset=_build_test_dataset(config, test_transform),
    )

    loader_kwargs = _get_common_loader_kwargs(config, device)
    batch_size = config["training"]["batch_size"]
    unlabeled_batch_ratio = ssl_cfg["unlabeled_batch_ratio"]
    loaders = LoaderBundle(
        train_loader=None,
        labeled_loader=DataLoader(
            datasets.labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        unlabeled_loader=DataLoader(
            datasets.unlabeled_dataset,
            batch_size=batch_size * unlabeled_batch_ratio,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        val_loader=DataLoader(
            datasets.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        test_loader=DataLoader(
            datasets.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    )
    return ExperimentBundle(datasets=datasets, loaders=loaders)


def build_mean_teacher_experiment(
    config: dict,
    device: torch.device,
    seed: int,
    num_labeled: int,
) -> ExperimentBundle:
    weak_transform, test_transform, augmentation_cfg = _build_common_transforms(config)
    image_size = config["dataset"]["image_size"]
    ssl_cfg = config["ssl"]
    raw_dataset = _build_raw_train_dataset(config)
    train_pool_indices, val_indices = _split_train_pool(config, raw_dataset, seed)

    labeled_indices = sample_balanced_labeled_indices(
        train_pool_indices, raw_dataset.labels, num_labeled, seed
    )
    labeled_set = set(labeled_indices)
    unlabeled_indices = [idx for idx in train_pool_indices if idx not in labeled_set]

    def build_aug(name: str):
        if name == "weak":
            return weak_transform
        if name == "mild_strong":
            return get_transforms(
                "mild_strong",
                image_size=image_size,
                **augmentation_cfg.get("mild_strong", {}),
            )
        return get_transforms(name, image_size=image_size)

    teacher_transform = build_aug(ssl_cfg.get("teacher_augmentation", "weak"))
    student_transform = build_aug(ssl_cfg.get("student_augmentation", "mild_strong"))

    datasets = DatasetBundle(
        raw_dataset=raw_dataset,
        train_dataset=None,
        labeled_dataset=FixMatchLabeledDataset(Subset(raw_dataset, labeled_indices), weak_transform),
        unlabeled_dataset=TeacherStudentUnlabeledDataset(
            Subset(raw_dataset, unlabeled_indices),
            teacher_transform=teacher_transform,
            student_transform=student_transform,
        ),
        val_dataset=TransformSubset(Subset(raw_dataset, val_indices), test_transform),
        test_dataset=_build_test_dataset(config, test_transform),
    )

    loader_kwargs = _get_common_loader_kwargs(config, device)
    batch_size = config["training"]["batch_size"]
    unlabeled_batch_ratio = ssl_cfg.get("unlabeled_batch_ratio", 2)
    loaders = LoaderBundle(
        train_loader=None,
        labeled_loader=DataLoader(
            datasets.labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        unlabeled_loader=DataLoader(
            datasets.unlabeled_dataset,
            batch_size=batch_size * unlabeled_batch_ratio,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        val_loader=DataLoader(
            datasets.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        test_loader=DataLoader(
            datasets.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    )
    return ExperimentBundle(datasets=datasets, loaders=loaders)
