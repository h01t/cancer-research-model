"""Dataset registry for supervised and future multi-dataset experiments."""

from __future__ import annotations

from typing import Any

from .dataset import CBISDDSMDataset


def build_dataset(
    dataset_name: str,
    split: str,
    abnormality_type: str,
    data_dir: str,
    labeled_subset_size: int | None = None,
    transform: Any = None,
):
    """Build a registered dataset by name."""
    dataset_name = dataset_name.lower()
    if dataset_name != "cbis-ddsm":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return CBISDDSMDataset(
        split=split,
        abnormality_type=abnormality_type,
        labeled_subset_size=labeled_subset_size,
        transform=transform,
        data_dir=data_dir,
    )
