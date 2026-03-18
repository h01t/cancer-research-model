"""
SSL-specific dataset wrappers for FixMatch training.

Design principle: CBISDDSMDataset with transform=None returns raw PIL images.
These wrappers apply augmentations at the correct layer:

- FixMatchLabeledDataset: applies weak augmentation, returns (img_weak, label)
- FixMatchUnlabeledDataset: applies weak AND strong augmentation to the SAME image,
  returns (img_weak, img_strong)
- TransformSubset: applies a specified transform to a Subset (for validation)
"""

from typing import Any

from torch.utils.data import Dataset, Subset


class FixMatchLabeledDataset(Dataset):
    """Wraps a dataset (or Subset) to apply weak augmentation for labeled training.

    Input dataset must return (PIL_image, label).
    Output: (weak_augmented_tensor, label)
    """

    def __init__(self, dataset: Dataset | Subset, weak_transform: Any):
        self.dataset = dataset
        self.weak_transform = weak_transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return self.weak_transform(img), label


class FixMatchUnlabeledDataset(Dataset):
    """Wraps a dataset (or Subset) to produce weak+strong views for FixMatch.

    Input dataset must return (PIL_image, label) — label is discarded.
    Output: (weak_augmented_tensor, strong_augmented_tensor)
    """

    def __init__(
        self, dataset: Dataset | Subset, weak_transform: Any, strong_transform: Any
    ):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]
        return self.weak_transform(img), self.strong_transform(img)


class TransformSubset(Dataset):
    """Wraps a dataset (or Subset) with a different transform.

    Useful for applying test transforms to a validation split
    that was drawn from a raw (untransformed) dataset.

    Input dataset must return (PIL_image, label).
    Output: (transformed_tensor, label)
    """

    def __init__(self, dataset: Dataset | Subset, transform: Any):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return self.transform(img), label
