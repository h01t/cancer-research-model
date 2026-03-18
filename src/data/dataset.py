import os
import logging
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CBISDDSMDataset(Dataset):
    """PyTorch Dataset for CBIS-DDSM mammography images.

    When transform is None, returns raw PIL images.
    When transform is provided, returns transformed tensors.
    """

    def __init__(
        self,
        split: str = "train",
        abnormality_type: str = "mass",
        labeled_subset_size: int | None = None,
        transform: Any = None,
        data_dir: str = "data",
    ):
        """
        Args:
            split: 'train' or 'test'
            abnormality_type: 'mass', 'calc', or 'both'
            labeled_subset_size: number of labeled samples to use (if None, use all)
            transform: torchvision transforms to apply (None returns raw PIL)
            data_dir: root directory containing 'csv' and 'jpeg' folders
        """
        self.split = split
        self.abnormality_type = abnormality_type
        self.labeled_subset_size = labeled_subset_size
        self.transform = transform
        self.data_dir = data_dir

        # Load and merge data
        self.df = self._load_and_merge_data()

        # Filter labeled subset if specified
        if labeled_subset_size is not None:
            self.df = self._sample_labeled_subset(labeled_subset_size)

        # Extract image paths and labels
        self.image_paths = self.df["jpeg_path"].tolist()
        self.labels = self.df["label"].tolist()

        logger.info(f"Loaded {len(self)} samples ({split}, {abnormality_type})")

    def _load_and_merge_data(self) -> pd.DataFrame:
        """Load dicom_info and case description CSVs, merge them."""
        # Load dicom_info.csv
        dicom_info_path = os.path.join(self.data_dir, "csv", "dicom_info.csv")
        dicom_df = pd.read_csv(dicom_info_path)

        # Filter for full mammogram images (use .copy() to avoid SettingWithCopyWarning)
        full_mammogram_df = dicom_df[
            dicom_df["SeriesDescription"] == "full mammogram images"
        ].copy()

        # Load appropriate case description CSV(s)
        dfs: list[pd.DataFrame] = []
        if self.abnormality_type in ["mass", "both"]:
            mass_csv = f"mass_case_description_{self.split}_set.csv"
            mass_path = os.path.join(self.data_dir, "csv", mass_csv)
            mass_df = pd.read_csv(mass_path)
            mass_df["abnormality_type"] = "mass"
            dfs.append(mass_df)
        if self.abnormality_type in ["calc", "both"]:
            calc_csv = f"calc_case_description_{self.split}_set.csv"
            calc_path = os.path.join(self.data_dir, "csv", calc_csv)
            calc_df = pd.read_csv(calc_path)
            calc_df["abnormality_type"] = "calc"
            dfs.append(calc_df)

        if not dfs:
            raise ValueError(f"Unknown abnormality_type: {self.abnormality_type}")

        case_df: pd.DataFrame = (
            pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0].copy()
        )

        # Clean case_df: remove rows with missing pathology
        case_df = case_df.dropna(subset=["pathology"])

        # Create matching key in case_df
        # Build key: "{abnormality_type}-{split}_P_{patient_id}_{laterality}_{view}"
        split_cap = "Training" if self.split == "train" else "Test"
        case_df["patient_key"] = case_df.apply(
            lambda row: (
                f"{row['abnormality_type'].capitalize()}-{split_cap}_"
                + f"P_{row['patient_id'].split('_')[-1]}_{row['left or right breast']}_{row['image view']}"
            ),
            axis=1,
        )

        # Create patient_key in full_mammogram_df from PatientID
        full_mammogram_df["patient_key"] = full_mammogram_df["PatientID"]

        # Merge on patient_key (exact match first)
        merged_df = pd.merge(
            case_df,
            full_mammogram_df[["patient_key", "image_path", "SeriesInstanceUID"]],
            on="patient_key",
            how="inner",
        )

        # If exact merge fails, try prefix matching
        if len(merged_df) == 0:
            logger.info("Exact match failed, trying prefix matching...")
            # Remove possible trailing abnormality suffix (_1, _2, etc.)
            case_df["patient_prefix"] = case_df["patient_key"].apply(
                lambda x: x.rsplit("_", 1)[0] if x.rsplit("_", 1)[-1].isdigit() else x
            )
            full_mammogram_df["patient_prefix"] = full_mammogram_df[
                "patient_key"
            ].apply(
                lambda x: x.rsplit("_", 1)[0] if x.rsplit("_", 1)[-1].isdigit() else x
            )
            merged_df = pd.merge(
                case_df,
                full_mammogram_df[
                    ["patient_prefix", "image_path", "SeriesInstanceUID"]
                ],
                on="patient_prefix",
                how="inner",
            )

        if len(merged_df) == 0:
            raise ValueError(
                f"No matching images found for {self.abnormality_type} {self.split}"
            )

        # Convert image_path to local file path
        # image_path format: "CBIS-DDSM/jpeg/SeriesInstanceUID/1-XXX.jpg"
        merged_df["jpeg_path"] = merged_df["image_path"].apply(
            lambda x: os.path.join(self.data_dir, x.replace("CBIS-DDSM/", ""))
        )

        # Create binary labels: MALIGNANT = 1, BENIGN or BENIGN_WITHOUT_CALLBACK = 0
        merged_df["label"] = merged_df["pathology"].apply(
            lambda x: 1 if x == "MALIGNANT" else 0
        )

        # Remove duplicate entries (same image multiple abnormalities)
        merged_df = merged_df.drop_duplicates(subset=["jpeg_path"])

        return merged_df

    def _sample_labeled_subset(self, n):
        """Sample n labeled examples, preserving class balance."""
        malignant = self.df[self.df["label"] == 1]
        benign = self.df[self.df["label"] == 0]

        n_malignant = min(len(malignant), n // 2)
        n_benign = min(len(benign), n - n_malignant)

        if n_malignant + n_benign < n:
            print(
                f"Warning: requested {n} samples but only {n_malignant + n_benign} available"
            )

        sampled = pd.concat(
            [
                malignant.sample(n_malignant, random_state=42),
                benign.sample(n_benign, random_state=42),
            ]
        )

        return sampled.reset_index(drop=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, IOError) as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a blank image of expected size to avoid crashing training
            img = Image.new("RGB", (512, 512), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_counts(self) -> dict[str, int]:
        """Return counts of benign and malignant samples."""
        benign = sum(1 for label in self.labels if label == 0)
        malignant = sum(1 for label in self.labels if label == 1)
        return {"benign": benign, "malignant": malignant}


def split_labeled_unlabeled(
    dataset: CBISDDSMDataset, num_labeled: int, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split dataset into labeled and unlabeled subsets, preserving class balance.

    Args:
        dataset: CBISDDSMDataset instance
        num_labeled: number of labeled samples
        seed: random seed for reproducibility

    Returns:
        (labeled_indices, unlabeled_indices)
    """
    import random as _random

    rng = _random.Random(seed)

    # Get indices of each class
    benign_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    malignant_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

    # Ensure we have enough samples per class
    num_labeled_per_class = num_labeled // 2
    if (
        len(benign_indices) < num_labeled_per_class
        or len(malignant_indices) < num_labeled_per_class
    ):
        raise ValueError(
            f"Insufficient samples per class for labeled subset {num_labeled}"
        )

    # Sample using local RNG (no global state mutation)
    labeled_benign = rng.sample(benign_indices, num_labeled_per_class)
    labeled_malignant = rng.sample(malignant_indices, num_labeled_per_class)
    labeled_indices = labeled_benign + labeled_malignant

    # Remaining indices are unlabeled (set lookup for O(n) instead of O(n*m))
    labeled_set = set(labeled_indices)
    unlabeled_indices = [i for i in range(len(dataset)) if i not in labeled_set]

    # Shuffle
    rng.shuffle(labeled_indices)
    rng.shuffle(unlabeled_indices)

    return labeled_indices, unlabeled_indices
