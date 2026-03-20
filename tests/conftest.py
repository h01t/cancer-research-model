"""
Shared test fixtures for SSL mammography tests.

All tests use synthetic data — no real CBIS-DDSM dataset required.
"""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def base_config():
    """Minimal training config for tests."""
    return {
        "dataset": {
            "name": "cbis-ddsm",
            "data_dir": "data",
            "abnormality_type": "mass",
            "image_size": 32,
            "labeled_subset_size": None,
        },
        "model": {
            "name": "efficientnet-b0",
            "num_classes": 2,
            "pretrained": False,
            "dropout_rate": 0.2,
            "freeze_backbone": False,
        },
        "training": {
            "batch_size": 2,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "warmup_epochs": 0,
            "early_stopping_patience": 5,
            "num_workers": 0,
            "use_amp": False,
            "gradient_clipping": True,
            "max_grad_norm": 1.0,
            "class_weighted_loss": False,
        },
        "ssl": {
            "method": "fixmatch",
            "confidence_threshold": 0.95,
            "lambda_u": 1.0,
            "unlabeled_batch_ratio": 2,
            "randaugment_n": 2,
            "randaugment_m": 10,
            "use_ema": False,
            "ema_decay": 0.999,
        },
        "augmentation": {
            "weak": {
                "random_horizontal_flip": True,
                "random_vertical_flip": False,
                "random_rotation": 0,
                "color_jitter": 0.0,
            },
        },
        "logging": {
            "save_freq": 1,
            "log_freq": 1,
        },
        "wandb": {
            "enabled": False,
        },
        "tensorboard": {
            "enabled": False,
            "log_dir": None,
            "flush_secs": 30,
        },
    }


@pytest.fixture
def synthetic_pil_images():
    """Create a batch of synthetic PIL images with labels."""
    images = []
    labels = []
    for i in range(8):
        # Create random-ish images (benign=darker, malignant=brighter)
        label = i % 2
        brightness = 100 + label * 100
        img = Image.new("RGB", (64, 64), (brightness, brightness, brightness))
        images.append(img)
        labels.append(label)
    return images, labels


@pytest.fixture
def synthetic_tensor_batch():
    """Create a synthetic batch of tensors for forward pass testing."""
    data = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 2, (4,))
    return data, targets


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset that returns PIL images or tensors."""

    def __init__(self, size=16, image_size=32, transform=None, return_pil=False):
        self.size = size
        self.image_size = image_size
        self.transform = transform
        self.return_pil = return_pil
        self.labels = [i % 2 for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.return_pil or self.transform:
            img = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return img, label
        else:
            img = torch.randn(3, self.image_size, self.image_size)
            return img, label

    def get_class_counts(self):
        return {
            "benign": sum(1 for l in self.labels if l == 0),
            "malignant": sum(1 for l in self.labels if l == 1),
        }

    def get_metadata_frame(self):
        import pandas as pd

        rows = []
        for idx, label in enumerate(self.labels):
            rows.append(
                {
                    "jpeg_path": f"synthetic_{idx}.png",
                    "label": label,
                    "patient_id": f"P_{idx // 2:03d}",
                    "exam_id": f"E_{idx // 2:03d}",
                    "laterality": "LEFT" if idx % 2 == 0 else "RIGHT",
                    "view": "CC" if idx % 2 == 0 else "MLO",
                    "abnormality_type": "mass",
                    "dataset_name": "synthetic",
                    "source_id": "synthetic",
                    "pathology": "MALIGNANT" if label == 1 else "BENIGN",
                }
            )
        return pd.DataFrame(rows)
