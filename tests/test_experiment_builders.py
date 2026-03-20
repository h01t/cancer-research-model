"""
Tests for shared experiment builders.
"""

from PIL import Image

from src.experiments import (
    build_fixmatch_experiment,
    build_mean_teacher_experiment,
    build_supervised_experiment,
)


class MockCBISDataset:
    def __init__(self, size=20, image_size=32):
        self.size = size
        self.image_size = image_size
        self.labels = [i % 2 for i in range(size)]
        self.patient_ids = [f"P_{i // 2:03d}" for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        return img, self.labels[idx]

    def get_class_counts(self):
        return {
            "benign": sum(label == 0 for label in self.labels),
            "malignant": sum(label == 1 for label in self.labels),
        }


def _patch_builders(monkeypatch, builders_module):
    monkeypatch.setattr(builders_module, "_build_raw_train_dataset", lambda config: MockCBISDataset())
    monkeypatch.setattr(
        builders_module,
        "_build_test_dataset",
        lambda config, test_transform: MockCBISDataset(size=8),
    )


class TestExperimentBuilders:
    def test_supervised_builder_is_seed_stable(self, base_config, monkeypatch):
        from src.experiments import builders as builders_module

        _patch_builders(monkeypatch, builders_module)
        device = __import__("torch").device("cpu")

        bundle_a = build_supervised_experiment(base_config, device, seed=42, labeled_subset_size=10)
        bundle_b = build_supervised_experiment(base_config, device, seed=42, labeled_subset_size=10)

        assert len(bundle_a.datasets.train_dataset) == len(bundle_b.datasets.train_dataset) == 10
        assert len(bundle_a.datasets.val_dataset) == len(bundle_b.datasets.val_dataset)

    def test_fixmatch_builder_returns_ssl_loaders(self, base_config, monkeypatch):
        from src.experiments import builders as builders_module

        _patch_builders(monkeypatch, builders_module)
        device = __import__("torch").device("cpu")

        bundle = build_fixmatch_experiment(base_config, device, seed=42, num_labeled=10)

        assert bundle.loaders.train_loader is None
        assert bundle.loaders.labeled_loader is not None
        assert bundle.loaders.unlabeled_loader is not None
        assert len(bundle.datasets.labeled_dataset) == 10

    def test_mean_teacher_builder_uses_teacher_student_views(self, base_config, monkeypatch):
        from src.experiments import builders as builders_module

        _patch_builders(monkeypatch, builders_module)
        device = __import__("torch").device("cpu")
        base_config["ssl"] = {
            "method": "mean_teacher",
            "unlabeled_batch_ratio": 2,
            "ema_decay": 0.999,
            "consistency_loss": "mse",
            "consistency_weight_start": 0.0,
            "consistency_weight_end": 1.0,
            "consistency_weight_ramp_epochs": 20,
            "teacher_augmentation": "weak",
            "student_augmentation": "mild_strong",
        }
        base_config["augmentation"]["mild_strong"] = {
            "random_horizontal_flip": True,
            "random_rotation": 8.0,
            "translate": [0.05, 0.05],
            "brightness": 0.08,
            "contrast": 0.12,
        }

        bundle = build_mean_teacher_experiment(base_config, device, seed=42, num_labeled=10)

        assert bundle.loaders.labeled_loader is not None
        assert bundle.loaders.unlabeled_loader is not None
        teacher_batch, student_batch = next(iter(bundle.loaders.unlabeled_loader))
        assert teacher_batch.shape[0] == student_batch.shape[0]
