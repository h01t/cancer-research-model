"""
Tests for the supervised training pipeline.

All tests use synthetic data — no CBIS-DDSM dataset required.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.trainer import BaseTrainer, get_device
from tests.conftest import SyntheticDataset


class TestDeviceDetection:
    """Test device auto-detection logic."""

    def test_get_device_cpu(self):
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_auto(self):
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")

    def test_get_device_explicit(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")


class TestModelForward:
    """Test EfficientNet model architecture."""

    def test_forward_shape(self):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 2)

    def test_get_features_shape(self):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 32, 32)
        features = model.get_features(x)
        assert features.ndim == 2
        assert features.shape[0] == 2
        # EfficientNet-B0 feature dim is 1280
        assert features.shape[1] == 1280

    def test_freeze_backbone(self):
        model = EfficientNetClassifier(num_classes=2, pretrained=False, freeze_backbone=True)
        # Backbone params should be frozen
        for param in model.backbone.features.parameters():
            assert not param.requires_grad
        # Classifier params should be trainable
        for param in model.backbone.classifier.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self):
        model = EfficientNetClassifier(num_classes=2, pretrained=False, freeze_backbone=True)
        model.unfreeze_backbone()
        for param in model.backbone.features.parameters():
            assert param.requires_grad


class TestTrainerInit:
    """Test BaseTrainer initialization."""

    def test_trainer_creates(self, base_config):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = BaseTrainer(model, base_config)
        assert trainer.device.type in ("cpu", "cuda", "mps")

    def test_trainer_with_class_weights(self, base_config):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        weights = torch.tensor([1.0, 1.5])
        trainer = BaseTrainer(model, base_config, class_weights=weights)
        assert trainer.criterion.weight is not None

    def test_trainer_sgd_optimizer(self, base_config):
        base_config["training"]["optimizer"] = "sgd"
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = BaseTrainer(model, base_config)
        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_trainer_adamw_optimizer(self, base_config):
        base_config["training"]["optimizer"] = "adamw"
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = BaseTrainer(model, base_config)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)


class TestTrainingLoop:
    """Test the actual training loop with synthetic data."""

    def test_single_epoch(self, base_config):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = BaseTrainer(model, base_config)

        dataset = SyntheticDataset(size=8, image_size=32)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        loss, metrics = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_validation(self, base_config):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = BaseTrainer(model, base_config)

        dataset = SyntheticDataset(size=8, image_size=32)
        loader = DataLoader(dataset, batch_size=2)

        loss, metrics = trainer.validate(loader)
        assert isinstance(loss, float)
        assert "accuracy" in metrics

    def test_full_train_loop(self, base_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config["training"]["num_epochs"] = 2
            model = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer = BaseTrainer(model, base_config, output_dir=tmpdir)

            train_dataset = SyntheticDataset(size=8, image_size=32)
            val_dataset = SyntheticDataset(size=4, image_size=32)

            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=2)

            trainer.train(train_loader, val_loader)

            assert len(trainer.history["train_loss"]) == 2
            assert len(trainer.history["val_loss"]) == 2
            assert len(trainer.history["learning_rate"]) == 2

    def test_checkpoint_save_load(self, base_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer = BaseTrainer(model, base_config, output_dir=tmpdir)

            trainer.best_val_auc = 0.85
            trainer.save_checkpoint(1, 0.85, best=True)
            checkpoint_path = Path(tmpdir) / "best_model.pth"
            assert checkpoint_path.exists()

            # Load into a fresh trainer
            model2 = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer2 = BaseTrainer(model2, base_config, output_dir=tmpdir)
            epoch = trainer2.load_checkpoint(checkpoint_path)
            assert epoch == 1
            assert trainer2.best_val_auc == 0.85
