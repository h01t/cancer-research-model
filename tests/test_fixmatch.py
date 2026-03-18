"""
Tests for the FixMatch SSL training pipeline.

All tests use synthetic data — no CBIS-DDSM dataset required.
"""

import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.ssl_dataset import FixMatchLabeledDataset, FixMatchUnlabeledDataset
from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.ema import EMAModel
from src.training.fixmatch_trainer import FixMatchTrainer
from tests.conftest import SyntheticDataset


class TestSSLDatasets:
    """Test SSL dataset wrappers."""

    def test_fixmatch_labeled_dataset(self):
        raw = SyntheticDataset(size=8, image_size=32, return_pil=True)
        transform = get_transforms("weak", image_size=32)
        dataset = FixMatchLabeledDataset(raw, weak_transform=transform)

        assert len(dataset) == 8
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)
        assert label in (0, 1)

    def test_fixmatch_unlabeled_dataset(self):
        raw = SyntheticDataset(size=8, image_size=32, return_pil=True)
        weak_t = get_transforms("weak", image_size=32)
        strong_t = get_transforms("strong", image_size=32)
        dataset = FixMatchUnlabeledDataset(raw, weak_transform=weak_t, strong_transform=strong_t)

        assert len(dataset) == 8
        weak_img, strong_img = dataset[0]
        assert isinstance(weak_img, torch.Tensor)
        assert isinstance(strong_img, torch.Tensor)
        assert weak_img.shape == (3, 32, 32)
        assert strong_img.shape == (3, 32, 32)


class TestEMAModel:
    """Test EMA weight averaging."""

    def test_ema_init(self):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        ema = EMAModel(model, decay=0.999)
        assert len(ema.shadow) > 0

    def test_ema_update(self):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        ema = EMAModel(model, decay=0.999)

        # Get a param name for comparison
        name = next(iter(ema.shadow.keys()))
        original_shadow = ema.shadow[name].clone()

        # Run a fake step to change model weights (batch_size>=2 for BatchNorm)
        x = torch.randn(2, 3, 32, 32)
        loss = model(x).sum()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ema.update(model)

        # Shadow should have changed (slightly)
        assert not torch.equal(ema.shadow[name], original_shadow)

    def test_ema_apply_restore(self):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        ema = EMAModel(model, decay=0.5)

        # Change model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model)

        # Save original weights
        name = next(iter(ema.shadow.keys()))
        original_param = dict(model.named_parameters())[name].data.clone()

        # Apply EMA
        ema.apply(model)
        ema_param = dict(model.named_parameters())[name].data.clone()
        assert not torch.equal(original_param, ema_param)

        # Restore
        ema.restore(model)
        restored_param = dict(model.named_parameters())[name].data.clone()
        assert torch.equal(original_param, restored_param)


class TestFixMatchTrainer:
    """Test FixMatch trainer."""

    def test_trainer_init(self, base_config):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = FixMatchTrainer(model, base_config)
        assert trainer.confidence_threshold == 0.95
        assert trainer.lambda_u == 1.0

    def test_trainer_init_with_ema(self, base_config):
        base_config["ssl"]["use_ema"] = True
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = FixMatchTrainer(model, base_config)
        assert trainer.ema is not None

    def test_single_epoch(self, base_config):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = FixMatchTrainer(model, base_config)

        # Create labeled dataset
        raw = SyntheticDataset(size=8, image_size=32, return_pil=True)
        weak_t = get_transforms("weak", image_size=32)
        strong_t = get_transforms("strong", image_size=32)

        labeled = FixMatchLabeledDataset(raw, weak_transform=weak_t)
        unlabeled = FixMatchUnlabeledDataset(
            SyntheticDataset(size=16, image_size=32, return_pil=True),
            weak_transform=weak_t,
            strong_transform=strong_t,
        )

        labeled_loader = DataLoader(labeled, batch_size=2, shuffle=True, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled, batch_size=4, shuffle=True, drop_last=True)

        loss, metrics = trainer.train_epoch_ssl(labeled_loader, unlabeled_loader)
        assert isinstance(loss, float)
        assert "accuracy" in metrics
        assert "sup_loss" in metrics
        assert "unsup_loss" in metrics
        assert "mask_ratio" in metrics

    def test_full_training_loop(self, base_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config["training"]["num_epochs"] = 2
            model = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer = FixMatchTrainer(model, base_config, output_dir=tmpdir)

            weak_t = get_transforms("weak", image_size=32)
            strong_t = get_transforms("strong", image_size=32)

            labeled = FixMatchLabeledDataset(
                SyntheticDataset(size=8, image_size=32, return_pil=True),
                weak_transform=weak_t,
            )
            unlabeled = FixMatchUnlabeledDataset(
                SyntheticDataset(size=16, image_size=32, return_pil=True),
                weak_transform=weak_t,
                strong_transform=strong_t,
            )
            val_dataset = SyntheticDataset(size=4, image_size=32)

            labeled_loader = DataLoader(labeled, batch_size=2, drop_last=True)
            unlabeled_loader = DataLoader(unlabeled, batch_size=4, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=2)

            trainer.train(labeled_loader, unlabeled_loader, val_loader)

            assert len(trainer.history["train_loss"]) == 2
            assert len(trainer.history["sup_loss"]) == 2
            assert len(trainer.history["mask_ratio"]) == 2
