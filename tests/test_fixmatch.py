"""
Tests for the FixMatch SSL training pipeline.

All tests use synthetic data — no CBIS-DDSM dataset required.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

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

    def test_ema_refresh_after_unfreeze(self):
        """EMA.refresh() should pick up newly-unfrozen parameters."""
        model = EfficientNetClassifier(num_classes=2, pretrained=False, freeze_backbone=True)

        # EMA only tracks classifier (requires_grad=True) params initially
        ema = EMAModel(model, decay=0.999)
        initial_count = len(ema.shadow)

        # All backbone params are frozen → not in shadow
        total_params = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        assert initial_count == total_params

        # Unfreeze backbone
        model.unfreeze_backbone()
        total_after_unfreeze = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        assert total_after_unfreeze > initial_count

        # Refresh EMA — should add the newly unfrozen params
        ema.refresh(model)
        assert len(ema.shadow) == total_after_unfreeze

        # Verify EMA update works with the new params
        ema.update(model)


class TestFixMatchTrainer:
    """Test FixMatch trainer."""

    def test_trainer_init(self, base_config):
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = FixMatchTrainer(model, base_config)
        assert trainer.confidence_threshold == 0.95
        assert trainer.lambda_u == 1.0

    def test_schedule_ramps(self, base_config):
        base_config["ssl"]["confidence_threshold_start"] = 0.7
        base_config["ssl"]["confidence_threshold_end"] = 0.95
        base_config["ssl"]["confidence_threshold_ramp_epochs"] = 10
        base_config["ssl"]["lambda_u_start"] = 0.0
        base_config["ssl"]["lambda_u_end"] = 1.0
        base_config["ssl"]["lambda_u_ramp_epochs"] = 20

        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = FixMatchTrainer(model, base_config)

        assert trainer._get_current_confidence_threshold(0) == pytest.approx(0.7)
        assert trainer._get_current_confidence_threshold(10) == pytest.approx(0.95)
        assert trainer._get_current_lambda_u(0) == pytest.approx(0.0)
        assert trainer._get_current_lambda_u(20) == pytest.approx(1.0)

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

    def test_tensorboard_logs_ssl_scalars(self, base_config, tmp_path):
        base_config["tensorboard"]["enabled"] = True

        class DummyWriter:
            def __init__(self, log_dir, flush_secs):
                self.scalars = []

            def add_scalar(self, *args, **kwargs):
                self.scalars.append((args, kwargs))

            def add_pr_curve(self, *args, **kwargs):
                pass

            def close(self):
                pass

        with patch("src.training.trainer._get_summary_writer_cls", return_value=DummyWriter):
            base_config["training"]["num_epochs"] = 1
            model = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer = FixMatchTrainer(model, base_config, output_dir=tmp_path)

            raw = SyntheticDataset(size=8, image_size=32, return_pil=True)
            weak_t = get_transforms("weak", image_size=32)
            strong_t = get_transforms("strong", image_size=32)

            labeled = FixMatchLabeledDataset(raw, weak_transform=weak_t)
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

            scalar_tags = [args[0] for args, _ in trainer.tb_writer.scalars]
            assert "train/sup_loss" in scalar_tags
            assert "train/unsup_loss" in scalar_tags
            assert "train/mask_ratio" in scalar_tags
            assert "train/lambda_u" in scalar_tags
            assert "train/confidence_threshold" in scalar_tags

    def test_checkpoint_roundtrip(self, base_config):
        """FixMatch checkpoint should preserve all SSL state for proper resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config["training"]["num_epochs"] = 2
            base_config["ssl"]["use_ema"] = True
            model = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer = FixMatchTrainer(model, base_config, output_dir=tmpdir)

            # Simulate some training state
            trainer.current_lambda_u = 0.42
            trainer.current_confidence_threshold = 0.78
            trainer.backbone_unfrozen = True
            trainer.best_val_auc = 0.75
            trainer.epochs_without_improvement = 3
            trainer.pseudo_label_distribution = torch.tensor([0.6, 0.4])
            trainer.history["train_loss"].append(0.5)

            # Save checkpoint
            trainer.save_checkpoint(epoch=5, auc=0.75, best=True)

            # Create a fresh trainer and load
            model2 = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer2 = FixMatchTrainer(model2, base_config, output_dir=tmpdir)

            checkpoint_path = Path(tmpdir) / "best_model.pth"
            epoch = trainer2.load_checkpoint(checkpoint_path)

            assert epoch == 5
            assert trainer2.best_val_auc == 0.75
            assert trainer2.epochs_without_improvement == 3
            assert trainer2.current_lambda_u == pytest.approx(0.42)
            assert trainer2.current_confidence_threshold == pytest.approx(0.78)
            assert trainer2.backbone_unfrozen is True
            assert torch.allclose(
                trainer2.pseudo_label_distribution.cpu(), torch.tensor([0.6, 0.4])
            )
            # EMA shadow should be restored
            assert trainer2.ema is not None
            assert len(trainer2.ema.shadow) > 0
            # History should be restored
            assert len(trainer2.history["train_loss"]) == 1
