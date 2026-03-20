"""
Tests for the Mean Teacher training pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.sampling import sample_balanced_labeled_indices
from src.data.ssl_dataset import FixMatchLabeledDataset, TeacherStudentUnlabeledDataset
from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.mean_teacher_trainer import MeanTeacherTrainer
from tests.conftest import SyntheticDataset


class TestSampling:
    def test_balanced_sampling_is_seed_stable(self):
        labels = [i % 2 for i in range(20)]
        indices = list(range(20))

        sample_a = sample_balanced_labeled_indices(indices, labels, num_labeled=10, seed=42)
        sample_b = sample_balanced_labeled_indices(indices, labels, num_labeled=10, seed=42)
        sample_c = sample_balanced_labeled_indices(indices, labels, num_labeled=10, seed=43)

        assert sample_a == sample_b
        assert sample_a != sample_c
        assert sum(labels[i] == 0 for i in sample_a) == 5
        assert sum(labels[i] == 1 for i in sample_a) == 5


class TestTeacherStudentDataset:
    def test_teacher_student_views(self):
        raw = SyntheticDataset(size=8, image_size=32, return_pil=True)
        teacher_transform = get_transforms("weak", image_size=32)
        student_transform = get_transforms("mild_strong", image_size=32)
        dataset = TeacherStudentUnlabeledDataset(
            raw,
            teacher_transform=teacher_transform,
            student_transform=student_transform,
        )

        teacher_img, student_img = dataset[0]
        assert isinstance(teacher_img, torch.Tensor)
        assert isinstance(student_img, torch.Tensor)
        assert teacher_img.shape == (3, 32, 32)
        assert student_img.shape == (3, 32, 32)


class TestMeanTeacherTrainer:
    def _make_config(self, base_config):
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
        return base_config

    def test_consistency_weight_ramp(self, base_config):
        config = self._make_config(base_config)
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        trainer = MeanTeacherTrainer(model, config)

        assert trainer._get_current_consistency_weight(0) == pytest.approx(0.0)
        assert trainer._get_current_consistency_weight(20) == pytest.approx(1.0)

    def test_full_training_loop(self, base_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(base_config)
            config["training"]["num_epochs"] = 2
            model = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer = MeanTeacherTrainer(model, config, output_dir=tmpdir)

            weak_t = get_transforms("weak", image_size=32)
            student_t = get_transforms("mild_strong", image_size=32)

            labeled = FixMatchLabeledDataset(
                SyntheticDataset(size=8, image_size=32, return_pil=True),
                weak_transform=weak_t,
            )
            unlabeled = TeacherStudentUnlabeledDataset(
                SyntheticDataset(size=16, image_size=32, return_pil=True),
                teacher_transform=weak_t,
                student_transform=student_t,
            )
            val_dataset = SyntheticDataset(size=4, image_size=32)

            labeled_loader = DataLoader(labeled, batch_size=2, drop_last=True)
            unlabeled_loader = DataLoader(unlabeled, batch_size=4, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=2)

            trainer.train(labeled_loader, unlabeled_loader, val_loader)

            assert len(trainer.history["train_loss"]) == 2
            assert len(trainer.history["consistency_loss"]) == 2
            assert len(trainer.ema.shadow) > 0

    def test_checkpoint_roundtrip(self, base_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(base_config)
            model = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer = MeanTeacherTrainer(model, config, output_dir=tmpdir)

            trainer.current_consistency_weight = 0.5
            trainer.backbone_unfrozen = True
            trainer.best_val_auc = 0.81
            trainer.history["train_loss"].append(0.4)
            trainer.save_checkpoint(epoch=3, auc=0.81, best=True)

            model2 = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer2 = MeanTeacherTrainer(model2, config, output_dir=tmpdir)
            epoch = trainer2.load_checkpoint(Path(tmpdir) / "best_model.pth")

            assert epoch == 3
            assert trainer2.best_val_auc == 0.81
            assert trainer2.current_consistency_weight == pytest.approx(0.5)
            assert trainer2.backbone_unfrozen is True
            assert len(trainer2.ema.shadow) > 0

    def test_tensorboard_logs_consistency_scalars(self, base_config, tmp_path):
        config = self._make_config(base_config)
        config["tensorboard"]["enabled"] = True

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
            config["training"]["num_epochs"] = 1
            model = EfficientNetClassifier(num_classes=2, pretrained=False)
            trainer = MeanTeacherTrainer(model, config, output_dir=tmp_path)

            weak_t = get_transforms("weak", image_size=32)
            student_t = get_transforms("mild_strong", image_size=32)

            labeled = FixMatchLabeledDataset(
                SyntheticDataset(size=8, image_size=32, return_pil=True),
                weak_transform=weak_t,
            )
            unlabeled = TeacherStudentUnlabeledDataset(
                SyntheticDataset(size=16, image_size=32, return_pil=True),
                teacher_transform=weak_t,
                student_transform=student_t,
            )
            val_dataset = SyntheticDataset(size=4, image_size=32)

            labeled_loader = DataLoader(labeled, batch_size=2, drop_last=True)
            unlabeled_loader = DataLoader(unlabeled, batch_size=4, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=2)

            trainer.train(labeled_loader, unlabeled_loader, val_loader)

            scalar_tags = [args[0] for args, _ in trainer.tb_writer.scalars]
            assert "train/sup_loss" in scalar_tags
            assert "train/consistency_loss" in scalar_tags
            assert "train/consistency_weight" in scalar_tags
