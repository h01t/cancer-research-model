"""
Tests for shared experiment runtime helpers.
"""

from pathlib import Path
from unittest.mock import patch

import yaml
from torch.utils.data import DataLoader

from src.experiments.runtime import (
    build_experiment_context,
    compute_class_weights,
    create_model,
    create_trainer,
    evaluate_and_persist_results,
)
from tests.conftest import SyntheticDataset


class TestExperimentRuntime:
    def test_context_writes_config_snapshot(self, base_config, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(base_config))

        context = build_experiment_context(
            config_path=str(config_path),
            output_dir=str(tmp_path / "run"),
            seed=123,
            overrides={"experiment": {"method": "supervised"}},
        )

        saved_config = yaml.safe_load((context.output_dir / "config.yaml").read_text())
        assert context.seed == 123
        assert saved_config["experiment"]["seed"] == 123
        assert saved_config["experiment"]["method"] == "supervised"

    def test_compute_class_weights(self):
        dataset = SyntheticDataset(size=10, image_size=32)
        weights = compute_class_weights(dataset)
        assert weights.shape[0] == 2

    def test_evaluate_and_persist_results(self, base_config, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(base_config))
        context = build_experiment_context(
            config_path=str(config_path),
            output_dir=str(tmp_path / "run"),
            seed=42,
            overrides={"experiment": {"method": "supervised"}},
        )

        model = create_model(context.config)
        trainer = create_trainer("supervised", model, context, class_weights=None)

        train_loader = DataLoader(SyntheticDataset(size=8, image_size=32), batch_size=2)
        val_loader = DataLoader(SyntheticDataset(size=4, image_size=32), batch_size=2)
        trainer.train(train_loader, val_loader)

        threshold, val_metrics, test_metrics = evaluate_and_persist_results(
            trainer,
            context.output_dir,
            val_loader,
            val_loader,
        )

        assert 0.0 <= threshold <= 1.0
        assert "auc" in val_metrics
        assert "decision_threshold" in test_metrics
        assert (context.output_dir / "val_metrics.yaml").exists()
        assert (context.output_dir / "test_metrics.yaml").exists()
        assert (context.output_dir / "training_history.csv").exists()

    def test_evaluate_and_persist_results_logs_tensorboard_curves(self, base_config, tmp_path):
        base_config["tensorboard"]["enabled"] = True
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(base_config))

        class DummyWriter:
            instances = []

            def __init__(self, log_dir, flush_secs):
                self.scalars = []
                self.curves = []
                DummyWriter.instances.append(self)

            def add_scalar(self, *args, **kwargs):
                self.scalars.append((args, kwargs))

            def add_pr_curve(self, *args, **kwargs):
                self.curves.append((args, kwargs))

            def close(self):
                pass

        with patch("src.training.trainer._get_summary_writer_cls", return_value=DummyWriter):
            context = build_experiment_context(
                config_path=str(config_path),
                output_dir=str(tmp_path / "run"),
                seed=42,
                overrides={"experiment": {"method": "supervised"}},
            )

            model = create_model(context.config)
            trainer = create_trainer("supervised", model, context, class_weights=None)

            train_loader = DataLoader(SyntheticDataset(size=8, image_size=32), batch_size=2)
            val_loader = DataLoader(SyntheticDataset(size=4, image_size=32), batch_size=2)
            trainer.train(train_loader, val_loader)

            threshold, val_metrics, test_metrics = evaluate_and_persist_results(
                trainer,
                context.output_dir,
                val_loader,
                val_loader,
            )

            assert 0.0 <= threshold <= 1.0
            assert "auc" in val_metrics
            assert "decision_threshold" in test_metrics
            assert trainer.tb_writer is None
            writer = DummyWriter.instances[-1]
            assert len(writer.curves) >= 2
            scalar_tags = [args[0] for args, _ in writer.scalars]
            assert "val/decision_threshold" in scalar_tags
            assert "test/decision_threshold" in scalar_tags
