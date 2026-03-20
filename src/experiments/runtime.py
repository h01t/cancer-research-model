"""Shared experiment runtime helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch
import yaml

from src.models.efficientnet import EfficientNetClassifier
from src.training import BaseTrainer, FixMatchTrainer, MeanTeacherTrainer, get_device
from src.training.ema import EMAModel
from src.training.utils import set_seed

logger = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    config: dict
    device: torch.device
    output_dir: Path
    seed: int


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset) -> torch.Tensor:
    counts = dataset.get_class_counts()
    total = counts["benign"] + counts["malignant"]
    w_benign = total / (2.0 * max(counts["benign"], 1))
    w_malignant = total / (2.0 * max(counts["malignant"], 1))
    return torch.tensor([w_benign, w_malignant], dtype=torch.float32)


def build_experiment_context(
    config_path: str,
    output_dir: str,
    seed: int,
    device_override: str | None = None,
    overrides: dict | None = None,
) -> ExperimentContext:
    config = load_config(config_path)
    if overrides:
        for section, values in overrides.items():
            config.setdefault(section, {}).update(values)
    config.setdefault("experiment", {})["seed"] = seed
    set_seed(seed)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    with open(resolved_output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    device = get_device(device_override)
    return ExperimentContext(
        config=config,
        device=device,
        output_dir=resolved_output_dir,
        seed=seed,
    )


def create_model(config: dict) -> EfficientNetClassifier:
    model_name = config["model"]["name"].lower()
    if model_name != "efficientnet-b0":
        raise ValueError(f"Unsupported model: {model_name}")
    return EfficientNetClassifier(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout_rate"],
        freeze_backbone=config["model"].get("freeze_backbone", False),
    )


def create_trainer(
    method: str,
    model,
    context: ExperimentContext,
    class_weights: torch.Tensor | None,
):
    if method == "supervised":
        trainer_cls = BaseTrainer
    elif method == "fixmatch":
        trainer_cls = FixMatchTrainer
    elif method == "mean_teacher":
        trainer_cls = MeanTeacherTrainer
    else:
        raise ValueError(f"Unsupported training method: {method}")

    return trainer_cls(
        model,
        context.config,
        device=context.device,
        output_dir=context.output_dir,
        class_weights=class_weights,
    )


@contextmanager
def applied_ema(ema: EMAModel | None, model) -> Iterator[None]:
    if ema is not None:
        ema.apply(model)
    try:
        yield
    finally:
        if ema is not None:
            ema.restore(model)


def reload_best_checkpoint(trainer, output_dir: Path) -> None:
    checkpoint = output_dir / "best_model.pth"
    if checkpoint.exists():
        trainer.load_checkpoint(checkpoint)


def evaluate_and_persist_results(
    trainer,
    output_dir: Path,
    val_loader,
    test_loader,
    ema: EMAModel | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    reload_best_checkpoint(trainer, output_dir)
    with applied_ema(ema, trainer.model):
        val_loss, val_y_true, val_y_prob = trainer.predict(val_loader)
        threshold, val_metrics = trainer.tune_decision_threshold(val_loader)
        test_loss, test_y_true, test_y_prob = trainer.predict(test_loader)
        test_metrics = trainer.evaluate(test_loader, decision_threshold=threshold)

        val_metrics["loss"] = val_loss
        test_metrics["loss"] = test_loss
        trainer._log_eval_curves("val", val_y_true, val_y_prob, threshold, val_metrics)
        trainer._log_eval_curves("test", test_y_true, test_y_prob, threshold, test_metrics)

    with open(output_dir / "val_metrics.yaml", "w") as f:
        yaml.dump({k: float(v) for k, v in val_metrics.items()}, f)
    with open(output_dir / "test_metrics.yaml", "w") as f:
        yaml.dump({k: float(v) for k, v in test_metrics.items()}, f)

    pd.DataFrame(trainer.history).to_csv(output_dir / "training_history.csv", index=False)
    trainer._close_loggers()
    return threshold, val_metrics, test_metrics
