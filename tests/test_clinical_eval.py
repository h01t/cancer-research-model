"""Tests for clinical evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clinical_eval import build_evaluation_context


def test_build_evaluation_context_disables_pretrained(tmp_path):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config = {
        "experiment": {"method": "supervised"},
        "model": {
            "name": "efficientnet-b0",
            "num_classes": 2,
            "pretrained": True,
            "dropout_rate": 0.2,
        },
    }
    config_path.write_text(yaml.safe_dump(config))

    context = build_evaluation_context(
        config_path=config_path,
        output_dir=output_dir,
        seed=42,
        device_override="cpu",
    )

    assert context.config["model"]["pretrained"] is False
    assert context.config["training"]["num_workers"] == 0
    saved = yaml.safe_load((output_dir / "config.yaml").read_text())
    assert saved["model"]["pretrained"] is False
    assert saved["training"]["num_workers"] == 0
