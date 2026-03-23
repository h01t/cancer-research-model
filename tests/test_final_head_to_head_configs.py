"""Tests for the final head-to-head config pair."""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_config(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_final_head_to_head_configs_match_family_except_optimizer():
    adam = _load_config("default_nofreeze_ls_adam.yaml")
    adamw = _load_config("default_nofreeze_ls_adamw.yaml")

    assert adam["dataset"]["image_size"] == 512
    assert adamw["dataset"]["image_size"] == 512
    assert adam["training"]["label_smoothing"] == 0.1
    assert adamw["training"]["label_smoothing"] == 0.1
    assert adam["model"]["name"] == "efficientnet-b0"
    assert adamw["model"]["name"] == "efficientnet-b0"

    assert adam["training"]["optimizer"] == "adam"
    assert adamw["training"]["optimizer"] == "adamw"
    assert adam["training"]["learning_rate"] == 0.001
    assert adamw["training"]["learning_rate"] == 0.0003
    assert adam["training"]["weight_decay"] == 0.0001
    assert adamw["training"]["weight_decay"] == 0.0005
