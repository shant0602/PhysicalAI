from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Root of the repo — so we can import scripts/train.py
REPO_ROOT = Path(__file__).parents[2]


def _load_train_module():
    """Import scripts/train.py as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_script", REPO_ROOT / "scripts" / "train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_config(tmp_path: Path) -> Path:
    """Write a minimal training config that targets CPU with 0 epochs."""
    cfg = {
        "model": {
            "model_id": "openvla/openvla-7b",
            "device": "cpu",
            "dtype": "float32",
            "quantize": False,
            "unnorm_key": "bridge_orig",
        },
        "lora": {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "lora_dropout": 0.0},
        "data": {"split": "train", "max_samples": 5},
        "training": {
            "num_epochs": 0,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "output_dir": str(tmp_path / "checkpoints"),
        },
        "wandb": {"project": "physicalai-test", "run_name": "ci-smoke"},
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


def test_cli_missing_config_exits(capsys):
    train = _load_train_module()
    with pytest.raises(SystemExit):
        train.main.__wrapped__ if hasattr(train.main, "__wrapped__") else None
        sys.argv = ["train.py"]
        train.main()


def test_apply_overrides_dot_notation():
    train = _load_train_module()
    cfg = {"training": {"learning_rate": 1e-4}}
    result = train._apply_overrides(cfg, ["--training.learning_rate", "5e-5"])
    assert result["training"]["learning_rate"] == pytest.approx(5e-5)


def test_apply_overrides_bool_true():
    train = _load_train_module()
    cfg = {}
    result = train._apply_overrides(cfg, ["--model.quantize", "true"])
    assert result["model"]["quantize"] is True


def test_apply_overrides_bool_false():
    train = _load_train_module()
    cfg = {}
    result = train._apply_overrides(cfg, ["--model.quantize", "false"])
    assert result["model"]["quantize"] is False


def test_apply_overrides_int():
    train = _load_train_module()
    cfg = {}
    result = train._apply_overrides(cfg, ["--data.max_samples", "42"])
    assert result["data"]["max_samples"] == 42


def test_apply_overrides_nested_creates_keys():
    train = _load_train_module()
    cfg = {}
    result = train._apply_overrides(cfg, ["--a.b.c", "hello"])
    assert result["a"]["b"]["c"] == "hello"
