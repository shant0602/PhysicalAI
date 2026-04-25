from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Root of the repo — so we can import scripts/train.py
REPO_ROOT = Path(__file__).parents[2]


def _load_train_module():
    """Import scripts/train.py as a module without executing main()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_script", REPO_ROOT / "scripts" / "train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_config(tmp_path: Path) -> Path:
    """Write a minimal CPU training config with 0 epochs."""
    cfg = {
        "model": {
            "model_id": "openvla/openvla-7b",
            "device": "cpu",
            "dtype": "float32",
            "quantize": False,
            "unnorm_key": "bridge_orig",
        },
        "lora": {
            "r": 4,
            "lora_alpha": 8,
            "target_modules": ["q_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
        },
        "data": {"split": "train", "max_samples": 5},
        "training": {
            "num_epochs": 0,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "output_dir": str(tmp_path / "checkpoints"),
        },
        "wandb": {"project": "physicalai-test", "run_name": "ci-smoke"},
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


# ── CLI argument parsing ───────────────────────────────────────────────────────

def test_cli_missing_config_exits():
    train = _load_train_module()
    with pytest.raises(SystemExit):
        sys.argv = ["train.py"]
        train.main()


def test_apply_overrides_dot_notation():
    train = _load_train_module()
    cfg = {"training": {"learning_rate": 1e-4}}
    result = train._apply_overrides(cfg, ["--training.learning_rate", "5e-5"])
    assert result["training"]["learning_rate"] == pytest.approx(5e-5)


def test_apply_overrides_bool_true():
    train = _load_train_module()
    result = train._apply_overrides({}, ["--model.quantize", "true"])
    assert result["model"]["quantize"] is True


def test_apply_overrides_bool_false():
    train = _load_train_module()
    result = train._apply_overrides({}, ["--model.quantize", "false"])
    assert result["model"]["quantize"] is False


def test_apply_overrides_int():
    train = _load_train_module()
    result = train._apply_overrides({}, ["--data.max_samples", "42"])
    assert result["data"]["max_samples"] == 42


def test_apply_overrides_nested_creates_keys():
    train = _load_train_module()
    result = train._apply_overrides({}, ["--a.b.c", "hello"])
    assert result["a"]["b"]["c"] == "hello"


# ── CLI smoke test: full path from main() → trainer instantiation ─────────────

def test_cli_smoke_cpu(tmp_path):
    """
    Runs main() end-to-end with a mocked trainer.
    Verifies config parsing, override wiring, and trainer instantiation
    all work without a GPU or real model download.
    """
    config_path = _minimal_config(tmp_path)
    train = _load_train_module()

    with patch("physicalai.training.trainer.OpenVLALoRATrainer") as MockTrainer:
        mock_instance = MagicMock()
        MockTrainer.return_value = mock_instance

        sys.argv = ["train.py", "--config", str(config_path)]
        train.main()

        # Trainer was instantiated once
        MockTrainer.assert_called_once()
        # train() was called on the instance
        mock_instance.train.assert_called_once()


def test_cli_override_applied(tmp_path):
    """CLI dot-notation override reaches the trainer config."""
    config_path = _minimal_config(tmp_path)
    train = _load_train_module()

    with patch("physicalai.training.trainer.OpenVLALoRATrainer") as MockTrainer:
        MockTrainer.return_value = MagicMock()

        sys.argv = [
            "train.py",
            "--config", str(config_path),
            "--data.max_samples", "3",
        ]
        train.main()

        # Inspect the data_config argument (3rd positional arg to __init__)
        _, args, _ = MockTrainer.mock_calls[0]
        data_cfg = args[2]
        assert data_cfg.max_samples == 3


# ── Trainer init ───────────────────────────────────────────────────────────────

def test_trainer_init_stores_configs():
    """OpenVLALoRATrainer stores all three configs without error."""
    from physicalai.training.data.bridge_dataset import BridgeDatasetConfig
    from physicalai.training.trainer import LoRATrainingConfig, OpenVLALoRATrainer
    from physicalai.utils.config import OpenVLAConfig

    model_cfg = OpenVLAConfig(device="cpu", dtype="float32", quantize=False)
    train_cfg = LoRATrainingConfig(num_epochs=0)
    data_cfg = BridgeDatasetConfig(max_samples=5)

    trainer = OpenVLALoRATrainer(model_cfg, train_cfg, data_cfg)

    assert trainer._mc is model_cfg
    assert trainer._tc is train_cfg
    assert trainer._dc is data_cfg
