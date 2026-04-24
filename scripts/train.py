#!/usr/bin/env python3
"""
OpenVLA LoRA fine-tuning entry point.

Usage:
  python scripts/train.py --config configs/training/openvla_lora.yaml
  python scripts/train.py --config configs/training/openvla_lora.yaml --training.num_epochs 5
  python scripts/train.py --config configs/training/openvla_lora.yaml --data.max_samples 500
  python scripts/train.py --config configs/training/openvla_lora.yaml --model.device cpu --model.dtype float32
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import yaml

from physicalai.training.data.bridge_dataset import BridgeDatasetConfig
from physicalai.training.trainer import LoRATrainingConfig, OpenVLALoRATrainer
from physicalai.utils.config import OpenVLAConfig
from physicalai.utils.logging import get_logger

_log = get_logger(__name__)


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-notation CLI overrides to a nested config dict.

    Example: ["--training.learning_rate", "1e-4"] sets cfg["training"]["learning_rate"] = 1e-4
    """
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        if i + 1 >= len(overrides):
            _log.warning("Override key '%s' has no value — skipping", key)
            break
        raw_value = overrides[i + 1]
        i += 2

        # Parse value: try int, float, bool, then fall back to string
        value: Any
        if raw_value.lower() in ("true", "false"):
            value = raw_value.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    value = cast(raw_value)
                    break
                except ValueError:
                    pass
            else:
                value = raw_value

        # Navigate nested dict via dot-notation
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        _log.info("Override: %s = %s", key, value)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA with LoRA on BridgeData V2")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, remaining = parser.parse_known_args()

    with open(args.config) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    cfg = _apply_overrides(cfg, remaining)

    model_cfg = OpenVLAConfig(**cfg.get("model", {}))
    model_cfg.validate()

    train_cfg = LoRATrainingConfig(**cfg.get("training", {}))

    lora_keys = {"r", "lora_alpha", "target_modules", "lora_dropout"}
    lora_overrides = {k: v for k, v in cfg.get("lora", {}).items() if k in lora_keys}
    if lora_overrides:
        train_cfg = LoRATrainingConfig(**{**vars(train_cfg), **lora_overrides})

    data_cfg = BridgeDatasetConfig(**cfg.get("data", {}))

    wandb_cfg = cfg.get("wandb", {})
    train_cfg.wandb_project = wandb_cfg.get("project", train_cfg.wandb_project)
    train_cfg.wandb_run_name = wandb_cfg.get("run_name", train_cfg.wandb_run_name)

    _log.info("=== OpenVLA LoRA Training ===")
    _log.info("Model:    %s", model_cfg.model_id)
    _log.info("Device:   %s  dtype=%s", model_cfg.device, model_cfg.dtype)
    _log.info("Data:     split=%s  max_samples=%s", data_cfg.split, data_cfg.max_samples)
    _log.info("LoRA:     r=%d  alpha=%d  targets=%s", train_cfg.r, train_cfg.lora_alpha, train_cfg.target_modules)
    _log.info("Training: epochs=%d  batch=%d  lr=%s", train_cfg.num_epochs, train_cfg.batch_size, train_cfg.learning_rate)
    _log.info("Output:   %s", train_cfg.output_dir)

    trainer = OpenVLALoRATrainer(model_cfg, train_cfg, data_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
