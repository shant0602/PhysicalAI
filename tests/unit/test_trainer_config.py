from __future__ import annotations

from physicalai.training.trainer import LoRATrainingConfig


def test_lora_training_config_defaults():
    cfg = LoRATrainingConfig()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.num_epochs == 3
    assert cfg.batch_size == 4
    assert cfg.gradient_accumulation_steps == 4
    assert cfg.learning_rate == 2e-4
    assert cfg.warmup_ratio == 0.03
    assert cfg.max_grad_norm == 1.0
    assert cfg.save_every_n_steps == 200
    assert cfg.log_every_n_steps == 10
    assert cfg.seed == 42


def test_target_modules_is_list():
    cfg = LoRATrainingConfig()
    assert isinstance(cfg.target_modules, list)
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules


def test_target_modules_not_shared():
    # dataclasses.field(default_factory=...) ensures no shared mutable default
    cfg1 = LoRATrainingConfig()
    cfg2 = LoRATrainingConfig()
    cfg1.target_modules.append("k_proj")
    assert "k_proj" not in cfg2.target_modules


def test_output_dir_is_str():
    cfg = LoRATrainingConfig()
    assert isinstance(cfg.output_dir, str)


def test_wandb_run_name_defaults_none():
    cfg = LoRATrainingConfig()
    assert cfg.wandb_run_name is None
