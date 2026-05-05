from __future__ import annotations

import tempfile

import pytest
import yaml

from physicalai.utils.config import GROOTConfig


def test_defaults():
    cfg = GROOTConfig()
    assert cfg.model_id == "nvidia/GR00T-N1.5-3B"
    assert cfg.embodiment_tag == "gr1"
    assert cfg.data_config == "fourier_gr1_arms_only"
    assert cfg.device == "cuda:0"
    assert cfg.dtype == "bfloat16"
    assert cfg.quantize is False
    assert cfg.denoising_steps == 4


def test_from_yaml_round_trip():
    data = {
        "model_id": "nvidia/GR00T-N1.5-3B",
        "embodiment_tag": "gr1",
        "data_config": "fourier_gr1_arms_only",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "quantize": False,
        "denoising_steps": 8,
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(data, f)
        path = f.name

    cfg = GROOTConfig.from_yaml(path)
    assert cfg.model_id == "nvidia/GR00T-N1.5-3B"
    assert cfg.denoising_steps == 8
    assert cfg.embodiment_tag == "gr1"


def test_validate_accepts_cuda():
    cfg = GROOTConfig(device="cuda:0")
    cfg.validate()  # must not raise


def test_validate_rejects_cpu():
    cfg = GROOTConfig(device="cpu")
    with pytest.raises(ValueError, match="cuda"):
        cfg.validate()


def test_validate_rejects_mps():
    cfg = GROOTConfig(device="mps")
    with pytest.raises(ValueError, match="cuda"):
        cfg.validate()


def test_validate_rejects_zero_denoising_steps():
    cfg = GROOTConfig(device="cuda:0", denoising_steps=0)
    with pytest.raises(ValueError, match="denoising_steps"):
        cfg.validate()
