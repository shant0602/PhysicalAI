from __future__ import annotations

import warnings

import pytest
import yaml

from physicalai.utils.config import OpenVLAConfig


def test_default_config_values():
    cfg = OpenVLAConfig()
    assert cfg.model_id == "openvla/openvla-7b"
    assert cfg.device == "cuda:0"
    assert cfg.dtype == "bfloat16"
    assert cfg.quantize is False
    assert cfg.unnorm_key == "bridge_orig"


def test_validate_passes_on_cuda(monkeypatch):
    # Patch torch.cuda.is_available so validate doesn't raise for device strings
    cfg = OpenVLAConfig(device="cuda:0", dtype="bfloat16", quantize=False)
    cfg.validate()  # should not raise


def test_validate_cpu_downcasts_dtype():
    cfg = OpenVLAConfig(device="cpu", dtype="bfloat16")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg.validate()
    assert cfg.dtype == "float32"
    assert any("float32" in str(warning.message) for warning in w)


def test_validate_quantize_requires_cuda():
    cfg = OpenVLAConfig(device="cpu", quantize=True)
    with pytest.raises(ValueError, match="CUDA"):
        cfg.validate()


def test_from_yaml_roundtrip(tmp_path):
    data = {
        "model_id": "openvla/openvla-7b",
        "device": "cpu",
        "dtype": "float32",
        "quantize": False,
        "unnorm_key": "bridge_orig",
    }
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml.dump(data))

    cfg = OpenVLAConfig.from_yaml(str(yaml_file))
    assert cfg.model_id == data["model_id"]
    assert cfg.device == data["device"]
    assert cfg.dtype == data["dtype"]
    assert cfg.quantize == data["quantize"]
    assert cfg.unnorm_key == data["unnorm_key"]
