from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from physicalai.models.vla.openvla import _PROMPT_TEMPLATE, OpenVLAModel
from physicalai.utils.config import OpenVLAConfig

_PATCH_PROC = "physicalai.models.vla.openvla.AutoProcessor.from_pretrained"
_PATCH_MODEL = "physicalai.models.vla.openvla.AutoModelForVision2Seq.from_pretrained"


@pytest.fixture
def cpu_config():
    return OpenVLAConfig(device="cpu", dtype="float32", quantize=False)


def test_init_does_not_load_model(cpu_config):
    model = OpenVLAModel(cpu_config)
    assert model._model is None
    assert model._processor is None


def test_predict_raises_before_load(cpu_config, dummy_image):
    model = OpenVLAModel(cpu_config)
    with pytest.raises(RuntimeError, match="not loaded"):
        model.predict(dummy_image, "pick up the cup")


def test_load_returns_self(cpu_config):
    with patch(_PATCH_PROC) as mock_proc, patch(_PATCH_MODEL) as mock_model:
        mock_model.return_value = MagicMock()
        mock_model.return_value.to.return_value = mock_model.return_value
        mock_proc.return_value = MagicMock()

        model = OpenVLAModel(cpu_config)
        result = model.load()
        assert result is model


def test_load_calls_from_pretrained_with_correct_dtype(cpu_config):
    with patch(_PATCH_PROC) as mock_proc, patch(_PATCH_MODEL) as mock_model:
        mock_model.return_value = MagicMock()
        mock_model.return_value.to.return_value = mock_model.return_value
        mock_proc.return_value = MagicMock()

        OpenVLAModel(cpu_config).load()

        # cpu_config has dtype="float32" — verify the correct torch dtype is passed
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["torch_dtype"] == torch.float32
        mock_proc.assert_called_once_with(cpu_config.model_id, trust_remote_code=True)


def test_load_calls_from_pretrained_bfloat16():
    cfg = OpenVLAConfig(device="cpu", dtype="bfloat16", quantize=False)
    with patch(_PATCH_PROC), patch(_PATCH_MODEL) as mock_model:
        mock_model.return_value = MagicMock()
        mock_model.return_value.to.return_value = mock_model.return_value

        OpenVLAModel(cfg).load()

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["torch_dtype"] == torch.bfloat16


def test_predict_returns_ndarray(cpu_config, dummy_image):
    with patch(_PATCH_PROC) as mock_proc, patch(_PATCH_MODEL) as mock_model_cls:

        fake_action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        fake_model = MagicMock()
        fake_model.predict_action.return_value = fake_action
        fake_model.to.return_value = fake_model
        mock_model_cls.return_value = fake_model

        fake_processor = MagicMock()
        fake_inputs = {"input_ids": MagicMock(), "pixel_values": MagicMock()}
        fake_processor.return_value = MagicMock()
        fake_processor.return_value.to.return_value = fake_inputs
        mock_proc.return_value = fake_processor

        model = OpenVLAModel(cpu_config).load()
        action = model.predict(dummy_image, "pick up the cup")

        assert isinstance(action, np.ndarray)
        assert action.shape == (7,)


def test_prompt_template_contains_instruction():
    instruction = "pick up the red block"
    prompt = _PROMPT_TEMPLATE.format(instruction=instruction)
    assert instruction in prompt
    assert "In:" in prompt
    assert "Out:" in prompt
