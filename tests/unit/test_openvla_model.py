from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from physicalai.models.vla.openvla import OpenVLAModel, _PROMPT_TEMPLATE
from physicalai.utils.config import OpenVLAConfig


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
    with patch("physicalai.models.vla.openvla.AutoProcessor.from_pretrained") as mock_proc, \
         patch("physicalai.models.vla.openvla.AutoModelForVision2Seq.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        mock_model.return_value.to.return_value = mock_model.return_value
        mock_proc.return_value = MagicMock()

        model = OpenVLAModel(cpu_config)
        result = model.load()
        assert result is model


def test_load_calls_from_pretrained_once(cpu_config):
    with patch("physicalai.models.vla.openvla.AutoProcessor.from_pretrained") as mock_proc, \
         patch("physicalai.models.vla.openvla.AutoModelForVision2Seq.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        mock_model.return_value.to.return_value = mock_model.return_value
        mock_proc.return_value = MagicMock()

        OpenVLAModel(cpu_config).load()
        mock_model.assert_called_once_with(cpu_config.model_id, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=pytest.approx)
        mock_proc.assert_called_once_with(cpu_config.model_id, trust_remote_code=True)


def test_predict_returns_ndarray(cpu_config, dummy_image):
    with patch("physicalai.models.vla.openvla.AutoProcessor.from_pretrained") as mock_proc, \
         patch("physicalai.models.vla.openvla.AutoModelForVision2Seq.from_pretrained") as mock_model_cls:

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
