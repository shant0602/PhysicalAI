from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from physicalai.utils.config import OpenVLAConfig


@pytest.fixture
def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB")


@pytest.fixture
def base_model_config() -> OpenVLAConfig:
    return OpenVLAConfig(device="cpu", dtype="float32", quantize=False)


@pytest.fixture
def mock_hf_model() -> MagicMock:
    model = MagicMock()
    model.predict_action.return_value = np.zeros(7, dtype=np.float32)
    return model


@pytest.fixture
def mock_processor() -> MagicMock:
    processor = MagicMock()
    processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
        "attention_mask": MagicMock(),
    }
    return processor
