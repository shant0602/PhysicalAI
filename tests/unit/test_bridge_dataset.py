from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from physicalai.training.datasets.bridge_dataset import BridgeDatasetConfig, BridgeV2Dataset


def _make_fake_hf_dataset(n: int = 5) -> MagicMock:
    """Build a minimal fake HuggingFace dataset with Bridge V2 structure."""
    rows = []
    for i in range(n):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        rows.append(
            {
                "observation": {"image_primary": [img]},
                "language_instruction": f"pick up object {i}",
                "action": [np.zeros(7, dtype=np.float32).tolist()],
            }
        )

    ds = MagicMock()
    ds.__len__.return_value = n
    ds.__getitem__ = lambda self, idx: rows[idx]
    ds.select.return_value = ds
    return ds


def _make_mock_processor() -> MagicMock:
    processor = MagicMock()
    processor.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return processor


@pytest.fixture
def fake_dataset():
    with patch("physicalai.training.datasets.bridge_dataset.load_dataset") as mock_load:
        mock_load.return_value = _make_fake_hf_dataset(5)
        cfg = BridgeDatasetConfig(max_samples=5)
        processor = _make_mock_processor()
        ds = BridgeV2Dataset(processor=processor, config=cfg)
        yield ds


def test_dataset_len(fake_dataset):
    assert len(fake_dataset) == 5


def test_getitem_has_labels(fake_dataset):
    item = fake_dataset[0]
    assert "labels" in item


def test_labels_shape_and_dtype(fake_dataset):
    item = fake_dataset[0]
    labels = item["labels"]
    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (7,)
    assert labels.dtype == torch.float32


def test_getitem_has_input_ids(fake_dataset):
    item = fake_dataset[0]
    assert "input_ids" in item


def test_bridge_dataset_config_defaults():
    cfg = BridgeDatasetConfig()
    assert cfg.split == "train"
    assert cfg.max_samples is None
    assert cfg.image_size == 224
