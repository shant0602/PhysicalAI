from __future__ import annotations

import torch
import torch.nn as nn

from physicalai.utils.checkpoint import load_checkpoint, save_checkpoint


def _simple_model() -> nn.Module:
    return nn.Linear(4, 2)


def test_save_creates_file(tmp_path):
    model = _simple_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)
    assert path.exists()


def test_load_restores_weights(tmp_path):
    model = _simple_model()
    original_weight = model.weight.data.clone()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(model, path)

    model2 = _simple_model()
    model2.weight.data.fill_(0.0)  # corrupt
    load_checkpoint(path, model2)

    assert torch.allclose(model2.weight.data, original_weight)


def test_metadata_preserved(tmp_path):
    model = _simple_model()
    path = tmp_path / "ckpt.pt"
    meta = {"step": 42, "epoch": 2, "loss": 0.123}

    save_checkpoint(model, path, metadata=meta)
    returned_meta = load_checkpoint(path, model)

    assert returned_meta == meta


def test_save_creates_parent_dirs(tmp_path):
    model = _simple_model()
    deep_path = tmp_path / "a" / "b" / "c" / "ckpt.pt"
    save_checkpoint(model, deep_path)
    assert deep_path.exists()


def test_save_no_metadata(tmp_path):
    model = _simple_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path)
    meta = load_checkpoint(path, model)
    assert meta == {}
