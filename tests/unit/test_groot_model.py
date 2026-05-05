from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.utils.config import GROOTConfig

# ---------------------------------------------------------------------------
# Patch gr00t before importing GROOTModel so the import-time try/except
# sees a fake module and sets _GROOT_AVAILABLE = True.
# ---------------------------------------------------------------------------

_fake_gr00t = ModuleType("gr00t")
_fake_experiment = ModuleType("gr00t.experiment")
_fake_data_config_mod = ModuleType("gr00t.experiment.data_config")
_fake_model = ModuleType("gr00t.model")
_fake_policy_mod = ModuleType("gr00t.model.policy")

_fake_gr00t.experiment = _fake_experiment
_fake_gr00t.model = _fake_model
_fake_experiment.data_config = _fake_data_config_mod
_fake_model.policy = _fake_policy_mod

sys.modules.setdefault("gr00t", _fake_gr00t)
sys.modules.setdefault("gr00t.experiment", _fake_experiment)
sys.modules.setdefault("gr00t.experiment.data_config", _fake_data_config_mod)
sys.modules.setdefault("gr00t.model", _fake_model)
sys.modules.setdefault("gr00t.model.policy", _fake_policy_mod)

_fake_data_config_mod.load_data_config = MagicMock()
_fake_policy_mod.Gr00tPolicy = MagicMock()

from physicalai.models.vla.groot import GROOTModel  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PATCH_POLICY = "physicalai.models.vla.groot.Gr00tPolicy"
_PATCH_LOAD_CFG = "physicalai.models.vla.groot.load_data_config"


@pytest.fixture
def cuda_config():
    return GROOTConfig(device="cuda:0", denoising_steps=4)


@pytest.fixture
def dummy_image(dummy_image):  # reuses conftest fixture
    return dummy_image


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_does_not_load_model(cuda_config):
    model = GROOTModel(cuda_config)
    assert model._policy is None


def test_predict_raises_before_load(cuda_config, dummy_image):
    model = GROOTModel(cuda_config)
    with pytest.raises(RuntimeError, match="not loaded"):
        model.predict(dummy_image, "sort the can")


def test_load_returns_self(cuda_config):
    with patch(_PATCH_LOAD_CFG) as mock_cfg, patch(_PATCH_POLICY) as mock_policy_cls:
        mock_data_cfg = MagicMock()
        mock_cfg.return_value = mock_data_cfg
        mock_policy_cls.return_value = MagicMock()

        model = GROOTModel(cuda_config)
        result = model.load()
        assert result is model


def test_load_passes_correct_config(cuda_config):
    with patch(_PATCH_LOAD_CFG) as mock_cfg, patch(_PATCH_POLICY) as mock_policy_cls:
        mock_data_cfg = MagicMock()
        mock_cfg.return_value = mock_data_cfg

        GROOTModel(cuda_config).load()

        mock_cfg.assert_called_once_with(cuda_config.data_config)
        call_kwargs = mock_policy_cls.call_args.kwargs
        assert call_kwargs["embodiment_tag"] == cuda_config.embodiment_tag
        assert call_kwargs["denoising_steps"] == cuda_config.denoising_steps
        assert call_kwargs["device"] == cuda_config.device


def test_predict_returns_ndarray(cuda_config, dummy_image):
    with patch(_PATCH_LOAD_CFG), patch(_PATCH_POLICY) as mock_policy_cls:
        fake_action_chunk = np.zeros((1, 16, 14), dtype=np.float32)
        fake_policy = MagicMock()
        fake_policy.get_action.return_value = fake_action_chunk
        mock_policy_cls.return_value = fake_policy

        model = GROOTModel(cuda_config).load()
        action = model.predict(dummy_image, "sort the can")

        assert isinstance(action, np.ndarray)
        assert action.shape == (14,)


def test_predict_calls_get_action(cuda_config, dummy_image):
    with patch(_PATCH_LOAD_CFG), patch(_PATCH_POLICY) as mock_policy_cls:
        fake_action_chunk = np.zeros((1, 16, 14), dtype=np.float32)
        fake_policy = MagicMock()
        fake_policy.get_action.return_value = fake_action_chunk
        mock_policy_cls.return_value = fake_policy

        model = GROOTModel(cuda_config).load()
        model.predict(dummy_image, "sort the can")

        fake_policy.get_action.assert_called_once()
