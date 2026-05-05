from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physicalai.inference.policy import OpenVLAPolicy


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.zeros(7, dtype=np.float32)
    return model


@pytest.fixture
def policy(mock_model):
    return OpenVLAPolicy(model=mock_model)


@pytest.fixture
def obs(dummy_image):
    return {"image": dummy_image, "instruction": "pick up the red cube"}


def test_step_returns_dict(policy, obs):
    result = policy.step(obs)
    assert isinstance(result, dict)


def test_step_returns_action_key(policy, obs):
    result = policy.step(obs)
    assert "action" in result


def test_step_returns_action_axes_key(policy, obs):
    result = policy.step(obs)
    assert "action_axes" in result


def test_step_action_shape(policy, obs):
    result = policy.step(obs)
    assert result["action"].shape == (7,)


def test_step_action_axes_labels(policy, obs):
    result = policy.step(obs)
    assert result["action_axes"] == ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def test_step_passes_image_and_instruction(policy, mock_model, obs):
    policy.step(obs)
    mock_model.predict.assert_called_once_with(
        obs["image"], obs["instruction"], unnorm_key=None
    )


def test_call_delegates_to_step(policy, obs):
    result_step = policy.step(obs)
    # Reset mock call count
    policy._model.predict.reset_mock()
    policy._model.predict.return_value = np.zeros(7, dtype=np.float32)
    result_call = policy(obs)
    assert list(result_step.keys()) == list(result_call.keys())


def test_unnorm_key_passed_through(mock_model, obs):
    policy = OpenVLAPolicy(model=mock_model, unnorm_key="my_dataset")
    policy.step(obs)
    mock_model.predict.assert_called_once_with(
        obs["image"], obs["instruction"], unnorm_key="my_dataset"
    )
