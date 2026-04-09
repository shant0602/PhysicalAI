from __future__ import annotations

import numpy as np
from PIL import Image

from physicalai.models.vla.openvla import OpenVLAModel

ACTION_AXES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


class OpenVLAPolicy:
    """
    Stateless policy wrapper around OpenVLAModel.

    Observation dict keys:
        "image":       PIL.Image.Image (RGB)
        "instruction": str

    Action dict keys:
        "action":       np.ndarray shape (7,) — delta end-effector pose + gripper
        "action_axes":  list[str]             — axis labels for display / logging
    """

    def __init__(self, model: OpenVLAModel, unnorm_key: str | None = None) -> None:
        self._model = model
        self._unnorm_key = unnorm_key

    def step(self, obs: dict) -> dict:
        image: Image.Image = obs["image"]
        instruction: str = obs["instruction"]
        action = self._model.predict(image, instruction, unnorm_key=self._unnorm_key)
        return {"action": action, "action_axes": ACTION_AXES}

    def __call__(self, obs: dict) -> dict:
        return self.step(obs)
