from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from physicalai.utils.config import GROOTConfig
from physicalai.utils.logging import get_logger

# gr00t is available only when third_party/isaac_groot is installed.
# Importing at module level would break CI (no GPU env); defer to .load().
try:
    from gr00t.experiment.data_config import load_data_config  # type: ignore[import-untyped]
    from gr00t.model.policy import Gr00tPolicy  # type: ignore[import-untyped]

    _GROOT_AVAILABLE = True
except ImportError:
    _GROOT_AVAILABLE = False


class GROOTModel:
    """
    Thin wrapper around Isaac-GR00T's Gr00tPolicy.

    Exposes the same lazy-load / predict interface as OpenVLAModel so the two
    are interchangeable in evaluation harnesses.

    Action output: np.ndarray of shape (action_dim,) — the first action step
    from GR00T's predicted action chunk.  For gr1 (fourier_gr1_arms_only) this
    is typically 14-dimensional (7 joints × 2 arms).
    """

    def __init__(self, config: GROOTConfig) -> None:
        self._config = config
        self._log: logging.Logger = get_logger(__name__)
        self._policy: Gr00tPolicy | None = None  # type: ignore[type-arg]

    def load(self) -> GROOTModel:
        """Load model from checkpoint or HuggingFace. Returns self for chaining."""
        if not _GROOT_AVAILABLE:
            raise ImportError(
                "gr00t package not found. Install Isaac-GR00T:\n"
                "  pip install -e third_party/isaac_groot[base]"
            )

        config = self._config
        config.validate()

        self._log.info("Loading GR00T data config: %s", config.data_config)
        data_config = load_data_config(config.data_config)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        self._log.info("Loading GR00T policy from %s", config.model_id)
        self._policy = Gr00tPolicy(  # type: ignore[assignment]
            model_path=config.model_id,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=config.embodiment_tag,
            denoising_steps=config.denoising_steps,
            device=config.device,
        )
        self._log.info("GR00T policy loaded (denoising_steps=%d)", config.denoising_steps)
        return self

    def predict(
        self,
        image: Image.Image,
        instruction: str,
    ) -> np.ndarray:
        """
        Run inference on one observation.

        Args:
            image: PIL RGB image of the robot workspace.
            instruction: Natural language task (e.g. "sort the can").

        Returns:
            np.ndarray of shape (action_dim,) — first step of the predicted action chunk.
        """
        if self._policy is None:
            raise RuntimeError(
                f"Model '{self._config.model_id}' is not loaded. Call .load() first."
            )

        import torch  # lazy — keep module importable without torch in test env

        obs = {
            "video.ego_view": torch.from_numpy(
                np.array(image.convert("RGB"), dtype=np.uint8)
            ).unsqueeze(0).unsqueeze(0),  # (1, 1, H, W, 3)
            "annotation.human.task_description": [instruction],
            "state.single_arm": torch.zeros(1, 1, 7),   # placeholder; policy overwrites
            "state.gripper": torch.zeros(1, 1, 1),
        }

        with torch.no_grad():
            action_chunk = self._policy.get_action(obs)  # type: ignore[attr-defined]

        # action_chunk shape: (1, action_horizon, action_dim) — take step 0
        action: np.ndarray = np.asarray(action_chunk[0, 0])
        return action
