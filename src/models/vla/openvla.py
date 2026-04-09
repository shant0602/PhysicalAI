from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from physicalai.utils.config import OpenVLAConfig
from physicalai.utils.logging import get_logger

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


class OpenVLAModel:
    def __init__(self, config: OpenVLAConfig) -> None:
        self._config = config
        self._log = get_logger(__name__)
        self._processor: AutoProcessor | None = None
        self._model: AutoModelForVision2Seq | None = None

    def load(self) -> "OpenVLAModel":
        """Load processor and model from HuggingFace. Returns self for chaining."""
        config = self._config

        self._log.info("Loading processor from %s", config.model_id)
        self._processor = AutoProcessor.from_pretrained(
            config.model_id, trust_remote_code=True
        )

        dtype = _DTYPE_MAP[config.dtype]
        load_kwargs: dict = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if config.quantize:
            self._log.info("Loading model with 4-bit quantization")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ImportError(
                    "bitsandbytes is required for quantize=True. "
                    "Install with: pip install 'physicalai[inference]'"
                ) from e
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            self._log.info(
                "Loading model with dtype=%s on device=%s", config.dtype, config.device
            )
            load_kwargs["torch_dtype"] = dtype

        self._model = AutoModelForVision2Seq.from_pretrained(config.model_id, **load_kwargs)

        if not config.quantize:
            self._model = self._model.to(config.device)

        # Quiet down HuggingFace verbosity after load
        logging.getLogger("transformers").setLevel(logging.WARNING)
        self._log.info("Model loaded successfully")
        return self

    def predict(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key: str | None = None,
    ) -> np.ndarray:
        """
        Run inference on one observation.

        Args:
            image: PIL RGB image of the robot workspace.
            instruction: Natural language task (e.g. "pick up the red cup").
            unnorm_key: Dataset key for action unnormalization. Falls back to config.

        Returns:
            np.ndarray of shape (7,) — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        key = unnorm_key or self._config.unnorm_key
        prompt = _PROMPT_TEMPLATE.format(instruction=instruction)
        dtype = _DTYPE_MAP[self._config.dtype]

        inputs = self._processor(prompt, image).to(self._config.device, dtype=dtype)
        action: np.ndarray = self._model.predict_action(
            **inputs, unnorm_key=key, do_sample=False
        )
        return action
