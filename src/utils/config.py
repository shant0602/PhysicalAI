from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import Literal

import yaml


@dataclasses.dataclass
class OpenVLAConfig:
    model_id: str = "openvla/openvla-7b"
    device: str = "cuda:0"
    quantize: bool = False
    unnorm_key: str = "bridge_orig"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    @classmethod
    def from_yaml(cls, path: str) -> "OpenVLAConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate(self) -> None:
        if self.quantize and not self.device.startswith("cuda"):
            raise ValueError(
                f"quantize=True requires a CUDA device. Got device='{self.device}'. "
                "Use --device cuda:0 or remove --quantize."
            )
        if self.device == "cpu" and self.dtype != "float32":
            warnings.warn(
                f"dtype='{self.dtype}' is not supported on CPU. "
                "Switching to float32. Pass --dtype float32 to silence this warning.",
                stacklevel=2,
            )
            self.dtype = "float32"
