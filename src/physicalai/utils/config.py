from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import Literal

import yaml

_log = logging.getLogger(__name__)

__all__ = ["OpenVLAConfig", "GROOTConfig"]

# Devices that do not support bfloat16/float16 reliably
_NON_CUDA_DEVICES = ("cpu", "mps")


@dataclasses.dataclass
class OpenVLAConfig:
    model_id: str = "openvla/openvla-7b"
    device: str = "cuda:0"
    quantize: bool = False
    unnorm_key: str = "bridge_orig"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    @classmethod
    def from_yaml(cls, path: str) -> OpenVLAConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate(self) -> None:
        if self.quantize and not self.device.startswith("cuda"):
            raise ValueError(
                f"quantize=True requires a CUDA device. Got device='{self.device}'. "
                "Use --device cuda:0 or remove --quantize."
            )
        _is_non_cuda = not self.device.startswith("cuda")
        if _is_non_cuda and self.dtype != "float32":
            msg = (
                f"dtype='{self.dtype}' is not reliably supported on device='{self.device}'. "
                "Switching to float32. Pass --dtype float32 to silence this warning."
            )
            warnings.warn(msg, stacklevel=2)
            _log.warning(msg)
            self.dtype = "float32"


@dataclasses.dataclass
class GROOTConfig:
    model_id: str = "nvidia/GR00T-N1.5-3B"
    embodiment_tag: str = "gr1"
    data_config: str = "fourier_gr1_arms_only"
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    quantize: bool = False
    denoising_steps: int = 4

    @classmethod
    def from_yaml(cls, path: str) -> GROOTConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate(self) -> None:
        # GR00T's diffusion action head requires CUDA — CPU inference is not supported
        if not self.device.startswith("cuda"):
            raise ValueError(
                f"GROOTConfig: device must be 'cuda:N'. Got '{self.device}'. "
                "GR00T's diffusion action head requires a CUDA device."
            )
        if self.denoising_steps < 1:
            raise ValueError(f"denoising_steps must be >= 1, got {self.denoising_steps}")
