from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from physicalai.utils.logging import get_logger

_log = get_logger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)
    _log.info("Checkpoint saved to %s", path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    _log.info("Checkpoint loaded from %s", path)
    return payload.get("metadata", {})
