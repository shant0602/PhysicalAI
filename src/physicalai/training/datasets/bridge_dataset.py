from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset


@dataclasses.dataclass
class BridgeDatasetConfig:
    dataset_name: str = "jxu124/OpenX-Embodiment"
    dataset_config: str = "bridge"
    split: str = "train"
    max_samples: Optional[int] = None
    image_size: int = 224


class BridgeV2Dataset(Dataset):
    def __init__(self, processor, config: BridgeDatasetConfig) -> None:
        self.processor = processor
        self.config = config

        ds = load_dataset(config.dataset_name, config.dataset_config, split=config.split)
        if config.max_samples is not None:
            ds = ds.select(range(min(config.max_samples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        row = self._data[idx]

        img_array = row["observation"]["image_primary"][0]
        image = Image.fromarray(img_array).resize(
            (self.config.image_size, self.config.image_size)
        )

        encoding = self.processor(
            text=row["language_instruction"],
            images=image,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(row["action"][0], dtype=torch.float32)
        return item
