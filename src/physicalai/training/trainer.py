from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import torch
import wandb
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, get_cosine_schedule_with_warmup

from physicalai.training.data.bridge_dataset import BridgeDatasetConfig, BridgeV2Dataset
from physicalai.utils.config import OpenVLAConfig
from physicalai.utils.logging import get_logger

_log = get_logger(__name__)


@dataclasses.dataclass
class LoRATrainingConfig:
    # LoRA adapter settings
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = dataclasses.field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"

    # Paths
    output_dir: str = "checkpoints/openvla_lora"

    # Training hyperparams
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    save_every_n_steps: int = 200
    log_every_n_steps: int = 10
    seed: int = 42

    # W&B
    wandb_project: str = "physicalai-openvla"
    wandb_run_name: Optional[str] = None


class OpenVLALoRATrainer:
    def __init__(
        self,
        model_config: OpenVLAConfig,
        train_config: LoRATrainingConfig,
        data_config: BridgeDatasetConfig,
    ) -> None:
        self._mc = model_config
        self._tc = train_config
        self._dc = data_config

    def _save_adapter(self, model: torch.nn.Module, path: Path, metadata: dict) -> None:
        """Save only the LoRA adapter weights (~28MB), not the full base model."""
        model.save_pretrained(path)
        # Also write metadata alongside the adapter
        import json
        (path / "training_metadata.json").write_text(json.dumps(metadata))
        _log.info("LoRA adapter saved to %s", path)

    def train(self) -> None:
        torch.manual_seed(self._tc.seed)

        wandb.init(
            project=self._tc.wandb_project,
            name=self._tc.wandb_run_name,
            config=dataclasses.asdict(self._tc),
        )

        # --- Load processor and base model ---
        _log.info("Loading processor from %s", self._mc.model_id)
        processor = AutoProcessor.from_pretrained(self._mc.model_id, trust_remote_code=True)

        _log.info("Loading model %s on %s (dtype=bfloat16)", self._mc.model_id, self._mc.device)
        model = AutoModelForVision2Seq.from_pretrained(
            self._mc.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self._mc.device)

        # Freeze the entire base model before applying LoRA
        model.requires_grad_(False)

        # --- Apply LoRA adapters ---
        lora_cfg = LoraConfig(
            r=self._tc.r,
            lora_alpha=self._tc.lora_alpha,
            target_modules=self._tc.target_modules,
            lora_dropout=self._tc.lora_dropout,
            bias=self._tc.bias,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # --- Dataset and DataLoader ---
        _log.info("Loading BridgeData V2 (split=%s, max_samples=%s)", self._dc.split, self._dc.max_samples)
        dataset = BridgeV2Dataset(processor=processor, config=self._dc)
        loader = DataLoader(
            dataset,
            batch_size=self._tc.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        _log.info("Dataset size: %d samples, %d batches/epoch", len(dataset), len(loader))

        # --- Optimizer and scheduler ---
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self._tc.learning_rate,
            weight_decay=self._tc.weight_decay,
        )

        total_steps = (len(loader) // self._tc.gradient_accumulation_steps) * self._tc.num_epochs
        warmup_steps = int(total_steps * self._tc.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        _log.info("Training: total_steps=%d, warmup_steps=%d", total_steps, warmup_steps)

        # --- Training loop ---
        output_dir = Path(self._tc.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        global_step = 0
        model.train()

        for epoch in range(self._tc.num_epochs):
            _log.info("Epoch %d / %d", epoch + 1, self._tc.num_epochs)
            optimizer.zero_grad()

            for step, batch in enumerate(loader):
                batch = {k: v.to(self._mc.device) for k, v in batch.items()}
                # Pop action labels — not used for LM loss; model is trained
                # on next-token prediction over the action token sequence.
                batch.pop("labels")

                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss / self._tc.gradient_accumulation_steps
                loss.backward()

                is_accumulation_step = (step + 1) % self._tc.gradient_accumulation_steps == 0
                is_last_step = step == len(loader) - 1

                if is_accumulation_step or is_last_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self._tc.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self._tc.log_every_n_steps == 0:
                        lr = scheduler.get_last_lr()[0]
                        _log.info("step=%d  loss=%.4f  lr=%.2e", global_step, loss.item(), lr)
                        wandb.log(
                            {"train/loss": loss.item(), "train/lr": lr, "train/epoch": epoch},
                            step=global_step,
                        )

                    if global_step % self._tc.save_every_n_steps == 0:
                        # Save only the LoRA adapter weights (~28MB), not the full base model
                        ckpt_dir = output_dir / f"step_{global_step}"
                        self._save_adapter(
                            model, ckpt_dir,
                            metadata={"step": global_step, "epoch": epoch, "loss": loss.item()},
                        )

        # --- Final save ---
        final_dir = output_dir / "final"
        self._save_adapter(
            model, final_dir,
            metadata={"step": global_step, "epoch": self._tc.num_epochs},
        )

        wandb.finish()
        _log.info("Training complete. Artifacts at %s", output_dir)
