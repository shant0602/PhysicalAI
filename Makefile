SHELL := /bin/bash

.PHONY: install test lint format evaluate collect clean \
        submodule-init download-dataset download-libero \
        train train-libero train-dry-run deploy eval-libero \
        download-groot train-groot train-groot-dry-run eval-groot-offline \
        docker-build docker-build-dev docker-build-train docker-build-groot \
        docker-run docker-dev docker-shell docker-train docker-train-libero \
        docker-train-groot docker-cpu docker-cpu-test \
        docker-clean

OPENVLA_DIR   = third_party/openvla
GROOT_DIR     = third_party/isaac_groot
CONFIG_ENV    = configs/training/openvla_lora.env
GROOT_ENV     = configs/training/groot_lora.env
NUM_GPUS     ?= 1
DATASET      ?= bridge_orig
WANDB_ENTITY ?=
CHECKPOINT   ?=

# ── Local dev ─────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/ --exclude third_party/
	mypy src/ --exclude third_party/

format:
	ruff format src/ tests/ scripts/

evaluate:
	python scripts/evaluate.py

collect:
	python scripts/collect_data.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/

# ── Submodule ─────────────────────────────────────────────────────────────────
submodule-init:
	git submodule update --init --recursive

# ── Dataset download ──────────────────────────────────────────────────────────
download-dataset:
	python scripts/download_dataset.py $(DATASET)

download-libero:
	python scripts/download_dataset.py libero

# ── Training (calls OpenVLA's finetune.py via scripts/train.sh) ───────────────
train:
	source $(CONFIG_ENV) && DATASET_NAME=$(DATASET) WANDB_ENTITY=$(WANDB_ENTITY) NUM_GPUS=$(NUM_GPUS) bash scripts/train.sh

train-libero:
	source configs/training/libero_lora.env && WANDB_ENTITY=$(WANDB_ENTITY) NUM_GPUS=$(NUM_GPUS) bash scripts/train.sh

train-dry-run:
	bash scripts/train.sh --help

# ── Evaluation ────────────────────────────────────────────────────────────────
# Usage: make eval-libero CHECKPOINT=runs/<your-run-dir>
ifndef CHECKPOINT
eval-libero:
	$(error CHECKPOINT is required: make eval-libero CHECKPOINT=runs/<your-run-dir>)
else
eval-libero:
	python third_party/openvla/experiments/robot/libero/run_libero_eval.py \
		--model_family openvla \
		--pretrained_checkpoint $(CHECKPOINT) \
		--task_suite_name libero_spatial \
		--center_crop True
endif

# ── GR00T N1.5 ───────────────────────────────────────────────────────────────
download-groot:
	huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
		--repo-type dataset \
		--include "gr1_arms_only.CanSort/**" \
		--local-dir ~/gr00t_dataset

train-groot:
	set -a && source $(GROOT_ENV) && set +a && \
	WANDB_ENTITY=$(WANDB_ENTITY) NUM_GPUS=$(NUM_GPUS) bash scripts/train_groot.sh $(ARGS)

train-groot-dry-run:
	bash scripts/train_groot.sh --help

ifndef CHECKPOINT
eval-groot-offline:
	$(error CHECKPOINT is required: make eval-groot-offline CHECKPOINT=./checkpoints/gr00t_lora_cansort)
else
eval-groot-offline:
	bash scripts/eval_groot_offline.sh --model_path $(CHECKPOINT)
endif

# ── Deployment (calls OpenVLA's deploy.py via scripts/deploy.sh) ──────────────
deploy:
	bash scripts/deploy.sh --openvla_path openvla/openvla-7b

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile -t physicalai:latest .

docker-build-dev: docker-build
	docker build -f docker/Dockerfile.dev -t physicalai:dev .

docker-build-train: submodule-init
	docker build -f docker/Dockerfile.train -t physicalai:train .

docker-build-groot: submodule-init
	docker build -f docker/Dockerfile.train.groot -t physicalai:train-groot .

docker-run:
	docker compose --profile gpu run --rm physicalai

docker-train:
	docker compose --profile gpu run --rm physicalai-train

docker-train-libero:
	set -a && source configs/training/libero_lora.env && set +a && \
	WANDB_ENTITY=$(WANDB_ENTITY) docker compose --profile gpu run --rm physicalai-train

docker-train-groot:
	set -a && source $(GROOT_ENV) && set +a && \
	WANDB_ENTITY=$(WANDB_ENTITY) docker compose --profile gpu run --rm physicalai-train-groot

docker-dev:
	docker compose --profile gpu up physicalai-dev

docker-shell:
	docker compose --profile gpu run --rm --entrypoint bash physicalai

docker-cpu:
	docker compose run --rm --entrypoint bash physicalai-cpu

docker-cpu-test:
	docker compose run --rm physicalai-cpu pytest tests/unit/ -v

docker-clean:
	docker compose down --rmi local --volumes
