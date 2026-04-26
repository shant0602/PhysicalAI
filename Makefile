SHELL := /bin/bash

.PHONY: install test lint format evaluate collect clean \
        submodule-init download-dataset download-libero \
        train train-libero train-dry-run deploy eval-libero \
        docker-build docker-build-dev docker-build-train \
        docker-run docker-dev docker-shell docker-train docker-train-libero docker-cpu docker-cpu-test \
        docker-clean

OPENVLA_DIR   = third_party/openvla
CONFIG_ENV    = configs/training/openvla_lora.env
NUM_GPUS     ?= 1
DATASET      ?= bridge_orig
WANDB_ENTITY ?=

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

docker-run:
	docker compose --profile gpu run --rm physicalai

docker-train:
	docker compose --profile gpu run --rm physicalai-train

docker-train-libero:
	set -a && source configs/training/libero_lora.env && set +a && \
	WANDB_ENTITY=$(WANDB_ENTITY) docker compose --profile gpu run --rm physicalai-train

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
