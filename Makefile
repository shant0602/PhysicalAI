.PHONY: install test lint format train evaluate collect clean \
        docker-build docker-build-dev docker-run docker-dev docker-shell docker-clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/
	mypy src/

format:
	ruff format src/ tests/ scripts/

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

collect:
	python scripts/collect_data.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile -t physicalai:latest .

docker-build-dev: docker-build
	docker build -f docker/Dockerfile.dev -t physicalai:dev .

docker-run:
	docker compose run --rm physicalai

docker-dev:
	docker compose up physicalai-dev

docker-shell:
	docker compose run --rm physicalai bash

docker-cpu:
	docker compose run --rm physicalai-cpu bash

docker-clean:
	docker compose down --rmi local --volumes
