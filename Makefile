.PHONY: help install dev lint format type test smoke docker clean

PY ?= python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

install: ## install the package
	$(PY) -m pip install .

dev: ## install with dev extras
	$(PY) -m pip install -e ".[dev]"

lint: ## run ruff and isort checks
	ruff check .
	isort --check-only .

format: ## apply black and isort
	black .
	isort .

type: ## run mypy strict on the package
	mypy

test: ## run the test suite
	pytest

smoke: ## run two training steps on the smoke config
	$(PY) -m damascus.anvil weld --config configs/experiment/_smoke.yaml

docker: ## build the container image
	docker build -t damascus:latest .

clean: ## remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .mypy_cache .ruff_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
