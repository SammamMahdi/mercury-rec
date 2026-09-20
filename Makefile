# MercuryRec - convenience wrapper around the `mercury` CLI.
#
# The CLI is the canonical interface: `make` is not available on Windows,
# which is the primary development platform here, so every target below is a
# thin alias for a command that also works without make. Anything documented
# in the README is runnable either way.
#
#   make help      list targets

SHELL := /bin/bash
.DEFAULT_GOAL := help

# `uv run` resolves the project venv without needing it activated, and avoids
# the broken `py` launcher on the development machine entirely.
UV  := uv run
PY  := $(UV) python

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- environment -----------------------------------------------------------

.PHONY: install
install:  ## Create the venv and install all dependencies
	uv sync --extra dev --extra viz --extra perf

.PHONY: check-env
check-env:  ## Verify git identity, native libraries, thread configuration
	$(PY) scripts/check_env.py

.PHONY: check-gpu
check-gpu:  ## Verify the CUDA stack launches kernels and returns correct results
	$(PY) scripts/check_gpu.py

# --- quality ---------------------------------------------------------------

.PHONY: format
format:  ## Auto-format the codebase
	$(UV) ruff format .
	$(UV) ruff check --fix .

.PHONY: lint
lint:  ## Lint and format-check without modifying files
	$(UV) ruff format --check .
	$(UV) ruff check .

.PHONY: typecheck
typecheck:  ## Strict type check
	$(UV) mypy src/

.PHONY: test
test:  ## Run tests that need no external services
	$(UV) pytest -m "not integration and not gpu"

.PHONY: test-all
test-all:  ## Run every test, including those needing PostgreSQL and Redis
	$(UV) pytest

.PHONY: coverage
coverage:  ## Run tests with a coverage report
	$(UV) pytest -m "not integration and not gpu" --cov --cov-report=term-missing --cov-report=html

.PHONY: qa
qa: lint typecheck test  ## Full local quality gate

# --- data ------------------------------------------------------------------

.PHONY: download
download:  ## Fetch the Retailrocket dataset from Kaggle
	$(UV) mercury data download

# --- housekeeping ----------------------------------------------------------

.PHONY: clean
clean:  ## Remove caches and build artifacts (leaves data/ and artifacts/ alone)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
