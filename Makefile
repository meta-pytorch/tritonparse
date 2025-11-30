# Makefile for tritonparse project

.PHONY: help format format-check lint lint-check test test-cuda clean install-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  format        - Format all Python files"
	@echo "  format-check  - Check formatting without making changes"
	@echo "  lint          - Run all linters"
	@echo "  lint-check    - Check linting without making changes"
	@echo "  test          - Run tests (CPU only)"
	@echo "  test-cuda     - Run tests (including CUDA tests)"
	@echo "  clean         - Clean up cache files"
	@echo "  install-dev   - Install development dependencies"

# Formatting targets
format:
	@echo "Running format fix script..."
	python -m tritonparse.tools.format_fix --verbose

format-check:
	@echo "Checking formatting..."
	python -m tritonparse.tools.format_fix --check-only --verbose

# Linting targets
lint:
	@echo "Running linters..."
	ruff check .
	black --check .

lint-check:
	@echo "Checking linting..."
	ruff check --diff .
	black --check --diff .

# Testing targets
test:
	@echo "Running tests (CPU only)..."
	pytest tests/ -v -m "not cuda"

test-cuda:
	@echo "Running all tests (including CUDA)..."
	pytest tests/ -v

# Utility targets
clean:
	@echo "Cleaning up cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

install-dev:
	@echo "Installing development dependencies..."
	pip install -U black usort ruff coverage
