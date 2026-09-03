# Makefile for tritonparse project

.PHONY: help format format-check lint lint-check test test-cuda clean install-dev regen-examples regen-examples-install website-install website-lint website-build website-build-single website-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  format           - Format all Python files"
	@echo "  format-check     - Check formatting without making changes"
	@echo "  lint             - Fix Python lint issues"
	@echo "  lint-check       - Check Python lint issues without making changes"
	@echo "  test             - Run tests (CPU only)"
	@echo "  test-cuda        - Run tests (including CUDA tests)"
	@echo "  clean            - Clean up cache files"
	@echo "  install-dev      - Install development dependencies"
	@echo ""
	@echo "Example trace targets (require a CUDA GPU):"
	@echo "  regen-examples         - Regenerate example traces into ./example_output_regen"
	@echo "  regen-examples-install - Regenerate and overwrite the checked-in examples"
	@echo ""
	@echo "Website targets:"
	@echo "  website-install     - Install website dependencies"
	@echo "  website-lint        - Run ESLint on website"
	@echo "  website-build       - Build website"
	@echo "  website-build-single - Build standalone website"
	@echo "  website-dev         - Run website dev server"

# Formatting targets
format:
	@echo "Running format fix script..."
	python -m tritonparse.tools.format_fix --format-only --verbose

format-check:
	@echo "Checking formatting..."
	python -m tritonparse.tools.format_fix --format-only --check-only --verbose

lint:
	@echo "Fixing lint issues..."
	python -m tritonparse.tools.format_fix --lint-only --verbose

lint-check:
	@echo "Checking lint issues..."
	python -m tritonparse.tools.format_fix --lint-only --check-only --verbose

# Testing targets
test:
	@echo "Running tests (CPU only)..."
	python -m unittest discover -s tests/cpu -t . -v

test-cuda:
	@echo "Running all tests (including CUDA)..."
	python -m unittest discover -s tests -t . -v

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
	pip install -e ".[dev]"

# Example trace targets
# WORKLOAD selects which example to build; see --list for the options.
WORKLOAD ?= triton

regen-examples:
	@echo "Regenerating example trace '$(WORKLOAD)' (requires a CUDA GPU)..."
	python -m tritonparse.tools.generate_examples $(WORKLOAD)

regen-examples-install:
	@echo "Regenerating example trace '$(WORKLOAD)' and overwriting checked-in copies..."
	python -m tritonparse.tools.generate_examples $(WORKLOAD) --install

# Website targets
website-install:
	@echo "Installing website dependencies..."
	cd website && npm ci

website-lint:
	@echo "Running ESLint on website..."
	cd website && npm run lint

website-build:
	@echo "Building website..."
	cd website && npm run build

website-build-single:
	@echo "Building standalone website..."
	cd website && npm run build:single

website-dev:
	@echo "Starting website dev server..."
	cd website && npm run dev
