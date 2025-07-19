# AmbientMIDI Makefile
# Comprehensive project management for development, testing, and deployment

.PHONY: help install install-dev test test-coverage test-quick test-unit test-integration lint format clean docs docs-serve build dist upload check-deps security-check profile benchmark setup-dev

# Default target
help:
	@echo "AmbientMIDI Project Management"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  install        Install package and dependencies"
	@echo "  install-dev    Install package in development mode with dev dependencies"
	@echo "  setup-dev      Complete development environment setup"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests with coverage"
	@echo "  test-coverage  Run tests with detailed coverage report"
	@echo "  test-quick     Run fast tests only"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests"
	@echo "  test-watch     Run tests in watch mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linting (flake8, pylint)"
	@echo "  format         Format code (black, isort)"
	@echo "  type-check     Run type checking (mypy)"
	@echo "  security-check Run security analysis"
	@echo "  check-deps     Check for dependency vulnerabilities"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           Build documentation"
	@echo "  docs-serve     Serve documentation locally"
	@echo "  docs-clean     Clean documentation build"
	@echo ""
	@echo "Distribution:"
	@echo "  build          Build source and wheel distributions"
	@echo "  dist           Create distribution packages"
	@echo "  upload         Upload package to PyPI"
	@echo "  upload-test    Upload package to TestPyPI"
	@echo ""
	@echo "Performance:"
	@echo "  profile        Run performance profiling"
	@echo "  benchmark      Run benchmarks"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Clean temporary files and caches"
	@echo "  clean-all      Deep clean everything"
	@echo "  requirements   Update requirements files"

# Python and pip commands
PYTHON := python3
PIP := pip3
PYTEST := python3 -m pytest
FLAKE8 := flake8
BLACK := black
ISORT := isort
MYPY := mypy
PYLINT := pylint
COVERAGE := coverage

# Project directories
SRC_DIR := ambientmidi
TEST_DIR := tests
DOCS_DIR := docs
DIST_DIR := dist
BUILD_DIR := build

# Test and coverage settings
TEST_ARGS := --verbose --tb=short
COVERAGE_ARGS := --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing --cov-report=xml
COVERAGE_MIN := 80

# Installation targets
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt

setup-dev: install-dev
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	@echo "Creating necessary directories..."
	mkdir -p htmlcov test-reports docs/_build
	@echo "Development environment setup complete!"

# Testing targets
test:
	$(PYTHON) run_tests.py --coverage

test-coverage:
	$(PYTEST) $(TEST_ARGS) $(COVERAGE_ARGS) $(TEST_DIR)
	@echo ""
	@echo "Coverage report saved to htmlcov/index.html"

test-quick:
	$(PYTHON) run_tests.py --filter "not slow" -v 1

test-unit:
	$(PYTEST) $(TEST_ARGS) $(TEST_DIR)/test_*.py

test-integration:
	$(PYTEST) $(TEST_ARGS) $(TEST_DIR)/integration/

test-xml:
	$(PYTHON) run_tests.py --xml

test-watch:
	$(PYTEST) -f $(TEST_ARGS) $(TEST_DIR)

# Code quality targets
lint:
	@echo "Running flake8..."
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR) --max-line-length=100 --ignore=E203,W503
	@echo "Running pylint..."
	$(PYLINT) $(SRC_DIR) --reports=no --score=no

format:
	@echo "Running black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR) --line-length=100
	@echo "Running isort..."
	$(ISORT) $(SRC_DIR) $(TEST_DIR) --profile=black --line-length=100

format-check:
	@echo "Checking black formatting..."
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR) --line-length=100
	@echo "Checking isort formatting..."
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR) --profile=black

type-check:
	@echo "Running mypy type checking..."
	$(MYPY) $(SRC_DIR) --ignore-missing-imports --strict-optional

security-check:
	@echo "Running bandit security check..."
	bandit -r $(SRC_DIR) -f json -o security-report.json
	@echo "Running safety check..."
	safety check --json --output safety-report.json

check-deps:
	@echo "Checking for dependency vulnerabilities..."
	pip-audit --desc --output=json --output-file=deps-audit.json

# Documentation targets
docs:
	@echo "Building documentation..."
	cd $(DOCS_DIR) && $(MAKE) html
	@echo "Documentation built in $(DOCS_DIR)/_build/html/"

docs-serve:
	@echo "Serving documentation at http://localhost:8000"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	cd $(DOCS_DIR) && $(MAKE) clean

docs-auto:
	sphinx-autobuild $(DOCS_DIR) $(DOCS_DIR)/_build/html

# Build and distribution targets
build: clean
	$(PYTHON) setup.py sdist bdist_wheel

dist: build
	@echo "Distribution packages created in $(DIST_DIR)/"
	ls -la $(DIST_DIR)/

upload-test: dist
	twine upload --repository testpypi $(DIST_DIR)/*

upload: dist
	twine upload $(DIST_DIR)/*

check-dist:
	twine check $(DIST_DIR)/*

# Performance targets
profile:
	@echo "Running performance profiling..."
	$(PYTHON) -m cProfile -o profile.stats -s cumulative example_usage.py
	@echo "Profile saved to profile.stats"

benchmark:
	@echo "Running benchmarks..."
	$(PYTHON) -m pytest benchmarks/ -v --benchmark-only

# Maintenance targets
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf test-reports
	rm -rf profile.stats

clean-all: clean
	@echo "Deep cleaning..."
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf docs/_build
	rm -f *.json  # Remove report files

requirements:
	@echo "Updating requirements files..."
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Development workflow targets
pre-commit: format lint type-check test-quick
	@echo "Pre-commit checks passed!"

ci: format-check lint type-check test-coverage security-check
	@echo "CI checks completed!"

release-check: clean lint type-check test-coverage docs build check-dist
	@echo "Release checks completed!"

# Docker targets (if Docker is used)
docker-build:
	docker build -t ambientmidi:latest .

docker-test:
	docker run --rm ambientmidi:latest python -m pytest

docker-run:
	docker run --rm -it ambientmidi:latest

# Database/Migration targets (if applicable)
migrate:
	@echo "No migrations needed for this project"

# Environment setup
env-create:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

env-activate:
	@echo "Run: source venv/bin/activate"

# Package info
info:
	@echo "AmbientMIDI Package Information"
	@echo "==============================="
	@$(PYTHON) -c "import ambientmidi; print(f'Version: {ambientmidi.__version__}')"
	@$(PYTHON) -c "import ambientmidi; import sys; print(f'Python: {sys.version}')"
	@echo "Location: $(shell pwd)"

# Check if running in virtual environment
check-venv:
	@$(PYTHON) -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" || (echo "Warning: Not running in virtual environment" && false)

# Dependency targets
deps-list:
	$(PIP) list

deps-outdated:
	$(PIP) list --outdated

deps-upgrade:
	$(PIP) list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 $(PIP) install -U

# Git workflow helpers
git-clean:
	git clean -fdx

git-reset:
	git reset --hard HEAD

# Performance monitoring
monitor:
	@echo "Monitoring system resources during test run..."
	$(PYTHON) -c "import psutil; import time; p = psutil.Process(); print(f'Memory before: {p.memory_info().rss / 1024 / 1024:.2f} MB')"
	$(MAKE) test-quick
	$(PYTHON) -c "import psutil; import time; p = psutil.Process(); print(f'Memory after: {p.memory_info().rss / 1024 / 1024:.2f} MB')"

# All-in-one targets
dev-setup: env-create install-dev setup-dev
	@echo "Complete development setup finished!"

full-test: clean lint type-check test-coverage docs
	@echo "Full test suite completed!"

release: release-check upload
	@echo "Package released!"