#!/bin/bash

# Development script for detr_vision project using uv

set -e

case "$1" in
    "install")
        echo "Installing dependencies..."
        uv sync
        ;;
    "install-dev")
        echo "Installing development dependencies..."
        uv sync --extra dev
        ;;
    "test")
        echo "Running tests..."
        uv run pytest
        ;;
    "test-cov")
        echo "Running tests with coverage..."
        uv run pytest --cov=src --cov-report=html
        ;;
    "format")
        echo "Formatting code..."
        uv run black src/ tests/
        uv run isort src/ tests/
        ;;
    "lint")
        echo "Linting code..."
        uv run flake8 src/ tests/
        uv run mypy src/
        ;;
    "check")
        echo "Running all checks..."
        uv run black --check src/ tests/
        uv run isort --check-only src/ tests/
        uv run flake8 src/ tests/
        uv run mypy src/
        uv run pytest
        ;;
    "add")
        if [ -z "$2" ]; then
            echo "Usage: $0 add <package_name>"
            exit 1
        fi
        echo "Adding package: $2"
        uv add "$2"
        ;;
    "add-dev")
        if [ -z "$2" ]; then
            echo "Usage: $0 add-dev <package_name>"
            exit 1
        fi
        echo "Adding development package: $2"
        uv add --dev "$2"
        ;;
    "remove")
        if [ -z "$2" ]; then
            echo "Usage: $0 remove <package_name>"
            exit 1
        fi
        echo "Removing package: $2"
        uv remove "$2"
        ;;
    "run")
        shift
        uv run "$@"
        ;;
    *)
        echo "Usage: $0 {install|install-dev|test|test-cov|format|lint|check|add|add-dev|remove|run}"
        echo ""
        echo "Commands:"
        echo "  install      - Install all dependencies"
        echo "  install-dev  - Install development dependencies"
        echo "  test         - Run tests"
        echo "  test-cov     - Run tests with coverage"
        echo "  format       - Format code with black and isort"
        echo "  lint         - Lint code with flake8 and mypy"
        echo "  check        - Run all checks (format, lint, test)"
        echo "  add          - Add a new package"
        echo "  add-dev      - Add a new development package"
        echo "  remove       - Remove a package"
        echo "  run          - Run a command in the virtual environment"
        exit 1
        ;;
esac 