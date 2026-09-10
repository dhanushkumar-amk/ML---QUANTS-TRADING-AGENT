# ============================================================
# Smoke test — verifies that the project skeleton works
# ============================================================
"""Basic smoke tests for the project setup."""

import os
from pathlib import Path


# ---- Project root detection ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_structure_exists():
    """Essential directories are present."""
    expected_dirs = [
        "src",
        "src/data_pipeline",
        "src/features",
        "src/models",
        "src/backtest",
        "src/execution",
        "src/nlp",
        "src/utils",
        "data",
        "notebooks",
        "tests",
        "configs",
        "dashboard",
    ]
    for d in expected_dirs:
        assert (PROJECT_ROOT / d).is_dir(), f"Missing directory: {d}"


def test_config_loader_returns_dict():
    """Config loader can parse the default YAML file."""
    from src.utils.config_loader import load_config

    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    assert isinstance(cfg, dict)
    assert "data" in cfg
    assert "tickers" in cfg["data"]


def test_logger_creates_logger():
    """Logger factory returns a working logger with handlers."""
    from src.utils.logger import get_logger

    logger = get_logger("test")
    assert logger.name == "test"
    assert len(logger.handlers) > 0


def test_env_example_exists():
    """.env.example is present in the project root."""
    assert (PROJECT_ROOT / ".env.example").is_file()
