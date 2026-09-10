# ============================================================
# Unit Tests — Configuration Loader (Phase 1)
# ============================================================

from pathlib import Path

import pytest

from src.utils.config_loader import get_env, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_valid():
    """Verify loading default.yaml returns expected dictionary structure."""
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    assert isinstance(cfg, dict)
    assert "project" in cfg
    assert "data" in cfg
    assert "streaming" in cfg
    assert cfg["project"]["name"] == "ML + Quants Trading Agent"
    assert isinstance(cfg["data"]["tickers"], list)
    assert "AAPL" in cfg["data"]["tickers"]


def test_load_config_relative_path():
    """Verify loading via relative path works relative to project root."""
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg, dict)
    assert "features" in cfg


def test_load_config_file_not_found():
    """Verify FileNotFoundError is raised for non-existent configs."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_file_123.yaml")


def test_get_env_existing_variable(monkeypatch):
    """Verify get_env retrieves an active environment variable."""
    monkeypatch.setenv("TEST_QUANT_VAR", "quant_active_123")
    assert get_env("TEST_QUANT_VAR") == "quant_active_123"


def test_get_env_fallback_default(monkeypatch):
    """Verify get_env falls back to provided default when unset."""
    monkeypatch.delenv("UNSET_QUANT_VAR", raising=False)
    assert get_env("UNSET_QUANT_VAR", "fallback_val") == "fallback_val"
    assert get_env("UNSET_QUANT_VAR") is None
