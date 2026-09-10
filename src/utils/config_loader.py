# ============================================================
# Config Loader — YAML + .env based configuration
# ============================================================
"""
Central config loader for the project.

Usage:
    from src.utils.config_loader import load_config, get_env

    cfg = load_config("configs/default.yaml")
    api_key = get_env("ALPACA_API_KEY")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Auto-load .env when this module is first imported
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict.

    Parameters
    ----------
    path : str | Path
        Absolute or project-relative path to a .yaml / .yml file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    return cfg


def get_env(key: str, default: str | None = None) -> str | None:
    """Read an environment variable (loaded from .env).

    Parameters
    ----------
    key : str
        Environment variable name.
    default : str | None
        Fallback value if the variable is unset.

    Returns
    -------
    str | None
    """
    return os.getenv(key, default)
