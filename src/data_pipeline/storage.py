# ============================================================
# Raw Data Storage — Parquet + Metadata
# ============================================================
"""
Persist fetched data as Parquet files in ``data/raw/`` and
maintain a JSON metadata manifest tracking every fetch.

Directory layout::

    data/raw/
    ├── yfinance/
    │   ├── AAPL.parquet
    │   ├── MSFT.parquet
    │   └── SPY.parquet
    ├── huggingface/
    │   └── financial_phrasebank.parquet
    └── _metadata.json          ← append-only manifest

Usage:
    from src.data_pipeline.storage import save_dataframe, load_dataframe

    save_dataframe(df, source="yfinance", name="AAPL")
    df = load_dataframe(source="yfinance", name="AAPL")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_METADATA_PATH = _RAW_DIR / "_metadata.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---- save --------------------------------------------------------------

def save_dataframe(
    df: pd.DataFrame,
    source: str,
    name: str,
    raw_dir: Path | None = None,
) -> Path:
    """Save a DataFrame as Parquet, partitioned by source.

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    source : str
        Data source tag (``yfinance``, ``huggingface``, etc.).
    name : str
        Logical name (ticker symbol or dataset name).
    raw_dir : Path | None
        Override for the raw data root directory.

    Returns
    -------
    Path
        Absolute path to the written Parquet file.
    """
    base = Path(raw_dir) if raw_dir else _RAW_DIR
    dest_dir = base / source
    _ensure_dir(dest_dir)

    filename = f"{name}.parquet"
    filepath = dest_dir / filename

    df.to_parquet(filepath, index=False, engine="pyarrow")
    logger.info("Saved %s/%s — %d rows → %s", source, name, len(df), filepath)

    # Update metadata manifest
    _append_metadata(
        source=source,
        name=name,
        filepath=str(filepath),
        row_count=len(df),
        columns=list(df.columns),
        date_range=_date_range(df),
    )

    return filepath


# ---- load ---------------------------------------------------------------

def load_dataframe(
    source: str,
    name: str,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a previously saved Parquet file.

    Raises ``FileNotFoundError`` if the file doesn't exist.
    """
    base = Path(raw_dir) if raw_dir else _RAW_DIR
    filepath = base / source / f"{name}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(f"No Parquet found at {filepath}")

    df = pd.read_parquet(filepath, engine="pyarrow")
    logger.info("Loaded %s/%s — %d rows from %s", source, name, len(df), filepath)
    return df


# ---- metadata -----------------------------------------------------------

def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    """Extract min/max date if a 'date' column exists."""
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            return {
                "min_date": str(dates.min().date()),
                "max_date": str(dates.max().date()),
            }
    return {"min_date": None, "max_date": None}


def _append_metadata(
    source: str,
    name: str,
    filepath: str,
    row_count: int,
    columns: list[str],
    date_range: dict[str, str | None],
) -> None:
    """Append an entry to the metadata manifest (JSON lines style)."""
    _ensure_dir(_METADATA_PATH.parent)

    entry: dict[str, Any] = {
        "source": source,
        "name": name,
        "filepath": filepath,
        "row_count": row_count,
        "columns": columns,
        "date_range": date_range,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    # Append to existing manifest or create new one
    manifest: list[dict[str, Any]] = []
    if _METADATA_PATH.exists():
        try:
            manifest = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Corrupt metadata file — resetting.")

    manifest.append(entry)
    _METADATA_PATH.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.debug("Metadata updated: %s/%s", source, name)


def load_metadata(raw_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read the full metadata manifest."""
    meta_path = (Path(raw_dir) if raw_dir else _RAW_DIR) / "_metadata.json"
    if not meta_path.exists():
        return []
    return json.loads(meta_path.read_text(encoding="utf-8"))
