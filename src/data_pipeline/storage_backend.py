# ============================================================
# Storage Backend Abstraction — Partitioned Parquet & TimescaleDB
# ============================================================
"""
Provides a unified storage abstraction layer for market data.

Architectural Design & Decision:
--------------------------------
1. Why Storage Abstraction?
   Decouples data consumption (feature engineering, strategy models, backtesting)
   from low-level file I/O or database drivers. Future storage backends (e.g.,
   cloud object stores like S3/GCS, or distributed databases) can be swapped in
   with zero changes to downstream analytical code.

2. Why Partition by Ticker + Year for ParquetBackend?
   • Ticker Partitioning: In quant finance, >95% of data access queries target a
     single ticker over a historical window (e.g. computing 200-day moving average
     or volatility for AAPL). Partitioning by ticker eliminates cross-ticker scans.
   • Year Partitioning:
     - Keeps individual Parquet files compact and fast to read (~252 rows for daily
       data is ~15-30 KB; ~98,000 rows for 1-minute data is ~2-4 MB compressed).
     - Avoids both extremes: a single monolithic multi-gigabyte file (which slows
       down small date queries) vs. thousands of microscopic daily files (which
       overwhelms filesystem inodes and OS directory listing buffers).
     - Allows partition pruning: querying '2021-03-01 to 2021-09-01' reads only
       `2021.parquet`, skipping all other years entirely.
   • Local-First: Zero infrastructure required for hobbyist development, yet fully
     memory-mappable and compatible with DuckDB and PyArrow.

3. Optional TimescaleDBBackend:
   TimescaleDB is an open-source time-series engine built on PostgreSQL.
   A stubbed implementation and complete DDL schema (`db_schema.py`) are provided
   for optional deployment when concurrent multi-client real-time writes are needed.
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


class StorageBackend(ABC):
    """Abstract interface for all market data storage backends."""

    @abstractmethod
    def save(self, df: pd.DataFrame, key: str, **kwargs) -> Any:
        """Save a DataFrame under the specified key (e.g., ticker symbol)."""
        pass

    @abstractmethod
    def load(self, key: str, **kwargs) -> pd.DataFrame:
        """Load all historical data associated with the key."""
        pass

    @abstractmethod
    def query(
        self,
        key: str,
        start: str | datetime.date | pd.Timestamp | None = None,
        end: str | datetime.date | pd.Timestamp | None = None,
        columns: Sequence[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Query a time slice of data for the given key with column projection."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all available storage keys (e.g. tickers) matching prefix."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove all stored records for the specified key."""
        pass


class ParquetBackend(StorageBackend):
    """Local file-based columnar storage partitioned by ticker and year.

    Directory Layout:
        {root_dir}/
        └── {ticker}/
            ├── 2018.parquet
            ├── 2019.parquet
            └── 2020.parquet
    """

    def __init__(self, root_dir: Path | str | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else _DEFAULT_PROCESSED_DIR
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _ticker_dir(self, ticker: str) -> Path:
        return self.root_dir / ticker.upper()

    def save(
        self,
        df: pd.DataFrame,
        key: str,
        date_col: str = "date",
        overwrite: bool = True,
        **kwargs,
    ) -> list[Path]:
        """Save a DataFrame partitioned by year into {root}/{ticker}/{year}.parquet.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame containing a date column.
        key : str
            Ticker symbol or dataset key.
        date_col : str
            Name of the timestamp/date column.
        overwrite : bool
            Whether to replace existing year partitions or append.

        Returns
        -------
        list[Path]
            Paths of written Parquet partition files.
        """
        if df.empty:
            logger.warning("[%s] Attempted to save empty DataFrame — skipping.", key)
            return []

        if date_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{date_col}' column for time partitioning.")

        ticker = key.upper()
        ticker_path = self._ticker_dir(ticker)
        ticker_path.mkdir(parents=True, exist_ok=True)

        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col]).dt.tz_localize(None)

        written_paths: list[Path] = []
        df_work["_year"] = df_work[date_col].dt.year

        for year, group in df_work.groupby("_year"):
            year_int = int(year)
            out_file = ticker_path / f"{year_int}.parquet"
            data_to_write = group.drop(columns=["_year"])

            if out_file.exists() and not overwrite:
                # Merge existing partition and remove duplicate dates
                existing = pd.read_parquet(out_file)
                merged = pd.concat([existing, data_to_write], ignore_index=True)
                data_to_write = merged.drop_duplicates(subset=[date_col]).sort_values(date_col)

            data_to_write.to_parquet(out_file, index=False, engine="pyarrow")
            written_paths.append(out_file)

        logger.info(
            "[%s] Saved %d rows across %d year partition(s) -> %s",
            ticker,
            len(df),
            len(written_paths),
            ticker_path,
        )
        return written_paths

    def load(self, key: str, **kwargs) -> pd.DataFrame:
        """Load all historical year partitions for the specified ticker."""
        ticker = key.upper()
        ticker_path = self._ticker_dir(ticker)

        if not ticker_path.exists():
            raise FileNotFoundError(f"No storage found for ticker '{ticker}' at {ticker_path}")

        files = sorted(ticker_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No partition files found in {ticker_path}")

        dfs = [pd.read_parquet(f, engine="pyarrow") for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        if "date" in combined.columns:
            combined = combined.sort_values("date").reset_index(drop=True)
        return combined

    def query(
        self,
        key: str,
        start: str | datetime.date | pd.Timestamp | None = None,
        end: str | datetime.date | pd.Timestamp | None = None,
        columns: Sequence[str] | None = None,
        date_col: str = "date",
        **kwargs,
    ) -> pd.DataFrame:
        """Query a time slice for a ticker with partition pruning and column projection.

        Only loads partition files for years overlapping [start, end].
        """
        ticker = key.upper()
        ticker_path = self._ticker_dir(ticker)

        if not ticker_path.exists():
            return pd.DataFrame()

        start_dt = pd.to_datetime(start).tz_localize(None) if start else None
        end_dt = pd.to_datetime(end).tz_localize(None) if end else None

        # Partition Pruning: determine which year files need to be read
        all_files = sorted(ticker_path.glob("*.parquet"))
        candidate_files: list[Path] = []

        for f in all_files:
            stem = f.stem
            if stem.isdigit():
                file_year = int(stem)
                if start_dt and file_year < start_dt.year:
                    continue
                if end_dt and file_year > end_dt.year:
                    continue
            candidate_files.append(f)

        if not candidate_files:
            return pd.DataFrame()

        # Read only required partitions with column projection
        read_cols = list(columns) if columns else None
        if read_cols and date_col not in read_cols:
            read_cols.append(date_col)

        dfs: list[pd.DataFrame] = []
        for f in candidate_files:
            try:
                part_df = pd.read_parquet(f, columns=read_cols, engine="pyarrow")
                dfs.append(part_df)
            except Exception as exc:
                logger.warning("[%s] Error reading partition %s: %s", ticker, f.name, exc)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        if date_col in combined.columns:
            combined[date_col] = pd.to_datetime(combined[date_col])

            # Apply row filtering
            if start_dt:
                combined = combined[combined[date_col] >= start_dt]
            if end_dt:
                combined = combined[combined[date_col] <= end_dt]

            combined = combined.sort_values(date_col).reset_index(drop=True)

        if columns:
            # Retain only originally requested columns in exact order
            cols_to_keep = [c for c in columns if c in combined.columns]
            combined = combined[cols_to_keep]

        return combined

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all available ticker symbols stored in the backend."""
        if not self.root_dir.exists():
            return []

        tickers = [
            p.name
            for p in self.root_dir.iterdir()
            if p.is_dir()
            and any(f.suffix == ".parquet" and f.stem.isdigit() for f in p.glob("*.parquet"))
        ]
        if prefix:
            tickers = [t for t in tickers if t.startswith(prefix.upper())]
        return sorted(tickers)

    def delete(self, key: str) -> bool:
        """Delete all partitioned data for the given ticker."""
        import shutil

        ticker_path = self._ticker_dir(key)
        if ticker_path.exists():
            shutil.rmtree(ticker_path)
            logger.info("Deleted storage for %s", key)
            return True
        return False


class TimescaleDBBackend(StorageBackend):
    """Optional SQL time-series storage backend using TimescaleDB (PostgreSQL).

    Setup Instructions:
    -------------------
    1. Run TimescaleDB Docker container:
       docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=postgres timescale/timescaledb:latest-pg16

    2. Create database & apply schema:
       psql -U postgres -h localhost -c "CREATE DATABASE trading_db;"
       psql -U postgres -h localhost -d trading_db -f src/data_pipeline/migrations/001_initial_timescale.sql

    3. Configure connection string:
       Set environment variable `TIMESCALE_URL=postgresql://postgres:postgres@localhost:5432/trading_db`
    """

    def __init__(self, connection_url: str | None = None) -> None:
        self.connection_url = (
            connection_url or "postgresql://postgres:postgres@localhost:5432/trading_db"
        )
        self._connected = False

    def check_connection(self) -> bool:
        """Verify database connectivity."""
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(self.connection_url)
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            self._connected = True
            return True
        except Exception as exc:
            logger.debug("TimescaleDB not reachable at %s: %s", self.connection_url, exc)
            self._connected = False
            return False

    def save(self, df: pd.DataFrame, key: str, **kwargs) -> Any:
        if not self.check_connection():
            logger.warning(
                "[TimescaleDB] Cannot save '%s': Database connection unavailable. "
                "See TimescaleDBBackend docstring for setup instructions.",
                key,
            )
            return False

        import sqlalchemy

        engine = sqlalchemy.create_engine(self.connection_url)
        df_save = df.copy()
        df_save["ticker"] = key.upper()
        df_save.to_sql("ohlcv_data", engine, if_exists="append", index=False)
        logger.info("[TimescaleDB] Saved %d rows for %s", len(df), key)
        return True

    def load(self, key: str, **kwargs) -> pd.DataFrame:
        if not self.check_connection():
            logger.warning("[TimescaleDB] Cannot load '%s': Database unreachable.", key)
            return pd.DataFrame()

        import sqlalchemy

        engine = sqlalchemy.create_engine(self.connection_url)
        query = sqlalchemy.text(
            "SELECT * FROM ohlcv_data WHERE ticker = :ticker ORDER BY timestamp ASC"
        )
        return pd.read_sql_query(query, engine, params={"ticker": key.upper()})

    def query(
        self,
        key: str,
        start: str | datetime.date | pd.Timestamp | None = None,
        end: str | datetime.date | pd.Timestamp | None = None,
        columns: Sequence[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.check_connection():
            logger.warning("[TimescaleDB] Cannot query '%s': Database unreachable.", key)
            return pd.DataFrame()

        import sqlalchemy

        cols = ", ".join(columns) if columns else "*"
        clauses = ["ticker = :ticker"]
        params: dict[str, Any] = {"ticker": key.upper()}

        if start:
            clauses.append("timestamp >= :start")
            params["start"] = str(start)
        if end:
            clauses.append("timestamp <= :end")
            params["end"] = str(end)

        where = " AND ".join(clauses)
        sql = f"SELECT {cols} FROM ohlcv_data WHERE {where} ORDER BY timestamp ASC"
        engine = sqlalchemy.create_engine(self.connection_url)
        return pd.read_sql_query(sqlalchemy.text(sql), engine, params=params)

    def list_keys(self, prefix: str = "") -> list[str]:
        if not self.check_connection():
            return []

        import sqlalchemy

        engine = sqlalchemy.create_engine(self.connection_url)
        query = sqlalchemy.text("SELECT DISTINCT ticker FROM ohlcv_data ORDER BY ticker")
        df = pd.read_sql_query(query, engine)
        tickers = df["ticker"].tolist()
        if prefix:
            tickers = [t for t in tickers if t.startswith(prefix.upper())]
        return tickers

    def delete(self, key: str) -> bool:
        if not self.check_connection():
            return False

        import sqlalchemy

        engine = sqlalchemy.create_engine(self.connection_url)
        with engine.begin() as conn:
            conn.execute(
                sqlalchemy.text("DELETE FROM ohlcv_data WHERE ticker = :ticker"),
                {"ticker": key.upper()},
            )
        return True
