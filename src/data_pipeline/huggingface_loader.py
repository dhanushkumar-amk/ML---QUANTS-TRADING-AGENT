# ============================================================
# Hugging Face Dataset Loader
# ============================================================
"""
Download and cache financial datasets from Hugging Face Hub.

Selected datasets and rationale
-------------------------------
1. **zeroshot/twitter-financial-news-sentiment**  (PRIMARY)
   - ~11,900 tweets about publicly traded companies, each labeled
     Bearish / Bullish / Neutral by human annotators.
   - Covers real-time social-media language and market chatter —
     the kind of unstructured text our trading agent will encounter
     when scanning Twitter/X for live sentiment signals.
   - Stored natively as Parquet on the Hub, so it works out of the
     box with ``datasets >= 5.0`` (no legacy loading scripts).
   - Direct use-case: fine-tuning or evaluating our NLP sentiment
     model in Phase 31-40.

2. **takala/financial_phrasebank**  (SECONDARY — requires workaround)
   - 4,840 English sentences from financial news, labeled
     positive / negative / neutral by 16 finance-background
     annotators.  The gold-standard benchmark for financial NLP
     (FinBERT, FinancialBERT, etc.).
   - NOTE: As of ``datasets >= 5.0`` this repo ships a legacy
     loading script (``financial_phrasebank.py``) that is no longer
     executed.  It may fail to load unless the maintainer converts
     to a standard Parquet/CSV format.  We keep it in the catalogue
     for future use but default to the twitter dataset instead.

Why these?
   • Both are small enough to cache locally without storage concerns.
   • They cover two distinct NLP input distributions (news headlines
     vs. social media) that our trading agent will encounter.
   • Both are widely cited, so results are comparable to published
     benchmarks — good for the resume writeup.

Usage:
    from src.data_pipeline.huggingface_loader import HuggingFaceLoader

    hf = HuggingFaceLoader(cfg["data"])
    df = hf.fetch()                                # default: twitter
    df = hf.fetch("financial_phrasebank")           # try legacy dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ----- project root & default cache dir ---------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _PROJECT_ROOT / "data" / "external" / "hf_cache"

# ----- curated dataset catalogue ----------------------------------------
# Maps friendly keys → (hub_name, config/subset, split)
DATASET_CATALOGUE: dict[str, dict[str, str | None]] = {
    "twitter_financial_sentiment": {
        "path": "zeroshot/twitter-financial-news-sentiment",
        "name": None,
        "split": "train",
    },
    "financial_phrasebank": {
        "path": "takala/financial_phrasebank",
        "name": "sentences_allagree",  # 100 % annotator agreement
        "split": "train",
    },
}

# Default to twitter dataset — works with datasets >= 5.0 out of the box
_DEFAULT_DATASET = "twitter_financial_sentiment"


class HuggingFaceLoader:
    """Download & cache Hugging Face datasets for the NLP pipeline."""

    def __init__(
        self,
        data_cfg: dict[str, Any] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Allow config to override the default dataset key
        self.default_key: str = (data_cfg or {}).get("hf_dataset", _DEFAULT_DATASET)

    def fetch(
        self,
        dataset_key: str | None = None,
    ) -> pd.DataFrame | None:
        """Load a dataset from the catalogue (or arbitrary Hub path).

        Parameters
        ----------
        dataset_key : str | None
            A key from ``DATASET_CATALOGUE`` or a raw Hub path like
            ``"user/dataset"``.  Defaults to ``self.default_key``.

        Returns
        -------
        pd.DataFrame | None
            The dataset as a pandas DataFrame, or None on failure.
        """
        key = dataset_key or self.default_key

        # Resolve catalogue entry (or treat key as a raw Hub path)
        if key in DATASET_CATALOGUE:
            entry = DATASET_CATALOGUE[key]
            hub_path = entry["path"]
            subset = entry["name"]
            split = entry["split"] or "train"
            logger.info(
                "Loading HF dataset '%s' (hub: %s, subset: %s, split: %s)",
                key,
                hub_path,
                subset,
                split,
            )
        else:
            hub_path = key
            subset = None
            split = "train"
            logger.info("Loading arbitrary HF dataset '%s' (split: %s)", hub_path, split)

        try:
            # Lazy import so the rest of the pipeline doesn't break if
            # the `datasets` library isn't installed yet.
            from datasets import load_dataset  # type: ignore[import-untyped]

            # NOTE: trust_remote_code was removed in datasets >= 5.0.
            # Legacy script-based datasets (like financial_phrasebank)
            # may fail; that's expected — use a Parquet-native dataset.
            ds = load_dataset(
                hub_path,
                name=subset,
                split=split,
                cache_dir=str(self.cache_dir),
            )

            df: pd.DataFrame = ds.to_pandas()  # type: ignore[union-attr]

            # Tag with metadata
            df["source"] = "huggingface"
            df["hf_dataset"] = hub_path

            logger.info(
                "HF dataset '%s' loaded — %d rows, %d columns. Columns: %s",
                key,
                len(df),
                len(df.columns),
                list(df.columns),
            )
            return df

        except Exception as exc:
            logger.error("Failed to load HF dataset '%s': %s", key, exc)
            return None

    def list_catalogue(self) -> list[str]:
        """Return the keys available in the built-in catalogue."""
        return list(DATASET_CATALOGUE.keys())
