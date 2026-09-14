# ============================================================
# FinBERT Financial Sentiment Scorer & Aggregator (Phase 31)
# ============================================================
"""
Pretrained FinBERT sentiment scoring engine with batch inference, caching,
daily feature aggregation, and strict point-in-time anti-leakage validation.

Architecture:
-------------
1. Model:
   - Utilizes `ProsusAI/finbert` (or fine-tuned domain variants) via Hugging Face `transformers`.
   - Maps 3-class logits to calibrated probabilities: Positive, Negative, and Neutral.
   - Computes continuous compound sentiment score $s = p_{\\text{positive}} - p_{\\text{negative}} \\in [-1.0, +1.0]$.
2. Efficiency & Caching:
   - Vectorized batch scoring (`batch_size=32`) with sequence padding and truncation.
   - Dual-tier caching (RAM LRU cache + JSON/Parquet disk cache) to avoid redundant forward passes.
3. Daily Aggregations:
   - `mean_sentiment`: Average sentiment score on day $t$.
   - `sentiment_volatility`: Dispersion/disagreement across headlines on day $t$.
   - `headline_volume`: Article frequency (attention & volatility proxy).
4. Anti-Leakage Protocol:
   - Strict timestamp gating: headlines published at timestamp $T_{\\text{pub}}$ can NEVER be
     used to predict or evaluate price bars closing before $T_{\\text{pub}}$.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data_pipeline.data_access import DataAccessLayer, get_data_access
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Project root and cache locations
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SENTIMENT_CACHE_PATH = _PROJECT_ROOT / "data" / "news" / "sentiment_cache.json"

# Financial Sentiment Lexicon for lightweight/offline CI execution
POSITIVE_FINANCIAL_TOKENS = {
    "surge",
    "surges",
    "surged",
    "rally",
    "rallies",
    "rallied",
    "gain",
    "gains",
    "gained",
    "beat",
    "beats",
    "beating",
    "exceed",
    "exceeds",
    "exceeded",
    "record",
    "growth",
    "expand",
    "expansion",
    "profit",
    "profitable",
    "outperform",
    "outperformed",
    "bullish",
    "upgrade",
    "upgraded",
    "momentum",
    "soar",
    "soars",
    "high",
    "rebound",
}

NEGATIVE_FINANCIAL_TOKENS = {
    "slump",
    "slumps",
    "slumped",
    "drop",
    "drops",
    "dropped",
    "fall",
    "falls",
    "fell",
    "miss",
    "misses",
    "missed",
    "decline",
    "declines",
    "declined",
    "loss",
    "losses",
    "cut",
    "cuts",
    "cutting",
    "downgrade",
    "downgraded",
    "bearish",
    "plunge",
    "plunges",
    "recession",
    "inflation",
    "layoff",
    "layoffs",
    "investigation",
    "scrutiny",
    "outage",
    "dampen",
    "contract",
    "contraction",
    "crash",
    "threaten",
    "threatens",
    "threatened",
    "constraint",
    "constraints",
    "warning",
    "weak",
    "weaken",
    "weakened",
    "down",
    "collapse",
}


def _lexicon_score(text: str) -> dict[str, Any]:
    """Lightweight rule-based financial sentiment scoring fallback."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    pos_matches = sum(1 for t in tokens if t in POSITIVE_FINANCIAL_TOKENS)
    neg_matches = sum(1 for t in tokens if t in NEGATIVE_FINANCIAL_TOKENS)

    total = pos_matches + neg_matches
    if total == 0:
        return {
            "label": "neutral",
            "score": 0.0,
            "confidence": 0.85,
            "probabilities": {"positive": 0.075, "negative": 0.075, "neutral": 0.85},
        }

    p_pos = (pos_matches + 0.1) / (total + 0.3)
    p_neg = (neg_matches + 0.1) / (total + 0.3)
    p_neu = max(0.05, 1.0 - (p_pos + p_neg))
    # Normalize probabilities
    s_sum = p_pos + p_neg + p_neu
    p_pos, p_neg, p_neu = p_pos / s_sum, p_neg / s_sum, p_neu / s_sum

    score = float(p_pos - p_neg)
    if p_pos > p_neg and p_pos > p_neu:
        label = "positive"
        conf = float(p_pos)
    elif p_neg > p_pos and p_neg > p_neu:
        label = "negative"
        conf = float(p_neg)
    else:
        label = "neutral"
        conf = float(p_neu)

    return {
        "label": label,
        "score": score,
        "confidence": conf,
        "probabilities": {
            "positive": float(p_pos),
            "negative": float(p_neg),
            "neutral": float(p_neu),
        },
    }


class FinBERTSentimentScorer:
    """Pretrained FinBERT inference engine with batching and persistent caching.

    Parameters
    ----------
    model_name : str, default 'ProsusAI/finbert'
        Hugging Face model repository ID.
    device : str, default 'cpu'
        Execution device ('cpu' or 'cuda').
    cache_path : Path | str | None
        Disk cache location for previously scored headlines.
    mock_mode : bool, default False
        If True, forces fast deterministic sentiment scoring (ideal for fast CI tests).
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cpu",
        cache_path: Path | str | None = None,
        mock_mode: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.mock_mode = mock_mode
        self.cache_path = Path(cache_path or _DEFAULT_SENTIMENT_CACHE_PATH)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._load_disk_cache()

        self._pipeline = None
        self._model_loaded = False

    def _load_disk_cache(self) -> None:
        """Load persistent cache from disk if available."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._memory_cache = json.load(f)
                logger.info(
                    "Loaded %d scored headlines from cache %s",
                    len(self._memory_cache),
                    self.cache_path,
                )
            except Exception as e:
                logger.warning("Failed to read sentiment cache %s: %s", self.cache_path, e)

    def _save_disk_cache(self) -> None:
        """Flush memory cache to disk."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._memory_cache, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save sentiment cache: %s", e)

    def _init_model(self) -> None:
        """Lazily initialize the transformer pipeline."""
        if self._model_loaded or self.mock_mode:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            logger.info("Loading FinBERT model '%s' on %s...", self.model_name, self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            device_idx = 0 if self.device == "cuda" and torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=device_idx,
                return_all_scores=True,
                truncation=True,
                max_length=128,
            )
            self._model_loaded = True
            logger.info("FinBERT successfully initialized.")
        except Exception as exc:
            logger.warning(
                "Could not load full FinBERT transformer (%s); falling back to financial lexicon mode.",
                exc,
            )
            self.mock_mode = True

    # ------------------------------------------------------------
    # 1. Inference Methods
    # ------------------------------------------------------------

    def score_headline(self, text: str) -> dict[str, Any]:
        """Score sentiment for an individual headline string."""
        clean_text = text.strip()
        if not clean_text:
            return {
                "label": "neutral",
                "score": 0.0,
                "confidence": 1.0,
                "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            }

        # Check cache
        if clean_text in self._memory_cache:
            return self._memory_cache[clean_text]

        results = self.score_batch([clean_text])
        return results[0]

    def score_batch(
        self,
        headlines: Sequence[str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Score a list of headlines in minibatches for production efficiency."""
        self._init_model()

        results: list[dict[str, Any]] = [None] * len(headlines)  # type: ignore
        to_score_indices: list[int] = []
        to_score_texts: list[str] = []

        # Check cache first
        for i, text in enumerate(headlines):
            clean_text = text.strip()
            if clean_text in self._memory_cache:
                results[i] = self._memory_cache[clean_text]
            else:
                to_score_indices.append(i)
                to_score_texts.append(clean_text)

        # Batch evaluate remaining uncached items
        if to_score_texts:
            new_scores: list[dict[str, Any]] = []

            if self.mock_mode or self._pipeline is None:
                for t in to_score_texts:
                    new_scores.append(_lexicon_score(t))
            else:
                try:
                    for b_start in range(0, len(to_score_texts), batch_size):
                        b_texts = to_score_texts[b_start : b_start + batch_size]
                        raw_outputs = self._pipeline(b_texts)

                        for item_outputs in raw_outputs:
                            # Parse probabilities
                            probs = {
                                entry["label"].lower(): float(entry["score"])
                                for entry in item_outputs
                            }
                            p_pos = probs.get("positive", 0.0)
                            p_neg = probs.get("negative", 0.0)
                            p_neu = probs.get("neutral", 0.0)

                            # Continuous compound score
                            comp_score = float(p_pos - p_neg)
                            best_label = max(probs, key=probs.get)
                            conf = float(probs[best_label])

                            new_scores.append(
                                {
                                    "label": best_label,
                                    "score": comp_score,
                                    "confidence": conf,
                                    "probabilities": {
                                        "positive": p_pos,
                                        "negative": p_neg,
                                        "neutral": p_neu,
                                    },
                                }
                            )
                except Exception as exc:
                    logger.warning("Pipeline batch inference error (%s); applying fallback.", exc)
                    for t in to_score_texts[len(new_scores) :]:
                        new_scores.append(_lexicon_score(t))

            # Store in cache and populate results
            for idx, text, score_dict in zip(
                to_score_indices, to_score_texts, new_scores, strict=False
            ):
                self._memory_cache[text] = score_dict
                results[idx] = score_dict

            self._save_disk_cache()

        return results


# ------------------------------------------------------------
# 2. Daily Sentiment Aggregation Engine
# ------------------------------------------------------------


def daily_sentiment_aggregate(
    ticker: str,
    date: str | datetime.date | pd.Timestamp,
    df_news: pd.DataFrame | None = None,
    scorer: FinBERTSentimentScorer | None = None,
    dal: DataAccessLayer | None = None,
) -> dict[str, Any]:
    """Aggregate all headlines for a ticker on a given trading day into feature metrics.

    Parameters
    ----------
    ticker : str
        Asset symbol (e.g. 'AAPL').
    date : str | date | Timestamp
        Trading day date.
    df_news : pd.DataFrame, optional
        Pre-loaded news DataFrame. If None, fetched from DataAccessLayer.
    scorer : FinBERTSentimentScorer, optional
    dal : DataAccessLayer, optional

    Returns
    -------
    dict containing:
        - mean_sentiment : float, average continuous score
        - sentiment_volatility : float, disagreement among headlines
        - headline_volume : int, total articles published
        - pos_headline_count : int
        - neg_headline_count : int
        - neutral_headline_count : int
    """
    target_dt = pd.to_datetime(date).date()
    target_ticker = ticker.upper()

    # Load news if not provided
    if df_news is None:
        data_layer = dal or get_data_access()
        # Query window around the date
        df_news = data_layer.get_news(ticker=target_ticker)

    if df_news.empty or "published_at" not in df_news.columns:
        return {
            "date": str(target_dt),
            "ticker": target_ticker,
            "mean_sentiment": 0.0,
            "sentiment_volatility": 0.0,
            "headline_volume": 0,
            "pos_headline_count": 0,
            "neg_headline_count": 0,
            "neutral_headline_count": 0,
        }

    # Filter to matching ticker and calendar day
    news = df_news.copy()
    news["_pub_date"] = pd.to_datetime(news["published_at"], utc=True).dt.date
    day_news = news[(news["ticker"] == target_ticker) & (news["_pub_date"] == target_dt)]

    if day_news.empty:
        return {
            "date": str(target_dt),
            "ticker": target_ticker,
            "mean_sentiment": 0.0,
            "sentiment_volatility": 0.0,
            "headline_volume": 0,
            "pos_headline_count": 0,
            "neg_headline_count": 0,
            "neutral_headline_count": 0,
        }

    # Score headlines
    fin_scorer = scorer or FinBERTSentimentScorer()
    headlines = day_news["headline"].tolist()
    scored_items = fin_scorer.score_batch(headlines)

    scores = [item["score"] for item in scored_items]
    labels = [item["label"] for item in scored_items]

    mean_s = float(np.mean(scores))
    vol_s = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    vol_cnt = len(scores)
    pos_cnt = sum(1 for lbl in labels if lbl == "positive")
    neg_cnt = sum(1 for lbl in labels if lbl == "negative")
    neu_cnt = sum(1 for lbl in labels if lbl == "neutral")

    return {
        "date": str(target_dt),
        "ticker": target_ticker,
        "mean_sentiment": mean_s,
        "sentiment_volatility": vol_s,
        "headline_volume": vol_cnt,
        "pos_headline_count": pos_cnt,
        "neg_headline_count": neg_cnt,
        "neutral_headline_count": neu_cnt,
    }


def compute_daily_sentiment_series(
    ticker: str,
    start: str | datetime.date | pd.Timestamp | None = None,
    end: str | datetime.date | pd.Timestamp | None = None,
    df_news: pd.DataFrame | None = None,
    scorer: FinBERTSentimentScorer | None = None,
    dal: DataAccessLayer | None = None,
) -> pd.DataFrame:
    """Generate time series of daily sentiment features for a ticker."""
    data_layer = dal or get_data_access()
    target_ticker = ticker.upper()

    if df_news is None:
        df_news = data_layer.get_news(ticker=target_ticker, start=start, end=end)

    if df_news.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "mean_sentiment",
                "sentiment_volatility",
                "headline_volume",
                "pos_headline_count",
                "neg_headline_count",
                "neutral_headline_count",
            ]
        )

    news = df_news.copy()
    news["_pub_date"] = pd.to_datetime(news["published_at"], utc=True).dt.date
    unique_dates = sorted(news["_pub_date"].unique())

    fin_scorer = scorer or FinBERTSentimentScorer()
    records = []
    for d in unique_dates:
        agg = daily_sentiment_aggregate(
            ticker=target_ticker,
            date=d,
            df_news=news,
            scorer=fin_scorer,
            dal=data_layer,
        )
        records.append(agg)

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df["date"] = pd.to_datetime(res_df["date"])
        res_df = res_df.sort_values("date").reset_index(drop=True)
    return res_df


# ------------------------------------------------------------
# 3. Strict Anti-Leakage Timestamp Validation
# ------------------------------------------------------------


def validate_no_news_leakage(
    published_at: pd.Timestamp | str | datetime.datetime,
    target_bar_date: pd.Timestamp | str | datetime.date,
    strict: bool = True,
) -> bool:
    """Enforce strict point-in-time causality between news publication and market data.

    Rule: A headline published at timestamp T_pub can NEVER be incorporated into a
    decision or prediction made at timestamp T_decision unless T_pub <= T_decision.

    Parameters
    ----------
    published_at : Timestamp or datetime
        Timestamp when the article was published.
    target_bar_date : Timestamp, date, or str
        Closing timestamp or reference date of the target bar.
    strict : bool, default True
        If True, raises ValueError upon detecting lookahead leakage.

    Returns
    -------
    is_valid : bool
        True if publication strictly precedes or coincides with the cutoff.
    """
    pub_dt = pd.to_datetime(published_at, utc=True)
    target_dt = pd.to_datetime(target_bar_date, utc=True)

    if pub_dt > target_dt:
        msg = (
            f"DATA LEAKAGE DETECTED: Headline published at {pub_dt} is future relative to "
            f"target timestamp {target_dt} (lookahead difference: {pub_dt - target_dt})."
        )
        logger.error(msg)
        if strict:
            raise ValueError(msg)
        return False

    return True


def filter_leakage_free_news(
    df_news: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp | str | datetime.datetime,
) -> pd.DataFrame:
    """Filter news DataFrame to only include articles published strictly on or before cutoff."""
    if df_news.empty or "published_at" not in df_news.columns:
        return df_news

    cutoff_dt = pd.to_datetime(cutoff_timestamp, utc=True)
    clean_df = df_news.copy()
    clean_df["_pub_utc"] = pd.to_datetime(clean_df["published_at"], utc=True)

    valid_mask = clean_df["_pub_utc"] <= cutoff_dt
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info("Filtered out %d future headlines to prevent lookahead leakage.", n_dropped)

    return clean_df[valid_mask].drop(columns=["_pub_utc"]).reset_index(drop=True)
