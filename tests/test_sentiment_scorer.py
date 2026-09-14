# tests/test_sentiment_scorer.py
"""Unit tests for FinBERT sentiment scoring, daily aggregation, and anti-leakage validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nlp.sentiment_scorer import (
    FinBERTSentimentScorer,
    compute_daily_sentiment_series,
    daily_sentiment_aggregate,
    filter_leakage_free_news,
    validate_no_news_leakage,
)


@pytest.fixture
def scorer(tmp_path: Path) -> FinBERTSentimentScorer:
    """Fixture providing a FinBERTSentimentScorer with isolated cache in mock mode for fast tests."""
    cache_file = tmp_path / "sentiment_cache.json"
    return FinBERTSentimentScorer(cache_path=cache_file, mock_mode=True)


def test_score_headline_positive(scorer: FinBERTSentimentScorer):
    """Clearly positive financial headline should receive positive label and score > 0."""
    text = "Apple reports record revenue growth and surging iPhone sales beating estimates"
    res = scorer.score_headline(text)

    assert res["label"] == "positive"
    assert res["score"] > 0.3
    assert res["probabilities"]["positive"] > res["probabilities"]["negative"]


def test_score_headline_negative(scorer: FinBERTSentimentScorer):
    """Clearly negative financial headline should receive negative label and score < 0."""
    text = "Microsoft cloud growth slumps sharply amid global outage and widening losses"
    res = scorer.score_headline(text)

    assert res["label"] == "negative"
    assert res["score"] < -0.3
    assert res["probabilities"]["negative"] > res["probabilities"]["positive"]


def test_score_headline_neutral(scorer: FinBERTSentimentScorer):
    """Factual neutral headline without sentiment adjectives should be neutral."""
    text = "Federal Reserve holds scheduled monthly policy meeting on Wednesday"
    res = scorer.score_headline(text)

    assert res["label"] == "neutral"
    assert np.isclose(res["score"], 0.0, atol=0.2)


def test_score_batch_and_caching(scorer: FinBERTSentimentScorer):
    """Batch scoring should match individual scoring and properly populate cache."""
    headlines = [
        "Tesla deliveries surge to all-time record high",
        "Retail sales decline sharply in fourth quarter",
    ]

    batch_results = scorer.score_batch(headlines)
    assert len(batch_results) == 2
    assert batch_results[0]["label"] == "positive"
    assert batch_results[1]["label"] == "negative"

    # Confirm cache was populated
    assert headlines[0] in scorer._memory_cache
    assert headlines[1] in scorer._memory_cache

    # Second call should retrieve from cache
    cached_res = scorer.score_headline(headlines[0])
    assert cached_res == batch_results[0]


def test_daily_sentiment_aggregate(scorer: FinBERTSentimentScorer):
    """Daily aggregator computes correct mean, volatility, and volume."""
    df_news = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "headline": "Apple reports record profit and surging margins",
                "published_at": pd.Timestamp("2023-05-04 09:30:00", tz="UTC"),
            },
            {
                "ticker": "AAPL",
                "headline": "Supply chain constraints threaten holiday delivery quotas",
                "published_at": pd.Timestamp("2023-05-04 14:00:00", tz="UTC"),
            },
            {
                "ticker": "AAPL",
                "headline": "Apple shares unchanged in regular trading session",
                "published_at": pd.Timestamp("2023-05-04 16:00:00", tz="UTC"),
            },
        ]
    )

    agg = daily_sentiment_aggregate("AAPL", "2023-05-04", df_news=df_news, scorer=scorer)

    assert agg["ticker"] == "AAPL"
    assert agg["date"] == "2023-05-04"
    assert agg["headline_volume"] == 3
    assert agg["pos_headline_count"] == 1
    assert agg["neg_headline_count"] == 1
    assert agg["neutral_headline_count"] == 1
    assert isinstance(agg["mean_sentiment"], float)
    assert agg["sentiment_volatility"] > 0.0  # Disagreement between positive and negative stories


def test_daily_sentiment_aggregate_empty(scorer: FinBERTSentimentScorer):
    """Empty days return zero volume and zero sentiment."""
    empty_df = pd.DataFrame(columns=["ticker", "headline", "published_at"])
    agg = daily_sentiment_aggregate("AAPL", "2023-05-05", df_news=empty_df, scorer=scorer)

    assert agg["headline_volume"] == 0
    assert agg["mean_sentiment"] == 0.0
    assert agg["sentiment_volatility"] == 0.0


def test_compute_daily_sentiment_series(scorer: FinBERTSentimentScorer):
    """Daily series computes daily feature rows for all distinct dates."""
    df_news = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "headline": "Microsoft launches AI service with record demand",
                "published_at": pd.Timestamp("2023-06-01 10:00:00", tz="UTC"),
            },
            {
                "ticker": "MSFT",
                "headline": "Antitrust scrutiny escalates over software bundle",
                "published_at": pd.Timestamp("2023-06-02 11:00:00", tz="UTC"),
            },
        ]
    )

    series_df = compute_daily_sentiment_series("MSFT", df_news=df_news, scorer=scorer)
    assert len(series_df) == 2
    assert "mean_sentiment" in series_df.columns
    assert "headline_volume" in series_df.columns


def test_validate_no_news_leakage_success():
    """Headline published before or at the target bar should pass validation."""
    pub_dt = pd.Timestamp("2023-01-05 14:00:00", tz="UTC")
    bar_dt = pd.Timestamp("2023-01-05 16:00:00", tz="UTC")
    assert validate_no_news_leakage(pub_dt, bar_dt, strict=True) is True


def test_validate_no_news_leakage_violation():
    """Headline published after the target bar represents lookahead leakage and raises ValueError."""
    pub_dt = pd.Timestamp("2023-01-05 16:30:00", tz="UTC")
    bar_dt = pd.Timestamp("2023-01-05 16:00:00", tz="UTC")

    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        validate_no_news_leakage(pub_dt, bar_dt, strict=True)

    assert validate_no_news_leakage(pub_dt, bar_dt, strict=False) is False


def test_filter_leakage_free_news():
    """filter_leakage_free_news filters out any articles published after cutoff."""
    df = pd.DataFrame(
        [
            {"headline": "H1", "published_at": pd.Timestamp("2023-01-05 10:00:00", tz="UTC")},
            {"headline": "H2", "published_at": pd.Timestamp("2023-01-05 15:59:00", tz="UTC")},
            {
                "headline": "H3 (LEAK)",
                "published_at": pd.Timestamp("2023-01-05 16:05:00", tz="UTC"),
            },
        ]
    )
    cutoff = pd.Timestamp("2023-01-05 16:00:00", tz="UTC")

    clean = filter_leakage_free_news(df, cutoff_timestamp=cutoff)
    assert len(clean) == 2
    assert "H3 (LEAK)" not in clean["headline"].values
