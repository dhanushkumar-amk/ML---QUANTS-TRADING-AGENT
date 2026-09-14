# src.nlp — NLP & sentiment analysis pipelines
"""NLP pipelines: news sentiment, earnings call analysis, social media signals."""

from src.nlp.news_collector import (
    NewsArticle,
    NewsCollector,
    compute_headline_hash,
    normalize_headline_text,
)
from src.nlp.sentiment_scorer import (
    FinBERTSentimentScorer,
    compute_daily_sentiment_series,
    daily_sentiment_aggregate,
    filter_leakage_free_news,
    validate_no_news_leakage,
)

__all__ = [
    "NewsArticle",
    "NewsCollector",
    "normalize_headline_text",
    "compute_headline_hash",
    "FinBERTSentimentScorer",
    "daily_sentiment_aggregate",
    "compute_daily_sentiment_series",
    "validate_no_news_leakage",
    "filter_leakage_free_news",
]
