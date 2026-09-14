# src.nlp — NLP & sentiment analysis pipelines
"""NLP pipelines: news sentiment, earnings call analysis, social media signals."""

from src.nlp.earnings_sentiment import (
    EarningsCallRecord,
    TranscriptChunk,
    align_earnings_features_to_calendar,
    chunk_transcript_text,
    count_hedge_words,
    extract_earnings_features,
    load_earnings_transcripts,
    parse_transcript_sections,
)
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
from src.nlp.sentiment_validation import (
    build_multimodal_dataset,
    evaluate_sentiment_incremental_value,
    generate_sentiment_verdict,
    run_sentiment_feature_diagnostics,
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
    "EarningsCallRecord",
    "TranscriptChunk",
    "count_hedge_words",
    "parse_transcript_sections",
    "chunk_transcript_text",
    "extract_earnings_features",
    "align_earnings_features_to_calendar",
    "load_earnings_transcripts",
    "build_multimodal_dataset",
    "run_sentiment_feature_diagnostics",
    "evaluate_sentiment_incremental_value",
    "generate_sentiment_verdict",
]
