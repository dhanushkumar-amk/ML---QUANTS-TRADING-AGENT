# tests/test_news_collector.py
"""Unit tests for financial news collection, schema normalization, and deduplication."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data_pipeline.data_access import DataAccessLayer
from src.data_pipeline.storage_backend import ParquetBackend
from src.nlp.news_collector import (
    NewsArticle,
    NewsCollector,
    normalize_headline_text,
)


def test_news_article_schema():
    """NewsArticle dataclass should produce expected dictionary schema."""
    article = NewsArticle(
        ticker="AAPL",
        headline="Apple announces new product launch",
        source="Reuters",
        published_at=pd.Timestamp("2023-09-12 17:00:00", tz="UTC"),
        url="https://reuters.com/article/1",
    )
    d = article.to_dict()
    assert d["ticker"] == "AAPL"
    assert d["headline"] == "Apple announces new product launch"
    assert d["source"] == "Reuters"
    assert isinstance(d["published_at"], pd.Timestamp)
    assert d["url"] == "https://reuters.com/article/1"


def test_normalize_headline_text():
    """Text normalizer strips URLs, leading/trailing whitespace, and redundant spaces."""
    raw = "  Apple Surges 5%   https://t.co/abc1234  on Record Earnings! \n "
    clean = normalize_headline_text(raw)
    assert clean == "Apple Surges 5% on Record Earnings!"


def test_map_text_to_ticker():
    """Entity mapping maps company names and products to canonical tickers."""
    assert NewsCollector.map_text_to_ticker("iPhone 15 sales surge worldwide") == "AAPL"
    assert NewsCollector.map_text_to_ticker("Microsoft Azure signs enterprise deal") == "MSFT"
    assert NewsCollector.map_text_to_ticker("Federal Reserve signals pause in rate hikes") == "SPY"
    assert (
        NewsCollector.map_text_to_ticker("Random text without company", default_ticker="SPY")
        == "SPY"
    )


def test_deduplication_exact_and_near_duplicate():
    """Deduplication removes identical headlines and punctuation variants."""
    df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "headline": "Apple reports record quarterly revenue",
                "source": "Reuters",
                "published_at": pd.Timestamp("2023-01-01 10:00:00", tz="UTC"),
                "url": "http://1",
            },
            {
                "ticker": "AAPL",
                "headline": "Apple reports record quarterly revenue!",  # Near duplicate
                "source": "Bloomberg",
                "published_at": pd.Timestamp("2023-01-01 10:05:00", tz="UTC"),
                "url": "http://2",
            },
            {
                "ticker": "AAPL",
                "headline": "Supply chain issues affect iPad production",  # Distinct
                "source": "WSJ",
                "published_at": pd.Timestamp("2023-01-01 11:00:00", tz="UTC"),
                "url": "http://3",
            },
        ]
    )

    deduped = NewsCollector.deduplicate_headlines(df)
    assert len(deduped) == 2
    assert "Supply chain issues affect iPad production" in deduped["headline"].values


def test_collect_hf_news():
    """Test historical news collection pipeline."""
    collector = NewsCollector()
    df = collector.collect_hf_news(tickers=["AAPL"], limit_per_ticker=20)
    assert not df.empty
    assert "ticker" in df.columns
    assert "headline" in df.columns
    assert "published_at" in df.columns
    assert (df["ticker"] == "AAPL").all()


def test_fetch_live_news_mock_api():
    """Test live news fetching with mock HTTP responses and rate limit retry."""
    collector = NewsCollector(alpaca_api_key="TEST_KEY", alpaca_secret_key="TEST_SECRET")

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {"Retry-After": "0.01"}

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "news": [
            {
                "symbols": ["AAPL"],
                "headline": "Apple stock upgraded by analysts",
                "source": "Benzinga",
                "created_at": "2023-05-01T14:30:00Z",
                "url": "https://benzinga.com/1",
            }
        ]
    }

    with patch("requests.Session.get", side_effect=[mock_resp_429, mock_resp_200]):
        df = collector.fetch_live_news(tickers=["AAPL"], limit=10, backoff_factor=0.01)

    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["headline"] == "Apple stock upgraded by analysts"


def test_collect_and_store_integration(tmp_path: Path):
    """Test end-to-end ingestion and persistence in the storage layer."""
    backend = ParquetBackend(root_dir=tmp_path / "processed")
    dal = DataAccessLayer(backend=backend)
    collector = NewsCollector(dal=dal)

    counts = collector.collect_and_store(
        tickers=["AAPL", "MSFT"], sources=["hf"], limit_per_ticker=25
    )
    assert counts["AAPL"] > 0
    assert counts["MSFT"] > 0

    # Query back via DataAccessLayer
    df_aapl = dal.get_news(ticker="AAPL")
    assert not df_aapl.empty
    assert (df_aapl["ticker"] == "AAPL").all()
