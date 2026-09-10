# ============================================================
# Unit Tests — Hugging Face Dataset Loader (Phase 2)
# ============================================================

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data_pipeline.huggingface_loader import DATASET_CATALOGUE, HuggingFaceLoader


def test_catalogue_contents():
    loader = HuggingFaceLoader()
    catalogue = loader.list_catalogue()
    assert "twitter_financial_sentiment" in catalogue
    assert "financial_phrasebank" in catalogue
    assert loader.default_key == "twitter_financial_sentiment"


def test_loader_custom_cache_dir(tmp_path: Path):
    custom_cache = tmp_path / "custom_hf_cache"
    loader = HuggingFaceLoader(cache_dir=custom_cache)
    assert loader.cache_dir == custom_cache
    assert custom_cache.exists()


def test_fetch_success_mock(tmp_path: Path):
    mock_df = pd.DataFrame(
        {
            "text": ["AAPL reports record revenue", "Market drops on inflation fears"],
            "label": [1, 0],
        }
    )

    mock_dataset = MagicMock()
    mock_dataset.to_pandas.return_value = mock_df.copy()

    # Mock datasets module
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = mock_dataset

    with patch.dict(sys.modules, {"datasets": mock_datasets}):
        loader = HuggingFaceLoader(cache_dir=tmp_path)
        df = loader.fetch("twitter_financial_sentiment")

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "source" in df.columns
        assert df["source"].iloc[0] == "huggingface"
        assert "hf_dataset" in df.columns
        assert df["hf_dataset"].iloc[0] == DATASET_CATALOGUE["twitter_financial_sentiment"]["path"]


def test_fetch_arbitrary_hub_path(tmp_path: Path):
    mock_df = pd.DataFrame({"news": ["Fed raises rates"], "sentiment": ["bearish"]})
    mock_dataset = MagicMock()
    mock_dataset.to_pandas.return_value = mock_df.copy()

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = mock_dataset

    with patch.dict(sys.modules, {"datasets": mock_datasets}):
        loader = HuggingFaceLoader(cache_dir=tmp_path)
        df = loader.fetch("custom_user/finance_nlp")

        assert df is not None
        assert df["hf_dataset"].iloc[0] == "custom_user/finance_nlp"


def test_fetch_exception_returns_none(tmp_path: Path):
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.side_effect = ConnectionError("HF Hub Unreachable")

    with patch.dict(sys.modules, {"datasets": mock_datasets}):
        loader = HuggingFaceLoader(cache_dir=tmp_path)
        df = loader.fetch("twitter_financial_sentiment")

        assert df is None
