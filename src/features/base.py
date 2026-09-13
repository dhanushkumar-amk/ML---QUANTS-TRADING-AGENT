# ============================================================
# Feature Engineering Base Architecture (Phase 12)
# ============================================================
"""
Abstract base class and contract for all feature extraction modules.

Design Principles:
1. Strict Temporal Integrity (No Look-Ahead):
   - Every feature must be computed strictly as of time T using only data available
     at or before time T (data[t <= T]).
   - No rolling operations with center=True or forward-looking shifts (.shift(-k)).
2. Consistent Interface:
   - All extractors inherit from `FeatureBase` and implement `compute(df: pd.DataFrame)`.
   - Preserves original DataFrame indices and row alignment.
3. Clean Namespace & Standardized Column Prefixes:
   - Output feature columns follow descriptive standard conventions
     (e.g., 'mom_20d', 'rsi_14', 'macd_12_26_9').
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBase(ABC):
    """Abstract base class for quantitative feature extractors.

    Parameters
    ----------
    name : str
        Human-readable name of the feature or feature family.
    category : str
        Category classification (e.g., 'momentum', 'mean_reversion', 'volatility',
        'microstructure', 'regime').
    required_columns : Sequence[str] | None
        Input columns required by this extractor (e.g., ['close'] or ['open', 'high', 'low', 'close', 'volume']).
    """

    def __init__(
        self,
        name: str,
        category: str,
        required_columns: Sequence[str] | None = None,
    ) -> None:
        self.name = name
        self.category = category
        self.required_columns = list(required_columns or ["close"])

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate that the input DataFrame meets all structural requirements.

        Parameters
        ----------
        df : pd.DataFrame
            Input market data.

        Raises
        ------
        ValueError
            If df is empty or missing required columns.
        """
        if df is None or df.empty:
            raise ValueError(f"[{self.name}] Input DataFrame is empty or None.")

        lower_cols = {c.lower(): c for c in df.columns}
        missing = [rc for rc in self.required_columns if rc.lower() not in lower_cols]
        if missing:
            raise ValueError(
                f"[{self.name}] Input DataFrame is missing required column(s): {missing}. "
                f"Available: {list(df.columns)}"
            )

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from input market data.

        Parameters
        ----------
        df : pd.DataFrame
            Input OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the newly computed feature columns, matching
            the input DataFrame's index and row count.
        """
        raise NotImplementedError("Subclasses must implement compute().")

    def transform(self, df: pd.DataFrame, append: bool = True) -> pd.DataFrame:
        """Validate input and compute features, optionally appending to input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input market data.
        append : bool, default True
            If True, joins newly computed features to a copy of the input DataFrame.
            If False, returns only the computed feature columns.

        Returns
        -------
        pd.DataFrame
            Feature-engineered DataFrame.
        """
        self.validate_input(df)
        features = self.compute(df)

        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                f"[{self.name}] compute() must return a pandas DataFrame, got {type(features)}."
            )

        if len(features) != len(df):
            raise ValueError(
                f"[{self.name}] Output feature DataFrame length ({len(features)}) "
                f"does not match input DataFrame length ({len(df)})."
            )

        if append:
            out = df.copy()
            for col in features.columns:
                out[col] = features[col].values
            return out

        return features

    def __call__(self, df: pd.DataFrame, append: bool = True) -> pd.DataFrame:
        """Convenience callable delegating to transform."""
        return self.transform(df, append=append)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}')"
