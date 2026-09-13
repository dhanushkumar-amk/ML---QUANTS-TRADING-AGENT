# ============================================================
# Feature Scaling & Normalization Pipeline (Phase 17)
# ============================================================
"""
Unified feature scaling, normalization, and assembly pipeline for quantitative machine learning.

Methodological Principles:
---------------------------
1. Prevention of Lookahead / Data Leakage:
   Scalers must strictly be fitted ONLY on historical training partitions (t <= T_train) and
   subsequently applied via `.transform()` to out-of-sample validation and test partitions.
   Computing global statistics across the entire dataset introduces subtle lookahead bias into
   model inputs.

2. Fat-Tailed Financial Distributions (Robust Scaling):
   Financial features (returns, volume spikes, Amihud illiquidity) exhibit heavy tails and excess
   kurtosis (as confirmed in Phase 8). Standard Z-score scaling is distorted by extreme outliers.
   The RobustScaler (based on median and Interquartile Range, IQR = Q75 - Q25) provides robust
   centering and scaling that preserves tail structure without blowing up feature variances.

3. Explicit Separation of Time-Series vs. Cross-Sectional Scaling:
   - Time-Series Scaling: Normalizes a single asset's features relative to its own time-series history.
   - Cross-Sectional Scaling: Normalizes features across the universe of assets on a single date t
     (e.g., cross-sectional momentum z-score or percentile rank across the S&P 500).

4. Principled Missing-Value Strategy:
   Financial data contains structural missing values (warmup periods from rolling windows, holiday
   closures). The pipeline implements:
   - Bounded forward-fill (max_ffill = 5 bars) for short calendar gaps.
   - Strict dropping of initial warmup NaNs.
   - ZERO SILENT FILLING: Zero-filling missing indicators (e.g. replacing NaN with 0.0) distorts
     economic signals (e.g. implying 0% volatility or neutral RSI) and is strictly forbidden.
"""

from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

from src.features.feature_registry import feature_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Time-Series Scaler (scikit-learn compatible)
# ============================================================


class TimeSeriesScaler:
    """Per-feature time-series scaler preventing lookahead leakage.

    Supports:
    - 'standard': z = (x - mean) / std
    - 'minmax': z = (x - min) / (max - min)
    - 'robust': z = (x - median) / IQR
    """

    def __init__(
        self,
        method: Literal["standard", "minmax", "robust"] = "robust",
        clip_outliers: float | None = 5.0,
    ) -> None:
        self.method = method
        self.clip_outliers = clip_outliers
        self.stats_: dict[str, dict[str, float]] = {}
        self.columns_: list[str] = []
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> TimeSeriesScaler:
        """Compute normalization statistics strictly on the provided training partition."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        self.stats_ = {}
        self.columns_ = list(df.columns)

        for col in self.columns_:
            s = df[col].dropna().astype(float)
            if len(s) == 0:
                self.stats_[col] = {"center": 0.0, "scale": 1.0}
                continue

            if self.method == "standard":
                mean_val = float(s.mean())
                std_val = float(s.std(ddof=1))
                scale_val = std_val if std_val > 1e-8 else 1.0
                self.stats_[col] = {"center": mean_val, "scale": scale_val}

            elif self.method == "minmax":
                min_val = float(s.min())
                max_val = float(s.max())
                scale_val = (max_val - min_val) if (max_val - min_val) > 1e-8 else 1.0
                self.stats_[col] = {"center": min_val, "scale": scale_val}

            elif self.method == "robust":
                med_val = float(s.median())
                q25 = float(s.quantile(0.25))
                q75 = float(s.quantile(0.75))
                iqr = q75 - q25
                # Fallback to std if IQR is collapsed (e.g. constant data)
                if iqr <= 1e-8:
                    std_val = float(s.std(ddof=1))
                    scale_val = std_val if std_val > 1e-8 else 1.0
                else:
                    scale_val = iqr
                self.stats_[col] = {"center": med_val, "scale": scale_val}

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-computed training statistics to transform any partition."""
        if not self.is_fitted_:
            raise ValueError("Scaler must be fitted before calling transform.")

        out_df = df.copy()
        for col in self.columns_:
            if col in out_df.columns:
                stat = self.stats_[col]
                center = stat["center"]
                scale = stat["scale"]
                z = (out_df[col].astype(float) - center) / scale

                if self.clip_outliers is not None and self.method in ("standard", "robust"):
                    z = z.clip(lower=-self.clip_outliers, upper=self.clip_outliers)

                out_df[col] = z

        return out_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return transformed DataFrame."""
        return self.fit(df).transform(df)

    def transform_rolling(self, df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """Perform dynamic online rolling time-series scaling without lookahead."""
        out_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            s = df[col].astype(float)
            if self.method == "standard":
                roll_mean = s.rolling(window=window, min_periods=window).mean()
                roll_std = (
                    s.rolling(window=window, min_periods=window).std(ddof=1).replace(0.0, 1.0)
                )
                z = (s - roll_mean) / roll_std
            elif self.method == "minmax":
                roll_min = s.rolling(window=window, min_periods=window).min()
                roll_max = s.rolling(window=window, min_periods=window).max()
                denom = (roll_max - roll_min).replace(0.0, 1.0)
                z = (s - roll_min) / denom
            elif self.method == "robust":
                roll_med = s.rolling(window=window, min_periods=window).median()
                roll_q25 = s.rolling(window=window, min_periods=window).quantile(0.25)
                roll_q75 = s.rolling(window=window, min_periods=window).quantile(0.75)
                roll_iqr = (roll_q75 - roll_q25).replace(0.0, 1.0)
                z = (s - roll_med) / roll_iqr

            if self.clip_outliers is not None and self.method in ("standard", "robust"):
                z = z.clip(lower=-self.clip_outliers, upper=self.clip_outliers)

            out_df[col] = z

        return out_df


# ============================================================
# 2. Cross-Sectional Scaler
# ============================================================


class CrossSectionalScaler:
    """Cross-sectional normalization across universe constituents on each timestamp."""

    def __init__(
        self,
        method: Literal["standard", "rank"] = "standard",
    ) -> None:
        self.method = method

    def transform(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        feature_cols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Normalize features cross-sectionally for each date partition."""
        out_df = df.copy()

        cols_to_scale = (
            list(feature_cols)
            if feature_cols is not None
            else [c for c in df.columns if c not in (date_col, "ticker")]
        )

        if date_col in out_df.columns:
            groups = out_df.groupby(date_col)
        elif isinstance(out_df.index, pd.DatetimeIndex):
            groups = out_df.groupby(out_df.index)
        else:
            raise ValueError(
                "Data must have a date column or DatetimeIndex for cross-sectional scaling."
            )

        def _scale_group(group: pd.DataFrame) -> pd.DataFrame:
            grp = group.copy()
            for col in cols_to_scale:
                s = grp[col].astype(float)
                if len(s.dropna()) <= 1:
                    continue
                if self.method == "standard":
                    m = s.mean()
                    std = s.std(ddof=1)
                    grp[col] = (s - m) / (std if std > 1e-8 else 1.0)
                elif self.method == "rank":
                    grp[col] = (s.rank(method="average", pct=True) - 0.5) * 2.0  # Centered [-1, 1]
            return grp

        return groups.apply(_scale_group).reset_index(drop=True)


# ============================================================
# 3. End-to-End Feature Assembly Pipeline
# ============================================================


def _ensure_registered() -> None:
    """Ensure all feature modules have executed their top-level registration."""
    import src.features.mean_reversion_features  # noqa: F401
    import src.features.microstructure_proxies  # noqa: F401
    import src.features.momentum_features  # noqa: F401
    import src.features.regime_detection  # noqa: F401
    import src.features.volatility_models  # noqa: F401
    import src.features.volume_features  # noqa: F401


class FeaturePipeline:
    """Production pipeline integrating feature extraction, cleaning, and scaling."""

    def __init__(
        self,
        feature_names: list[str] | None = None,
        scaler_method: Literal["standard", "minmax", "robust"] = "robust",
        max_ffill: int = 5,
        drop_warmup: bool = True,
        clip_outliers: float | None = 5.0,
    ) -> None:
        self.feature_names = feature_names
        self.scaler = TimeSeriesScaler(method=scaler_method, clip_outliers=clip_outliers)
        self.max_ffill = max_ffill
        self.drop_warmup = drop_warmup
        self.is_fitted_ = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract requested features from global registry."""
        _ensure_registered()
        if self.feature_names is None:
            reg_df = feature_registry.to_dataframe()
            names = list(reg_df["name"].values)
        else:
            names = self.feature_names

        return feature_registry.compute_features(df, feature_names=names)

    def clean_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Enforce strict missing value policy (bounded ffill, drop warmup, no silent zero-fill)."""
        cleaned = feat_df.copy()

        # Bounded forward-fill for short calendar gaps
        if self.max_ffill > 0:
            cleaned = cleaned.ffill(limit=self.max_ffill)

        # Drop initial warmup NaNs
        if self.drop_warmup:
            cleaned = cleaned.dropna()

        # Verify no NaNs or Infs remain
        if cleaned.isna().any().any():
            nan_cols = cleaned.columns[cleaned.isna().any()].tolist()
            raise ValueError(
                f"Feature matrix contains NaNs in columns {nan_cols} after cleaning. "
                "Silent zero-fill is prohibited; investigate warmup length or missing inputs."
            )

        return cleaned

    def fit(self, df_train: pd.DataFrame) -> FeaturePipeline:
        """Extract, clean, and fit scalers strictly on training data."""
        raw_feats = self.extract_features(df_train)
        clean_feats = self.clean_features(raw_feats)
        self.scaler.fit(clean_feats)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract, clean, and apply pre-fitted scalers."""
        if not self.is_fitted_:
            raise ValueError(
                "FeaturePipeline must be fitted on training data before calling transform."
            )

        raw_feats = self.extract_features(df)
        clean_feats = self.clean_features(raw_feats)
        return self.scaler.transform(clean_feats)

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training dataset."""
        return self.fit(df_train).transform(df_train)
