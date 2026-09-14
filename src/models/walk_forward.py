# ============================================================
# Walk-Forward Validation Engine (Phase 20)
# ============================================================
"""
Production walk-forward time-series cross-validation framework.

=============================================================================
THEORETICAL FOUNDATION: TIME-SERIES VALIDATION VS. STANDARD K-FOLD
-----------------------------------------------------------------------------
Standard scikit-learn K-Fold (or stratified/shuffled cross-validation) is
FUNDAMENTALLY INVALID for financial time series for three core reasons:

1. Lookahead / Time-Travel Leakage:
   Standard K-Fold randomly assigns observations across folds. In fold k, the
   model is trained on future data points (e.g., bar t+100) and evaluated on past
   data points (e.g., bar t). Financial markets exhibit temporal dependency and
   non-stationarity; conditioning on future price action leaks forward distributions
   and yields wildly unrealistic backtest performance that collapses in live trading.

2. Serial Correlation & Information Persistence:
   Financial asset prices and volatility exhibit strong autocorrelation (as diagnosed
   in Phases 4 & 5). Random splitting places adjacent time points (t and t+1) into
   separate train/test sets, allowing the model to simply interpolate from nearest
   neighbors rather than learning genuine causal patterns.

3. Regime Non-Stationarity:
   Financial market regimes (e.g. 2020 COVID shock, 2022 Fed rate hikes) drift over
   time. A valid trading model must prove it can adapt strictly sequentially as
   regimes unfold chronologically.

=============================================================================
PURGE & EMBARGO GAP: PREVENTING FEATURE LOOKBACK CONTAMINATION
-----------------------------------------------------------------------------
When quantitative features are computed over rolling lookback windows (e.g.,
20-day momentum, 50-day moving average, 252-day volatility), an observation at
the start of the test window (t_test_start) relies on price data from the
immediate end of the training window (t_train_end).

Furthermore, if the prediction target is an h-day forward return (e.g. h = 5 days),
the target for the final training bar t_train_end requires price P_{t_train_end + h},
which overlaps directly with the test set if testing begins at t_train_end + 1.

The EMBARGO GAP (buffer period) explicitly excludes `embargo_bars` between the
training partition and the test partition:
[... Training Set ...] --- [EMBARGO BUFFER (Excluded)] --- [... Test Set ...]

This ensures that:
a) No forward-looking target from the training set overlaps with test prices.
b) Autocorrelated feature residuals do not bleed across the boundary.
=============================================================================
"""

from __future__ import annotations

from typing import Any, Callable, Generator, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Walk-Forward Cross-Validation Splitter
# ============================================================


class WalkForwardSplitter:
    """Time-series walk-forward cross-validation splitter compatible with scikit-learn.

    Supports:
    - 'expanding' (anchored origin): Train window expands over time [0, T_train_k].
    - 'rolling' (sliding fixed-size window): Train window maintains fixed length W_train.
    - 'embargo_bars': Buffer between train and test excluded from both to eliminate leakage.

    Parameters
    ----------
    n_splits : int, default 5
        Number of walk-forward folds.
    train_size : int, optional
        Length of training window in bars.
        If None, automatically computed based on sample length and test_size.
    test_size : int, optional
        Length of test window in bars for each fold (e.g., 126 bars = ~6 months).
        If None, automatically computed as (N - min_train_size) // n_splits.
    window_type : {'expanding', 'rolling'}, default 'expanding'
        Cross-validation window type.
    embargo_bars : int, default 5
        Number of bars excluded between train end and test start.
    min_train_size : int, default 252
        Minimum training observations required before first test fold (e.g. 1 trading year).
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int | None = None,
        test_size: int | None = None,
        window_type: Literal["expanding", "rolling"] = "expanding",
        embargo_bars: int = 5,
        min_train_size: int = 252,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
        if window_type not in ("expanding", "rolling"):
            raise ValueError(f"window_type must be 'expanding' or 'rolling', got '{window_type}'.")
        if embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {embargo_bars}.")
        if min_train_size < 10:
            raise ValueError(f"min_train_size must be >= 10, got {min_train_size}.")

        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.window_type = window_type
        self.embargo_bars = embargo_bars
        self.min_train_size = min_train_size

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return number of splitting iterations."""
        return self.n_splits

    def _compute_split_boundaries(self, n_samples: int) -> list[dict[str, int]]:
        """Calculate start and end integer indices for each fold."""
        if n_samples <= self.min_train_size + self.embargo_bars:
            raise ValueError(
                f"Sample length ({n_samples}) is insufficient for min_train_size "
                f"({self.min_train_size}) + embargo ({self.embargo_bars})."
            )

        # Determine minimum training requirement
        required_min_train = (
            self.train_size
            if (self.window_type == "rolling" and self.train_size is not None)
            else self.min_train_size
        )

        # Determine test size
        if self.test_size is not None:
            t_size = self.test_size
        else:
            available_for_test = (
                n_samples - required_min_train - (self.n_splits * self.embargo_bars)
            )
            if available_for_test <= 0:
                raise ValueError(
                    f"Sample length ({n_samples}) too short for {self.n_splits} folds "
                    f"with min_train={required_min_train} and embargo={self.embargo_bars}."
                )
            t_size = available_for_test // self.n_splits
            if t_size < 1:
                t_size = 1

        # Determine train size for rolling window
        r_train_size = self.train_size if self.train_size is not None else self.min_train_size

        splits = []
        # Total test span across all folds
        total_test_span = self.n_splits * t_size
        first_test_start = n_samples - total_test_span

        if first_test_start < required_min_train + self.embargo_bars:
            first_test_start = required_min_train + self.embargo_bars

        for k in range(self.n_splits):
            test_start = first_test_start + (k * t_size)
            test_end = min(test_start + t_size, n_samples)

            train_end = test_start - self.embargo_bars
            if train_end <= 0:
                continue

            if self.window_type == "expanding":
                train_start = 0
            else:  # rolling
                train_start = max(0, train_end - r_train_size)

            # Ensure minimum train size is satisfied
            if (train_end - train_start) < required_min_train:
                continue

            if test_end <= test_start:
                continue

            splits.append(
                {
                    "fold": len(splits) + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

        if len(splits) == 0:
            raise ValueError(
                "Unable to generate valid walk-forward splits with current parameters."
            )

        return splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
        groups : optional
            Ignored.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = len(X)
        boundaries = self._compute_split_boundaries(n_samples)

        for b in boundaries:
            train_idx = np.arange(b["train_start"], b["train_end"], dtype=int)
            test_idx = np.arange(b["test_start"], b["test_end"], dtype=int)
            yield train_idx, test_idx

    def split_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return human-readable metadata table of all walk-forward folds."""
        n_samples = len(X)
        boundaries = self._compute_split_boundaries(n_samples)
        has_dates = isinstance(X.index, pd.DatetimeIndex)

        summary_data = []
        for b in boundaries:
            row: dict[str, Any] = {
                "fold": b["fold"],
                "train_bars": b["train_end"] - b["train_start"],
                "embargo_bars": self.embargo_bars,
                "test_bars": b["test_end"] - b["test_start"],
                "train_start_idx": b["train_start"],
                "train_end_idx": b["train_end"] - 1,
                "test_start_idx": b["test_start"],
                "test_end_idx": b["test_end"] - 1,
            }
            if has_dates:
                row["train_start_date"] = X.index[b["train_start"]].strftime("%Y-%m-%d")
                row["train_end_date"] = X.index[b["train_end"] - 1].strftime("%Y-%m-%d")
                row["test_start_date"] = X.index[b["test_start"]].strftime("%Y-%m-%d")
                row["test_end_date"] = X.index[b["test_end"] - 1].strftime("%Y-%m-%d")
            summary_data.append(row)

        return pd.DataFrame(summary_data).set_index("fold")


# ============================================================
# 2. Walk-Forward Split Visualization
# ============================================================


def plot_walk_forward_splits(
    splitter: WalkForwardSplitter,
    X: pd.DataFrame,
    title: str = "Walk-Forward Validation Splits (Expanding/Rolling with Embargo)",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Render horizontal Gantt chart of training, embargo, and testing windows.

    Parameters
    ----------
    splitter : WalkForwardSplitter
    X : pd.DataFrame with DatetimeIndex or RangeIndex
    title : str
    figsize : tuple

    Returns
    -------
    plt.Figure
    """
    boundaries = splitter._compute_split_boundaries(len(X))
    has_dates = isinstance(X.index, pd.DatetimeIndex)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    n_folds = len(boundaries)

    for i, b in enumerate(boundaries):
        y_pos = n_folds - i - 1

        # Train span
        t_start = X.index[b["train_start"]] if has_dates else b["train_start"]
        t_end = X.index[b["train_end"] - 1] if has_dates else b["train_end"] - 1

        # Embargo span
        if splitter.embargo_bars > 0:
            e_start = X.index[b["train_end"]] if has_dates else b["train_end"]
            e_end = X.index[b["test_start"] - 1] if has_dates else b["test_start"] - 1
        else:
            e_start, e_end = None, None

        # Test span
        te_start = X.index[b["test_start"]] if has_dates else b["test_start"]
        te_end = X.index[b["test_end"] - 1] if has_dates else b["test_end"] - 1

        if has_dates:
            ax.plot(
                [t_start, t_end],
                [y_pos, y_pos],
                color="#1f77b4",
                linewidth=12,
                solid_capstyle="butt",
                label="Train" if i == 0 else "",
            )
            if e_start is not None and e_end is not None and e_end >= e_start:
                ax.plot(
                    [e_start, e_end],
                    [y_pos, y_pos],
                    color="#d62728",
                    linewidth=12,
                    solid_capstyle="butt",
                    label="Embargo Gap" if i == 0 else "",
                )
            ax.plot(
                [te_start, te_end],
                [y_pos, y_pos],
                color="#2ca02c",
                linewidth=12,
                solid_capstyle="butt",
                label="Test" if i == 0 else "",
            )
        else:
            ax.barh(
                y_pos,
                b["train_end"] - b["train_start"],
                left=b["train_start"],
                color="#1f77b4",
                height=0.6,
                label="Train" if i == 0 else "",
            )
            if splitter.embargo_bars > 0:
                ax.barh(
                    y_pos,
                    splitter.embargo_bars,
                    left=b["train_end"],
                    color="#d62728",
                    height=0.6,
                    label="Embargo Gap" if i == 0 else "",
                )
            ax.barh(
                y_pos,
                b["test_end"] - b["test_start"],
                left=b["test_start"],
                color="#2ca02c",
                height=0.6,
                label="Test" if i == 0 else "",
            )

    fold_labels = [f"Fold {b['fold']}" for b in reversed(boundaries)]
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels(fold_labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Date" if has_dates else "Sample Bar Index", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    return fig


# ============================================================
# 3. Walk-Forward Evaluation Runner
# ============================================================


def evaluate_walk_forward(
    model_factory: Callable[[], Any],
    splitter: WalkForwardSplitter,
    X: pd.DataFrame,
    y: pd.Series,
    eval_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], dict[str, float]],
    model_name: str = "Model",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute walk-forward cross-validation evaluation for any model estimator.

    Parameters
    ----------
    model_factory : callable
        Function returning a fresh estimator instance with fit, predict, and predict_proba methods.
    splitter : WalkForwardSplitter
    X : pd.DataFrame
    y : pd.Series
    eval_fn : callable
        Function (y_true, y_pred, y_prob) -> dict of metric values.
    model_name : str, default 'Model'

    Returns
    -------
    fold_metrics_df : pd.DataFrame
        Per-fold diagnostic metrics plus summary mean/std rows.
    oof_predictions_df : pd.DataFrame
        Concatenated out-of-sample test predictions aligned by time.
    """
    logger.info("Starting walk-forward evaluation for '%s'...", model_name)
    fold_results = []
    oof_dfs = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train_f = X.iloc[train_idx]
        y_train_f = y.iloc[train_idx]
        X_test_f = X.iloc[test_idx]
        y_test_f = y.iloc[test_idx]

        model = model_factory()

        # Handle NaivePersistenceModel vs Standard estimators
        if hasattr(model, "fit"):
            model.fit(X_train_f, y_train_f)

        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X_test_f)
            except TypeError:
                prob = model.predict_proba(X_test_f, y_actual=y_test_f)
        else:
            prob = None

        if hasattr(model, "predict"):
            try:
                pred = model.predict(X_test_f)
            except TypeError:
                pred = model.predict(X_test_f, y_actual=y_test_f)
        else:
            raise AttributeError(f"Model {model} must implement predict.")

        metrics = eval_fn(y_test_f.values, pred, prob)
        metrics["fold"] = fold_idx
        metrics["train_bars"] = len(train_idx)
        metrics["test_bars"] = len(test_idx)
        if isinstance(X.index, pd.DatetimeIndex):
            metrics["test_start_date"] = X.index[test_idx[0]].strftime("%Y-%m-%d")
            metrics["test_end_date"] = X.index[test_idx[-1]].strftime("%Y-%m-%d")

        fold_results.append(metrics)

        # Store out-of-sample predictions
        oof_df = pd.DataFrame(
            {
                "y_true": y_test_f.values,
                "y_pred": pred,
                "y_prob_1": prob[:, 1] if prob is not None and prob.shape[1] >= 2 else np.nan,
                "fold": fold_idx,
            },
            index=X_test_f.index,
        )
        oof_dfs.append(oof_df)

    fold_df = pd.DataFrame(fold_results).set_index("fold")
    oof_all = pd.concat(oof_dfs).sort_index()

    # Calculate aggregate out-of-sample metrics across all folds combined
    overall_metrics = eval_fn(
        oof_all["y_true"].values, oof_all["y_pred"].values, oof_all[["y_true", "y_prob_1"]].values
    )
    overall_metrics["fold"] = "AGGREGATE"
    overall_metrics["train_bars"] = int(fold_df["train_bars"].mean())
    overall_metrics["test_bars"] = len(oof_all)

    logger.info(
        "Walk-forward evaluation complete for '%s': Aggregate Acc=%.4f",
        model_name,
        overall_metrics.get("accuracy", 0.0),
    )
    return fold_df, oof_all
