# ============================================================
# Regime Detection Module (Phase 16)
# ============================================================
"""
Market regime identification using Hidden Markov Models (HMM) and Gaussian Mixture Models (GMM).

Theoretical Foundations:
------------------------
Financial markets alternate between distinct latent volatility and trend regimes:
  1. Low-Volatility Trending (Bull/Expansion): Modest positive drift, low variance, high autocorrelation.
  2. High-Volatility Choppy/Crisis (Bear/Panic): Negative drift, wide intraday ranges, high tail risk.

Standard linear models perform poorly when structural regime shifts occur. This module uncovers
unobservable latent states S_t in {0, 1, ..., K-1} strictly from observable return and volatility features:
  - Gaussian Hidden Markov Models (GaussianHMM via `hmmlearn`): Models temporal transitions P(S_t | S_{t-1}).
  - Gaussian Mixture Models (GMM via `sklearn`): Serves as a static density clustering comparison.

Critical Anti-Leakage & State Alignment Protocols:
--------------------------------------------------
1. State Order Standardization:
   Unsupervised clustering algorithms assign state labels arbitrarily (e.g. state 0 could be high-vol in one
   window and low-vol in another). We enforce deterministic ordering by ascending volatility variance:
     State 0: Lowest volatility (Calm / Low-Vol)
     State K-1: Highest volatility (Turbulent / High-Vol)
2. Rolling Out-Of-Sample Inference (`rolling_regime_features`):
   Fitting an HMM over the full dataset [0..T] causes future crash information (e.g. March 2020) to leak into
   2018 regime assignments. We implement rolling refits using only past data [t - window, t].
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

from src.features.base import FeatureBase
from src.features.feature_registry import feature_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Data Structures & Results
# ============================================================


@dataclass
class RegimeModelResult:
    """Structured container for regime detection model outputs."""

    model_name: str
    n_regimes: int
    regime_labels: pd.Series
    regime_probabilities: pd.DataFrame
    transition_matrix: pd.DataFrame | None
    state_means: pd.DataFrame
    state_covariances: list[np.ndarray]
    fitted_model: Any = field(repr=False, default=None)

    def summary_dict(self) -> dict[str, Any]:
        """Return serializable summary of regime properties."""
        return {
            "model_name": self.model_name,
            "n_regimes": self.n_regimes,
            "state_means": self.state_means.to_dict(),
            "transition_matrix": (
                self.transition_matrix.to_dict() if self.transition_matrix is not None else None
            ),
            "regime_counts": self.regime_labels.value_counts().to_dict(),
        }


# ============================================================
# 2. Feature Matrix Construction
# ============================================================


def _prepare_regime_features(
    df: pd.DataFrame,
    price_col: str = "close",
    vol_window: int = 20,
) -> tuple[np.ndarray, pd.Index, list[str]]:
    """Extract return and volatility features for regime clustering."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column.")

    price = df[price_col].astype(float)
    ret = np.log(price / price.shift(1))

    # Volatility proxy: rolling annualized standard deviation
    vol = ret.rolling(window=vol_window, min_periods=vol_window).std() * np.sqrt(252.0)

    feat_df = pd.DataFrame({"return": ret, "volatility": vol}, index=df.index).dropna()

    if len(feat_df) < 60:
        raise ValueError(
            f"Insufficient valid data points ({len(feat_df)}) for regime modeling. Minimum is 60."
        )

    return feat_df.values, feat_df.index, ["return", "volatility"]


# ============================================================
# 3. Gaussian Hidden Markov Model (HMM) Fitting
# ============================================================


def fit_hmm_regimes(
    df: pd.DataFrame,
    n_regimes: int = 2,
    price_col: str = "close",
    vol_window: int = 20,
    seed: int = 42,
    n_iter: int = 150,
) -> RegimeModelResult:
    """Fit a Gaussian Hidden Markov Model (GaussianHMM) to detect latent regimes.

    State Alignment:
    ----------------
    States are automatically re-indexed in order of ascending volatility variance,
    ensuring that State 0 is consistently Low-Vol / Trending Bull, and State K-1 is
    High-Vol / Choppy / Crisis.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    n_regimes : int, default 2
        Number of latent regime states (typically 2 or 3).
    price_col : str, default 'close'
        Price column name.
    vol_window : int, default 20
        Lookback window for volatility feature.
    seed : int, default 42
        Random state for EM algorithm reproducibility.
    n_iter : int, default 150
        Maximum iterations for Baum-Welch (EM) convergence.

    Returns
    -------
    RegimeModelResult
    """
    X, idx, feat_names = _prepare_regime_features(df, price_col=price_col, vol_window=vol_window)

    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.filterwarnings("ignore")
        hmm = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
        )
        hmm.fit(X)

    # Re-order states by ascending volatility variance
    # Feature 1 is volatility
    vol_idx = feat_names.index("volatility")
    # State variances for volatility feature
    vol_variances = [hmm.covars_[k][vol_idx, vol_idx] for k in range(n_regimes)]
    sorted_order = np.argsort(vol_variances)

    # Invert mapping: map old state to new sorted state
    map_to_sorted = {orig_k: new_k for new_k, orig_k in enumerate(sorted_order)}

    raw_states = hmm.predict(X)
    sorted_states = np.array([map_to_sorted[s] for s in raw_states])
    s_labels = pd.Series(sorted_states, index=idx, name="regime_label")

    # Posterior probabilities: re-order columns to match sorted state indices
    raw_posteriors = hmm.predict_proba(X)
    sorted_posteriors = raw_posteriors[:, sorted_order]
    prob_cols = [f"regime_prob_{k}" for k in range(n_regimes)]
    df_probs = pd.DataFrame(sorted_posteriors, index=idx, columns=prob_cols)

    # Re-order transition matrix
    orig_trans = hmm.transmat_
    sorted_trans = orig_trans[np.ix_(sorted_order, sorted_order)]
    df_trans = pd.DataFrame(
        sorted_trans,
        index=[f"from_regime_{k}" for k in range(n_regimes)],
        columns=[f"to_regime_{k}" for k in range(n_regimes)],
    )

    # Re-order state means
    sorted_means = hmm.means_[sorted_order]
    df_means = pd.DataFrame(
        sorted_means, index=[f"regime_{k}" for k in range(n_regimes)], columns=feat_names
    )

    sorted_covars = [hmm.covars_[orig_k] for orig_k in sorted_order]

    return RegimeModelResult(
        model_name="GaussianHMM",
        n_regimes=n_regimes,
        regime_labels=s_labels,
        regime_probabilities=df_probs,
        transition_matrix=df_trans,
        state_means=df_means,
        state_covariances=sorted_covars,
        fitted_model=hmm,
    )


# ============================================================
# 4. Gaussian Mixture Model (GMM) Fitting
# ============================================================


def fit_gmm_regimes(
    df: pd.DataFrame,
    n_regimes: int = 2,
    price_col: str = "close",
    vol_window: int = 20,
    seed: int = 42,
) -> RegimeModelResult:
    """Fit a Gaussian Mixture Model (GMM) for static density regime clustering.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    n_regimes : int, default 2
        Number of clusters.
    price_col : str, default 'close'
        Price column name.
    vol_window : int, default 20
        Lookback window for volatility feature.
    seed : int, default 42
        Random seed.

    Returns
    -------
    RegimeModelResult
    """
    X, idx, feat_names = _prepare_regime_features(df, price_col=price_col, vol_window=vol_window)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            random_state=seed,
        )
        gmm.fit(X)

    # Order by ascending volatility variance
    vol_idx = feat_names.index("volatility")
    vol_variances = [gmm.covariances_[k][vol_idx, vol_idx] for k in range(n_regimes)]
    sorted_order = np.argsort(vol_variances)
    map_to_sorted = {orig_k: new_k for new_k, orig_k in enumerate(sorted_order)}

    raw_states = gmm.predict(X)
    sorted_states = np.array([map_to_sorted[s] for s in raw_states])
    s_labels = pd.Series(sorted_states, index=idx, name="regime_label")

    raw_posteriors = gmm.predict_proba(X)
    sorted_posteriors = raw_posteriors[:, sorted_order]
    prob_cols = [f"regime_prob_{k}" for k in range(n_regimes)]
    df_probs = pd.DataFrame(sorted_posteriors, index=idx, columns=prob_cols)

    sorted_means = gmm.means_[sorted_order]
    df_means = pd.DataFrame(
        sorted_means, index=[f"regime_{k}" for k in range(n_regimes)], columns=feat_names
    )
    sorted_covars = [gmm.covariances_[orig_k] for orig_k in sorted_order]

    return RegimeModelResult(
        model_name="GaussianMixture",
        n_regimes=n_regimes,
        regime_labels=s_labels,
        regime_probabilities=df_probs,
        transition_matrix=None,
        state_means=df_means,
        state_covariances=sorted_covars,
        fitted_model=gmm,
    )


# ============================================================
# 5. Anti-Leakage Rolling Out-Of-Sample Feature Generation
# ============================================================


def rolling_regime_features(
    df: pd.DataFrame,
    window: int = 252,
    refit_frequency: int = 20,
    method: Literal["hmm", "gmm"] = "hmm",
    n_regimes: int = 2,
    price_col: str = "close",
    vol_window: int = 20,
) -> pd.DataFrame:
    """Generate regime features using rolling out-of-sample refits (Zero Lookahead).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    window : int, default 252
        Historical lookback estimation window in bars (e.g. 1 year).
    refit_frequency : int, default 20
        Bars between model parameter re-estimations.
    method : 'hmm' | 'gmm', default 'hmm'
        Clustering algorithm.
    n_regimes : int, default 2
        Number of regimes.
    price_col : str, default 'close'
        Price column name.
    vol_window : int, default 20
        Lookback window for volatility proxy.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'regime_label', 'regime_prob_0', ..., and 'regime_entropy'
        strictly aligned with df.index.
    """
    if window < 60:
        raise ValueError(f"Rolling window ({window}) must be >= 60 for reliable regime estimation.")

    fit_fn = fit_hmm_regimes if method == "hmm" else fit_gmm_regimes

    # Prepare complete feature array
    price = df[price_col].astype(float)
    ret = np.log(price / price.shift(1))
    vol = ret.rolling(window=vol_window, min_periods=vol_window).std() * np.sqrt(252.0)
    feat_df = pd.DataFrame({"return": ret, "volatility": vol}, index=df.index)

    n = len(df)
    out_labels = np.full(n, np.nan, dtype=float)
    out_probs = np.full((n, n_regimes), np.nan, dtype=float)

    current_model_res: RegimeModelResult | None = None

    for i in range(window, n):
        # Time to refit?
        if (i - window) % refit_frequency == 0 or current_model_res is None:
            sub_df = df.iloc[i - window : i]
            try:
                current_model_res = fit_fn(
                    sub_df,
                    n_regimes=n_regimes,
                    price_col=price_col,
                    vol_window=vol_window,
                )
            except Exception as exc:
                logger.debug("Regime refit failed at index %d: %s. Reusing previous model.", i, exc)

        if current_model_res is not None:
            # Predict regime state for current observation i using fitted model
            curr_x = feat_df.iloc[i : i + 1][["return", "volatility"]].values
            if not np.isnan(curr_x).any():
                try:
                    if current_model_res.model_name == "GaussianHMM":
                        hmm_obj = current_model_res.fitted_model
                        raw_prob = hmm_obj.predict_proba(curr_x)[0]
                    else:
                        gmm_obj = current_model_res.fitted_model
                        raw_prob = gmm_obj.predict_proba(curr_x)[0]

                    # Standardize probability order matching current_model_res ordering
                    # Sort order was already applied inside fit_fn, so posterior from predict_proba
                    # needs to be ordered by the fitted model's original covars
                    if current_model_res.model_name == "GaussianHMM":
                        orig_covars = [hmm_obj.covars_[k][1, 1] for k in range(n_regimes)]
                    else:
                        orig_covars = [gmm_obj.covariances_[k][1, 1] for k in range(n_regimes)]
                    sort_order = np.argsort(orig_covars)
                    sorted_prob = raw_prob[sort_order]

                    out_probs[i] = sorted_prob
                    out_labels[i] = int(np.argmax(sorted_prob))
                except Exception:
                    pass

    # Shannon entropy: H = - sum(p * ln(p))
    # Measures regime uncertainty: H ~ 0 when confident, H max when uncertain
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(out_probs > 0, np.log(out_probs), 0.0)
        entropy = -np.sum(out_probs * log_p, axis=1)

    res_dict: dict[str, Any] = {"regime_label": pd.Series(out_labels, index=df.index)}
    for k in range(n_regimes):
        res_dict[f"regime_prob_{k}"] = pd.Series(out_probs[:, k], index=df.index)
    res_dict["regime_entropy"] = pd.Series(entropy, index=df.index)

    return pd.DataFrame(res_dict, index=df.index)


# ============================================================
# 6. Unified Regime Feature Extractor (FeatureBase)
# ============================================================


class RegimeFeatureExtractor(FeatureBase):
    """Unified regime feature extractor subclassing FeatureBase.

    Features Generated:
    - regime_label: Categorical state index (0 = Low-Vol Calm, 1 = High-Vol Crisis)
    - regime_prob_0: Posterior probability of Low-Vol regime
    - regime_prob_1: Posterior probability of High-Vol regime
    - regime_entropy: Shannon uncertainty of regime assignment
    """

    def __init__(
        self,
        name: str = "regime_suite",
        window: int = 252,
        refit_frequency: int = 20,
        method: Literal["hmm", "gmm"] = "hmm",
        n_regimes: int = 2,
        price_col: str = "close",
        vol_window: int = 20,
    ) -> None:
        super().__init__(
            name=name,
            category="regime",
            required_columns=[price_col],
        )
        self.window = window
        self.refit_frequency = refit_frequency
        self.method = method
        self.n_regimes = n_regimes
        self.price_col = price_col
        self.vol_window = vol_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling out-of-sample regime features without lookahead."""
        return rolling_regime_features(
            df=df,
            window=self.window,
            refit_frequency=self.refit_frequency,
            method=self.method,
            n_regimes=self.n_regimes,
            price_col=self.price_col,
            vol_window=self.vol_window,
        )


# ============================================================
# 7. Central Feature Registry Registration
# ============================================================


def _register_regime_features() -> None:
    """Register regime features in global feature_registry."""
    feature_registry.register(
        name="regime_label",
        category="regime",
        description="Categorical market regime state (0 = Low-Vol Calm, 1 = High-Vol Crisis)",
        lookback_horizon=252,
        compute_fn=lambda df: rolling_regime_features(df, window=252, refit_frequency=20)[
            ["regime_label"]
        ],
        tags=["regime", "hmm", "categorical", "latent_state"],
    )
    feature_registry.register(
        name="regime_prob_0",
        category="regime",
        description="Posterior probability of Low-Volatility / Calm regime from rolling HMM",
        lookback_horizon=252,
        compute_fn=lambda df: rolling_regime_features(df, window=252, refit_frequency=20)[
            ["regime_prob_0"]
        ],
        tags=["regime", "probability", "low_vol", "posterior"],
    )
    feature_registry.register(
        name="regime_prob_1",
        category="regime",
        description="Posterior probability of High-Volatility / Crisis regime from rolling HMM",
        lookback_horizon=252,
        compute_fn=lambda df: rolling_regime_features(df, window=252, refit_frequency=20)[
            ["regime_prob_1"]
        ],
        tags=["regime", "probability", "high_vol", "posterior"],
    )
    feature_registry.register(
        name="regime_entropy",
        category="regime",
        description="Shannon entropy of posterior regime probabilities (uncertainty metric)",
        lookback_horizon=252,
        compute_fn=lambda df: rolling_regime_features(df, window=252, refit_frequency=20)[
            ["regime_entropy"]
        ],
        tags=["regime", "entropy", "uncertainty"],
    )


_register_regime_features()
