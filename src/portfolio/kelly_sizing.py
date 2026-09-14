# ============================================================
# Kelly Criterion Position Sizing Engine (Phase 35)
# ============================================================
"""
Production Kelly Criterion and Fractional Kelly position sizing framework.

=============================================================================
THEORETICAL FOUNDATION: THE KELLY CRITERION IN QUANTITATIVE FINANCE
-----------------------------------------------------------------------------
Formulated by J. L. Kelly Jr. (1956) at Bell Labs and popularized in finance by
Ed Thorp, the Kelly criterion determines the mathematically optimal fraction of wealth
$f^*$ to allocate to an investment with positive expected value in order to maximize
the expected compound growth rate of capital:

    g(f) = E[log(1 + f * R)]

In the classic discrete payoff formulation with win probability $p$, loss probability
$q = 1 - p$, and win/loss payoff ratio $b = \\frac{\\overline{W}}{|\\overline{L}|}$:

    f^* = \\frac{p \\cdot b - q}{b} = p - \\frac{1 - p}{b}

=============================================================================
WHY PRACTITIONERS VIRTUALLY NEVER USE FULL KELLY (f = 1.0)
-----------------------------------------------------------------------------
While full Kelly is asymptotically optimal for an infinite horizon with perfectly
known stationary parameters, professional quant desks virtually never deploy full Kelly:

1. Parameter Estimation Risk:
   - In trading, true $p$ and $b$ are never known; they are estimated with sampling error
     from finite historical bars.
   - If true edge is overestimated even slightly, full Kelly operates in the *overbetting*
     regime where the growth rate turns negative, causing severe capital decay.

2. Extreme Volatility & Catastrophic Drawdowns:
   - A full Kelly bettor has an approximately 33% probability of suffering a 50% capital
     drawdown before doubling their wealth.
   - Such volatility triggers investor redemption, breach of margin limits, and emotional ruin.

3. Non-Gaussian / Fat-Tailed Asset Returns:
   - Kelly assumes stationary binomial or Gaussian returns. Financial asset returns exhibit
     fat tails, volatility clustering, and unexpected jump shocks (e.g. flash crashes).

4. Fractional Kelly (0.25x - 0.5x):
   - Half-Kelly ($f = 0.5 \\times f^*$) captures 75% of the maximum theoretical growth rate
     while slashing return variance by 75% and reducing drawdown probabilities exponentially.
   - Quarter-Kelly ($f = 0.25 \\times f^*$) is standard institutional practice in high-uncertainty
     regimes.

=============================================================================
MULTI-ASSET CORRELATION & SAFETY RAILS
-----------------------------------------------------------------------------
- Correlation Dilemma: Applying univariate Kelly independently to correlated assets (e.g.,
  AAPL, MSFT, and SPY with $\\rho \\approx 0.70$) severely overstates safe leverage, as a market
  selloff causes correlated simultaneous drawdowns.
- Full multi-asset Kelly requires the inverted covariance matrix $\\mathbf{f}^* = \\mathbf{\\Sigma}^{-1} \\mathbf{\\mu}$,
  which Phase 36 formalizes via Mean-Variance Optimization.
- Here we introduce an explicit correlation dampening factor and a hard single-asset
  allocation cap (e.g., max 25% of capital per position) to ensure survival.
=============================================================================
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Core Kelly Mathematical Formulas
# ============================================================


def kelly_criterion(
    win_prob: float,
    win_loss_ratio: float,
) -> float:
    """Calculate the optimal unconstrained Kelly fraction from scratch.

    Parameters
    ----------
    win_prob : float
        Probability of a winning outcome p in [0.0, 1.0].
    win_loss_ratio : float
        Payoff ratio b = (average win amount) / (average loss amount). Must be > 0.

    Returns
    -------
    f_star : float
        Optimal unconstrained Kelly fraction. Returns 0.0 if expected edge <= 0.
    """
    if win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0
    if win_loss_ratio <= 0.0:
        return 0.0

    q = 1.0 - win_prob
    b = float(win_loss_ratio)

    # Edge = p * b - q
    edge = win_prob * b - q
    if edge <= 0.0:
        return 0.0

    f_star = edge / b
    return float(f_star)


def fractional_kelly(
    win_prob: float,
    win_loss_ratio: float,
    fraction: float = 0.5,
) -> float:
    """Calculate fractional Kelly fraction (e.g., 0.5x Half-Kelly, 0.25x Quarter-Kelly).

    Parameters
    ----------
    win_prob : float
        Estimated win probability p in [0.0, 1.0].
    win_loss_ratio : float
        Estimated payoff ratio b > 0.
    fraction : float, default 0.5
        Kelly fraction multiplier kappa in (0.0, 1.0].

    Returns
    -------
    f_frac : float
        Fractional Kelly allocation fraction.
    """
    if fraction <= 0.0:
        return 0.0
    k_fraction = min(1.0, float(fraction))
    f_star = kelly_criterion(win_prob=win_prob, win_loss_ratio=win_loss_ratio)
    return float(f_star * k_fraction)


# ============================================================
# 2. Historical Parameter Derivation
# ============================================================


def derive_kelly_parameters(
    returns: Sequence[float] | pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Derive empirical win probability and payoff ratio from historical trading returns.

    Parameters
    ----------
    returns : array-like
        Series of historical trade or strategy returns.

    Returns
    -------
    win_rate : float
        Empirical proportion of positive returns p.
    payoff_ratio : float
        Ratio of mean winning return to mean losing return magnitude b.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        logger.warning(
            "Insufficient return history (%d observations) for Kelly estimation.", len(arr)
        )
        return 0.50, 1.0

    wins = arr[arr > 0]
    losses = arr[arr < 0]

    if len(wins) == 0:
        return 0.0, 1.0
    if len(losses) == 0:
        return 1.0, 2.0

    win_rate = float(len(wins) / len(arr))
    avg_win = float(np.mean(wins))
    avg_loss = float(np.abs(np.mean(losses)))

    payoff_ratio = float(avg_win / (avg_loss + 1e-9))
    return round(win_rate, 4), round(payoff_ratio, 4)


# ============================================================
# 3. Dynamic Position Sizing Engine
# ============================================================


def position_size(
    model_confidence: float,
    historical_win_rate: float,
    win_loss_ratio: float,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.25,
    min_confidence: float = 0.50,
) -> float:
    """Compute recommended capital allocation % for a single trading position.

    Combines:
    1. Historical base edge (win rate p and payoff ratio b).
    2. Dynamic model prediction confidence scaling (|p_pred - 0.5|).
    3. Fractional Kelly safety dampening (default 0.5x).
    4. Hard risk cap (default 25% max capital).

    Parameters
    ----------
    model_confidence : float
        Model predicted class probability (e.g. 0.58 indicates 58% confidence long).
    historical_win_rate : float
        Empirical out-of-fold win rate p.
    win_loss_ratio : float
        Empirical out-of-fold payoff ratio b.
    kelly_fraction : float, default 0.5
        Fractional Kelly scaling factor (0.5 = Half-Kelly).
    max_position_pct : float, default 0.25
        Hard safety ceiling on single-position capital allocation (25%).
    min_confidence : float, default 0.50
        Threshold below which position size is zero.

    Returns
    -------
    allocation_pct : float
        Recommended allocation as a fraction of portfolio equity in [0.0, max_position_pct].
    """
    if model_confidence < min_confidence:
        return 0.0

    # Base fractional Kelly size
    base_f = fractional_kelly(
        win_prob=historical_win_rate,
        win_loss_ratio=win_loss_ratio,
        fraction=kelly_fraction,
    )
    if base_f <= 0.0:
        return 0.0

    # Dynamic scaling based on confidence excess over 0.50
    # e.g., confidence 0.50 -> 0.0x scale; confidence 0.65 -> 1.5x scale (capped at 2.0x)
    confidence_excess = max(0.0, model_confidence - min_confidence)
    confidence_multiplier = min(2.0, 1.0 + (confidence_excess * 4.0))

    scaled_f = base_f * confidence_multiplier

    # Apply hard risk ceiling
    capped_f = min(float(max_position_pct), float(scaled_f))
    return round(float(capped_f), 4)


# ============================================================
# 4. Multi-Asset Correlation Adjustment
# ============================================================


def adjust_multi_asset_kelly(
    positions: dict[str, float],
    corr_matrix: pd.DataFrame,
    max_portfolio_leverage: float = 1.0,
) -> dict[str, float]:
    """Adjust individual Kelly positions to account for cross-asset correlation.

    THEORETICAL MOTIVATION:
    -----------------------
    If asset A (AAPL) and asset B (MSFT) have correlation rho = 0.75, taking full Kelly
    bets on both simultaneously doubles systematic market exposure and violates
    the independence assumption of univariate Kelly sizing.

    We compute the portfolio diversification multiplier:
        D = sqrt(w^T C w) / sum(w)
    and downscale correlated bets to maintain safe aggregate leverage.

    Parameters
    ----------
    positions : dict[str, float]
        Mapping of ticker -> raw recommended position size.
    corr_matrix : pd.DataFrame
        Empirical pairwise correlation matrix.
    max_portfolio_leverage : float, default 1.0
        Maximum total gross leverage permitted (1.0 = 100% equity).

    Returns
    -------
    adjusted_positions : dict[str, float]
        Correlation-adjusted position sizes.
    """
    active_tickers = [t for t, w in positions.items() if w > 0.0]
    if len(active_tickers) <= 1:
        # Single asset or empty: no cross-asset correlation penalty
        return {t: min(max_portfolio_leverage, w) for t, w in positions.items()}

    sub_tickers = [t for t in active_tickers if t in corr_matrix.index and t in corr_matrix.columns]
    if len(sub_tickers) < len(active_tickers):
        # Missing correlation entries; default to gross leverage scaling
        total_gross = sum(positions.values())
        scale = min(1.0, max_portfolio_leverage / (total_gross + 1e-9))
        return {t: round(w * scale, 4) for t, w in positions.items()}

    w_vec = np.array([positions[t] for t in sub_tickers], dtype=float)
    c_mat = corr_matrix.loc[sub_tickers, sub_tickers].values

    # Portfolio correlated variance factor
    port_var = float(w_vec @ c_mat @ w_vec)
    port_std = np.sqrt(max(1e-6, port_var))
    sum_w = float(np.sum(w_vec))

    # Average correlation penalty
    # If perfectly correlated (C = 1), port_std = sum_w -> dampening = 1.0 / N
    if sum_w > 0:
        dampening = min(1.0, sum_w / (port_std + 1e-9))
    else:
        dampening = 1.0

    # Ensure total gross leverage <= max_portfolio_leverage
    total_leverage = sum_w * dampening
    leverage_scale = min(1.0, max_portfolio_leverage / (total_leverage + 1e-9))

    final_scale = dampening * leverage_scale
    adjusted: dict[str, float] = {}
    for t in positions:
        if t in sub_tickers:
            adjusted[t] = round(float(positions[t] * final_scale), 4)
        else:
            adjusted[t] = 0.0

    return adjusted


# ============================================================
# 5. Position Sizing Time Series Generator
# ============================================================


def generate_kelly_allocation_series(
    probabilities: Sequence[float] | pd.Series,
    asset_returns: Sequence[float] | pd.Series,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.25,
    rolling_window: int = 60,
) -> pd.DataFrame:
    """Generate dynamic time series of Kelly allocations across historical bars.

    Parameters
    ----------
    probabilities : array-like
        Model predicted upward probabilities.
    asset_returns : array-like
        Underlying asset periodic returns.
    kelly_fraction : float, default 0.5
    max_position_pct : float, default 0.25
    rolling_window : int, default 60
        Lookback window for dynamic win rate and payoff ratio estimation.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['confidence', 'rolling_win_rate', 'rolling_payoff',
        'full_kelly', 'fractional_kelly', 'capped_position_size'].
    """
    probs = np.asarray(probabilities, dtype=float)
    rets = np.asarray(asset_returns, dtype=float)
    n = len(probs)

    full_k = np.zeros(n, dtype=float)
    frac_k = np.zeros(n, dtype=float)
    capped = np.zeros(n, dtype=float)
    win_rates = np.zeros(n, dtype=float)
    payoffs = np.zeros(n, dtype=float)

    # Initial estimates from overall sample
    base_p, base_b = derive_kelly_parameters(rets)

    for t in range(n):
        if t >= rolling_window:
            window_rets = rets[t - rolling_window : t]
            p_t, b_t = derive_kelly_parameters(window_rets)
        else:
            p_t, b_t = base_p, base_b

        win_rates[t] = p_t
        payoffs[t] = b_t

        f_full = kelly_criterion(p_t, b_t)
        f_frac = fractional_kelly(p_t, b_t, fraction=kelly_fraction)
        f_pos = position_size(
            model_confidence=probs[t],
            historical_win_rate=p_t,
            win_loss_ratio=b_t,
            kelly_fraction=kelly_fraction,
            max_position_pct=max_position_pct,
        )

        full_k[t] = f_full
        frac_k[t] = f_frac
        capped[t] = f_pos

    idx = probabilities.index if isinstance(probabilities, pd.Series) else pd.RangeIndex(n)
    return pd.DataFrame(
        {
            "confidence": probs,
            "rolling_win_rate": win_rates,
            "rolling_payoff": payoffs,
            "full_kelly": full_k,
            "fractional_kelly": frac_k,
            "capped_position_size": capped,
        },
        index=idx,
    )
