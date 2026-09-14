# src.portfolio.risk_parity — Equal Risk Contribution (Risk Parity) Portfolio Allocation
"""Equal Risk Contribution (Risk Parity) Portfolio Optimizer.

Theoretical Overview:
---------------------
In traditional Mean-Variance optimization (Markowitz), portfolio weights are chosen
to maximize Sharpe ratio or minimize variance based on return forecasts mu and
covariance Sigma. However, as demonstrated in Phases 28 (DL vs ML) and 33 (Sentiment
Validation), predicting expected returns out-of-sample is notoriously difficult and
plagued by low signal-to-noise ratios. When mu is noisy, mean-variance optimization
acts as an "error maximizer" (Michaud 1989), allocating extreme capital to assets
whose future returns are overestimated.

Risk Parity (also known as Equal Risk Contribution / ERC, Qian 2005, Maillard et al. 2010)
resolves this fundamental weakness by being completely return-agnostic. Instead of
allocating capital based on uncertain return forecasts, Risk Parity allocates capital
such that every asset contributes equally to total portfolio volatility.

Mathematical Formulation:
-------------------------
For a portfolio with weights w = [w_1, ..., w_N]^T and covariance matrix Sigma:
1. Portfolio Variance:
       sigma_p^2 = w^T Sigma w
2. Portfolio Volatility:
       sigma_p = sqrt(w^T Sigma w)
3. Marginal Risk Contribution (MRC):
       MRC_i = d(sigma_p) / d(w_i) = (Sigma w)_i / sigma_p
4. Total Risk Contribution (RC):
       RC_i = w_i * MRC_i = w_i * (Sigma w)_i / sigma_p
   By Euler's homogeneous function theorem, the sum of risk contributions equals
   total portfolio volatility:
       sum_{i=1}^N RC_i = sigma_p
5. Fractional Risk Contribution (FRC):
       FRC_i = RC_i / sigma_p = w_i * (Sigma w)_i / (w^T Sigma w)
   where sum_{i=1}^N FRC_i = 1.

Equal Risk Contribution Objective:
----------------------------------
Risk Parity seeks weights w such that:
       RC_i = RC_j = sigma_p / N   for all i, j in {1, ..., N}
       (or equivalently, FRC_i = 1 / N)

Because a closed-form analytical solution does not exist for N > 2 correlated assets,
we formulate this as a nonlinear optimization problem:
       min_w  sum_{i=1}^N ( FRC_i - 1/N )^2
       s.t.   sum(w) = 1,  w_i >= 0

Properties & Empirical Intuition:
---------------------------------
- Higher volatility assets receive lower capital weights, while lower volatility assets
  receive higher capital weights.
- When correlation is zero, ERC weights are inversely proportional to asset volatilities:
      w_i proportional to 1 / sigma_i
- Risk Parity prevents high-beta or high-volatility assets from dominating portfolio
  risk, yielding far more stable drawdowns in volatile regimes.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio.mean_variance_optimizer import estimate_covariance


def calculate_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculate Marginal Risk Contributions and Total Risk Contributions.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights of shape (N,) summing to 1.
    cov_matrix : np.ndarray
        Covariance matrix of shape (N, N).

    Returns
    -------
    marginal_risk_contributions : np.ndarray
        MRC_i = (Sigma w)_i / sigma_p of shape (N,).
    risk_contributions : np.ndarray
        RC_i = w_i * MRC_i of shape (N,).
    portfolio_volatility : float
        sigma_p = sqrt(w^T Sigma w).
    """
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)

    port_variance = float(np.dot(w.T, np.dot(sigma, w)))
    port_vol = float(np.sqrt(max(port_variance, 1e-12)))

    # Marginal risk contribution: (Sigma w) / sigma_p
    mrc = np.dot(sigma, w) / port_vol

    # Total risk contribution: w_i * MRC_i
    rc = w * mrc

    return mrc, rc, port_vol


def optimize_risk_parity(
    cov_matrix: np.ndarray,
    target_risk_budget: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    min_weight: float = 1e-4,
    max_weight: float = 1.0,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve for the Equal Risk Contribution (Risk Parity) portfolio weights.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of shape (N, N).
    target_risk_budget : np.ndarray, optional
        Target fractional risk contribution for each asset, summing to 1.
        If None, equal risk budget is used: b_i = 1 / N.
    bounds : list of tuple, optional
        Custom per-asset bounds [(min_0, max_0), ...]. Overrides min_weight/max_weight.
    min_weight : float, default=1e-4
        Lower bound per asset (must be > 0 to ensure strict risk parity).
    max_weight : float, default=1.0
        Upper bound cap per asset.
    tol : float, default=1e-9
        Optimization tolerance.

    Returns
    -------
    weights : np.ndarray
        Optimal Risk Parity weights of shape (N,).
    info : dict
        Optimization details including risk contributions, fractional contributions,
        and max dispersion error.
    """
    sigma = np.asarray(cov_matrix, dtype=float)
    n_assets = sigma.shape[0]

    if target_risk_budget is None:
        b = np.ones(n_assets) / n_assets
    else:
        b = np.asarray(target_risk_budget, dtype=float)
        b = b / np.sum(b)

    if bounds is None:
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    def objective(w: np.ndarray) -> float:
        # Avoid division by zero
        var = float(np.dot(w.T, np.dot(sigma, w)))
        if var <= 1e-12:
            return 1e6
        # Total risk contribution: w * (Sigma w)
        rc = w * np.dot(sigma, w)
        # Fractional risk contribution: rc / var
        frc = rc / var
        # Sum of squared deviations from target risk budget
        return float(np.sum((frc - b) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Initial guess: inverse volatility weighting (good heuristic start)
    asset_vols = np.sqrt(np.diag(sigma))
    asset_vols = np.where(asset_vols > 1e-8, asset_vols, 1.0)
    init_guess = (1.0 / asset_vols) / np.sum(1.0 / asset_vols)

    res = minimize(
        objective,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": 1000},
    )

    weights = np.clip(res.x, [bound[0] for bound in bounds], [bound[1] for bound in bounds])
    weights = weights / np.sum(weights)

    mrc, rc, port_vol = calculate_risk_contributions(weights, sigma)
    frc = rc / port_vol if port_vol > 1e-12 else np.ones(n_assets) / n_assets
    max_dispersion = float(np.max(np.abs(frc - b)))

    info = {
        "success": bool(res.success),
        "message": res.message,
        "portfolio_volatility": port_vol,
        "marginal_risk_contributions": mrc,
        "risk_contributions": rc,
        "fractional_risk_contributions": frc,
        "max_risk_dispersion": max_dispersion,
    }
    return weights, info


def compare_allocations(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: Optional[List[str]] = None,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.5,
) -> pd.DataFrame:
    """Compare Mean-Variance, Risk Parity, and Equal Weight allocations side-by-side.

    Parameters
    ----------
    expected_returns : np.ndarray
        Annualized expected returns of shape (N,).
    cov_matrix : np.ndarray
        Annualized covariance matrix of shape (N, N).
    tickers : list of str, optional
        Asset names. Defaults to Asset_0, Asset_1, ...
    risk_free_rate : float, default=0.0
        Annualized risk-free rate.
    max_weight : float, default=0.5
        Maximum weight cap per asset.

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame with weights, expected returns, volatilities, and Sharpe ratios
        for Equal Weight, Min Variance, Max Sharpe, and Risk Parity.
    """
    from src.portfolio.mean_variance_optimizer import (
        optimize_maximum_sharpe,
        optimize_minimum_variance,
        portfolio_performance,
    )

    n_assets = len(expected_returns)
    if tickers is None:
        tickers = [f"Asset_{i}" for i in range(n_assets)]

    # 1. Equal Weight
    ew_w = np.ones(n_assets) / n_assets
    ew_ret, ew_vol, ew_sharpe = portfolio_performance(
        ew_w, expected_returns, cov_matrix, risk_free_rate
    )
    _, ew_rc, _ = calculate_risk_contributions(ew_w, cov_matrix)

    # 2. Minimum Variance
    mv_w, _ = optimize_minimum_variance(cov_matrix, max_weight=max_weight)
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(
        mv_w, expected_returns, cov_matrix, risk_free_rate
    )
    _, mv_rc, _ = calculate_risk_contributions(mv_w, cov_matrix)

    # 3. Maximum Sharpe
    ms_w, _ = optimize_maximum_sharpe(
        expected_returns, cov_matrix, risk_free_rate=risk_free_rate, max_weight=max_weight
    )
    ms_ret, ms_vol, ms_sharpe = portfolio_performance(
        ms_w, expected_returns, cov_matrix, risk_free_rate
    )
    _, ms_rc, _ = calculate_risk_contributions(ms_w, cov_matrix)

    # 4. Risk Parity
    rp_w, _ = optimize_risk_parity(cov_matrix, max_weight=max_weight)
    rp_ret, rp_vol, rp_sharpe = portfolio_performance(
        rp_w, expected_returns, cov_matrix, risk_free_rate
    )
    _, rp_rc, _ = calculate_risk_contributions(rp_w, cov_matrix)

    rows = []
    strategies = [
        ("Equal Weight (1/N)", ew_w, ew_ret, ew_vol, ew_sharpe, ew_rc),
        ("Min Variance", mv_w, mv_ret, mv_vol, mv_sharpe, mv_rc),
        ("Max Sharpe", ms_w, ms_ret, ms_vol, ms_sharpe, ms_rc),
        ("Risk Parity (ERC)", rp_w, rp_ret, rp_vol, rp_sharpe, rp_rc),
    ]

    for name, w, ret, vol, sharpe, rc in strategies:
        entry = {
            "Strategy": name,
            "Exp Return": f"{ret:.2%}",
            "Volatility": f"{vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
        }
        for ticker, weight, risk_cont in zip(tickers, w, rc, strict=False):
            entry[f"Weight_{ticker}"] = f"{weight:.2%}"
            entry[f"RiskContrib_{ticker}"] = f"{risk_cont:.2%}"
        rows.append(entry)

    return pd.DataFrame(rows)


def run_rolling_risk_parity_rebalance(
    returns_df: pd.DataFrame,
    lookback_window: int = 126,
    rebalance_freq: str = "M",
    cov_method: str = "ledoit_wolf",
    max_weight: float = 0.5,
    min_weight: float = 1e-4,
) -> Dict[str, Any]:
    """Run walk-forward rolling risk parity portfolio rebalancing.

    At each rebalance date, uses trailing historical returns up to t to estimate
    the covariance matrix, then solves for Equal Risk Contribution weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns for universe assets with DatetimeIndex.
    lookback_window : int, default=126
        Lookback window in trading days for covariance estimation.
    rebalance_freq : str, default='M'
        Rebalance frequency ('M' for monthly, 'W' for weekly).
    cov_method : str, default='ledoit_wolf'
        Covariance estimation method: 'ledoit_wolf' or 'sample'.
    max_weight : float, default=0.5
        Maximum weight cap per asset.
    min_weight : float, default=1e-4
        Minimum weight per asset.

    Returns
    -------
    results : dict
        Contains 'weights' (DataFrame of daily weights), 'rebalance_weights'
        (DataFrame at rebalance dates), 'returns' (Series of daily realized portfolio returns),
        'shrinkages' (Series of Ledoit-Wolf shrinkage intensities over time).
    """
    clean_returns = returns_df.dropna()
    tickers = list(clean_returns.columns)
    n_assets = len(tickers)
    n_days = len(clean_returns)

    if n_days <= lookback_window:
        raise ValueError(
            f"returns_df length ({n_days}) must exceed lookback_window ({lookback_window})"
        )

    freq_map = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    rule = freq_map.get(rebalance_freq, rebalance_freq)
    try:
        period_bounds = clean_returns.iloc[lookback_window:].resample(rule).first().index
    except Exception:
        period_bounds = clean_returns.iloc[lookback_window:].resample(rebalance_freq).first().index

    rebal_dates = []
    for dt in period_bounds:
        matching = clean_returns.index[clean_returns.index >= dt]
        if len(matching) > 0:
            rebal_dates.append(matching[0])
    rebal_dates = sorted(set(rebal_dates))
    if not rebal_dates:
        step = 21 if rebalance_freq in ("M", "ME") else 63
        rebal_dates = list(clean_returns.index[lookback_window::step])

    rebalance_weights: Dict[pd.Timestamp, np.ndarray] = {}
    shrinkage_history: Dict[pd.Timestamp, float] = {}

    current_weights = np.ones(n_assets) / n_assets
    weights_matrix = np.zeros((n_days, n_assets))

    rebal_set = set(rebal_dates)
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    for i in range(lookback_window, n_days):
        current_date = clean_returns.index[i]

        if current_date in rebal_set or i == lookback_window:
            hist_slice = clean_returns.iloc[i - lookback_window : i]
            cov, delta = estimate_covariance(hist_slice, method=cov_method, annualize=True)
            if delta is not None:
                shrinkage_history[current_date] = delta

            opt_w, _ = optimize_risk_parity(cov, bounds=bounds)
            current_weights = opt_w
            rebalance_weights[current_date] = current_weights

        weights_matrix[i] = current_weights

    oos_index = clean_returns.index[lookback_window:]
    weights_df = pd.DataFrame(weights_matrix[lookback_window:], index=oos_index, columns=tickers)

    rebal_weights_df = pd.DataFrame.from_dict(rebalance_weights, orient="index", columns=tickers)

    oos_returns = clean_returns.loc[oos_index]
    daily_port_returns = (weights_df * oos_returns).sum(axis=1)

    return {
        "weights": weights_df,
        "rebalance_weights": rebal_weights_df,
        "returns": daily_port_returns,
        "shrinkages": pd.Series(shrinkage_history, name="shrinkage_intensity"),
    }
