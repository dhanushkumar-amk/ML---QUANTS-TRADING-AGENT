# src.portfolio.mean_variance_optimizer — Markowitz Mean-Variance Optimization with Robust Shrinkage
"""Mean-Variance Portfolio Optimization with Ledoit-Wolf Shrinkage and ML Return Forecasts.

Theoretical Overview:
---------------------
Markowitz (1952) Modern Portfolio Theory solves for optimal portfolio weights w
that balance expected return against portfolio variance:

    min  (1/2) * w^T Sigma w - lambda * w^T mu
    s.t. sum(w) = 1,  w_min <= w_i <= w_max

Standard reference points on the efficient frontier include:
1. Minimum Variance Portfolio (MVP):
       min w^T Sigma w  s.t. sum(w) = 1, w >= 0
2. Maximum Sharpe Ratio Portfolio (MSR / Tangency):
       max (w^T mu - r_f) / sqrt(w^T Sigma w)  s.t. sum(w) = 1, w >= 0

Defensible Design Choice: Model-Forecasted Returns vs Historical Sample Means:
-----------------------------------------------------------------------------
Historical sample mean returns are notorious for estimation error. In empirical
finance, sample means have standard errors that can exceed the mean itself over
typical sample sizes (Merton 1980). Feeding naive historical means into a quadratic
optimizer makes Markowitz an "error maximizer" (Michaud 1989), placing the largest
bets on assets whose past returns were overstated purely by luck.
In contrast, using ML model-forecasted returns (e.g., from our Phase 34 production
directional model conditioned on multi-factor features and sentiment shocks):
1. Reflects conditional forward-looking expectations rather than backward-looking noise.
2. Incorporates regime-dependent probability shifts.
3. Provides bounded, regularized signal estimates.

Robust Covariance Estimation: Ledoit-Wolf Shrinkage:
---------------------------------------------------
Sample covariance matrices S = (1/T) X^T X suffer when sample size T is not orders
of magnitude larger than the number of assets N (or when asset returns exhibit high
inter-correlation). Small sample eigenvalues are dispersed: the largest eigenvalues
are biased upward and the smallest eigenvalues are biased downward. When inverted or
optimized against, the optimizer over-allocates to spuriously low-variance combinations.
Ledoit & Wolf (2004) solve this via analytical shrinkage:
    Sigma* = delta * F + (1 - delta) * S
where F is a structured target (e.g., constant correlation or scaled identity) and
delta in [0, 1] is an optimal shrinkage intensity estimated without cross-validation.
Shrinkage reduces the condition number of the covariance matrix, stabilizes optimal
weights, and dramatically cuts out-of-sample portfolio turnover.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def estimate_covariance(
    returns: Union[pd.DataFrame, np.ndarray],
    method: str = "ledoit_wolf",
    annualize: bool = True,
    periods_per_year: int = 252,
) -> Tuple[np.ndarray, Optional[float]]:
    """Estimate asset return covariance matrix using sample covariance or Ledoit-Wolf shrinkage.

    Parameters
    ----------
    returns : pd.DataFrame or np.ndarray
        Matrix of asset returns of shape (n_samples, n_assets).
    method : str, default='ledoit_wolf'
        Covariance estimation method: 'ledoit_wolf' or 'sample'.
    annualize : bool, default=True
        Whether to scale covariance matrix by periods_per_year.
    periods_per_year : int, default=252
        Annualization factor (252 for daily trading days).

    Returns
    -------
    cov_matrix : np.ndarray
        Covariance matrix of shape (n_assets, n_assets).
    shrinkage : float or None
        Optimal shrinkage intensity delta if method='ledoit_wolf', else None.
    """
    if isinstance(returns, pd.DataFrame):
        ret_arr = returns.dropna().values
    else:
        ret_arr = np.asarray(returns)

    if ret_arr.ndim != 2:
        raise ValueError(f"Expected 2D returns array, got shape {ret_arr.shape}")
    if ret_arr.shape[0] < 2:
        raise ValueError(f"Need at least 2 return observations, got {ret_arr.shape[0]}")

    if method == "ledoit_wolf":
        lw = LedoitWolf()
        lw.fit(ret_arr)
        cov_matrix = lw.covariance_
        shrinkage = float(lw.shrinkage_)
    elif method == "sample":
        cov_matrix = np.cov(ret_arr, rowvar=False, ddof=1)
        # Ensure 2D if single asset edge case
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]])
        shrinkage = None
    else:
        raise ValueError(
            f"Unsupported covariance method '{method}'. Choose 'ledoit_wolf' or 'sample'."
        )

    if annualize:
        cov_matrix = cov_matrix * periods_per_year

    return cov_matrix, shrinkage


def compute_model_expected_returns(
    probabilities: Union[pd.Series, Dict[str, float], np.ndarray],
    volatilities: Union[pd.Series, Dict[str, float], np.ndarray],
    scaling_factor: float = 1.0,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Convert directional ML probabilities into annualized expected returns.

    Given a probability p_i in [0, 1] of positive next-period return and annual
    volatility sigma_i, the expected directional signal is centered at zero:
        signal_i = 2 * (p_i - 0.5) = 2 * p_i - 1
    The expected annualized return is scaled proportionally to volatility:
        mu_i = signal_i * sigma_i * scaling_factor

    Parameters
    ----------
    probabilities : pd.Series, dict, or np.ndarray
        Model probabilities of upward movement for each asset.
    volatilities : pd.Series, dict, or np.ndarray
        Annualized volatility estimates for each asset.
    scaling_factor : float, default=1.0
        Multiplier scaling the directional confidence to expected return magnitude.
    periods_per_year : int, default=252
        Number of trading days per year.

    Returns
    -------
    expected_returns : np.ndarray
        Array of expected annualized returns for each asset.
    """
    if isinstance(probabilities, dict):
        p = np.array(list(probabilities.values()), dtype=float)
    elif isinstance(probabilities, pd.Series):
        p = probabilities.values.astype(float)
    else:
        p = np.asarray(probabilities, dtype=float)

    if isinstance(volatilities, dict):
        v = np.array(list(volatilities.values()), dtype=float)
    elif isinstance(volatilities, pd.Series):
        v = volatilities.values.astype(float)
    else:
        v = np.asarray(volatilities, dtype=float)

    if len(p) != len(v):
        raise ValueError(f"Mismatched lengths: probabilities ({len(p)}) vs volatilities ({len(v)})")

    # Bounded directional signal in [-1.0, 1.0]
    directional_signal = 2.0 * p - 1.0
    expected_returns = directional_signal * v * scaling_factor
    return expected_returns


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """Compute portfolio expected return, volatility, and Sharpe ratio.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio asset weights summing to 1.
    expected_returns : np.ndarray
        Expected annualized asset returns.
    cov_matrix : np.ndarray
        Annualized covariance matrix.
    risk_free_rate : float, default=0.0
        Annualized risk-free rate.

    Returns
    -------
    port_return : float
        Expected annualized portfolio return.
    port_vol : float
        Expected annualized portfolio volatility.
    sharpe_ratio : float
        Annualized Sharpe ratio.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)

    port_return = float(np.dot(w, mu))
    variance = float(np.dot(w.T, np.dot(sigma, w)))
    port_vol = float(np.sqrt(max(variance, 1e-12)))
    sharpe_ratio = float((port_return - risk_free_rate) / port_vol)
    return port_return, port_vol, sharpe_ratio


def optimize_minimum_variance(
    cov_matrix: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve for the global Minimum Variance Portfolio on the efficient frontier.

    min_w  w^T Sigma w
    s.t.   sum(w) = 1, min_weight <= w_i <= max_weight

    Parameters
    ----------
    cov_matrix : np.ndarray
        Annualized asset covariance matrix of shape (N, N).
    bounds : list of tuple, optional
        Custom per-asset bounds [(min_0, max_0), ...]. Overrides min_weight/max_weight.
    max_weight : float, default=1.0
        Upper bound cap per asset (e.g. 0.5 for 50% cap).
    min_weight : float, default=0.0
        Lower bound per asset (0.0 enforces no shorting).

    Returns
    -------
    weights : np.ndarray
        Optimal asset weights.
    info : dict
        Optimization status, variance, and volatility.
    """
    sigma = np.asarray(cov_matrix, dtype=float)
    n_assets = sigma.shape[0]

    def objective(w: np.ndarray) -> float:
        return float(np.dot(w.T, np.dot(sigma, w)))

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * np.dot(sigma, w)

    if bounds is None:
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    init_guess = np.ones(n_assets) / n_assets

    res = minimize(
        objective,
        init_guess,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    weights = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
    weights = weights / np.sum(weights)

    port_var = float(np.dot(weights.T, np.dot(sigma, weights)))
    port_vol = float(np.sqrt(max(port_var, 1e-12)))

    info = {
        "success": bool(res.success),
        "message": res.message,
        "portfolio_variance": port_var,
        "portfolio_volatility": port_vol,
    }
    return weights, info


def optimize_maximum_sharpe(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    bounds: Optional[List[Tuple[float, float]]] = None,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve for the Maximum Sharpe Ratio (Tangency) Portfolio.

    max_w  (w^T mu - r_f) / sqrt(w^T Sigma w)
    s.t.   sum(w) = 1, min_weight <= w_i <= max_weight

    Parameters
    ----------
    expected_returns : np.ndarray
        Annualized expected returns of shape (N,).
    cov_matrix : np.ndarray
        Annualized covariance matrix of shape (N, N).
    risk_free_rate : float, default=0.0
        Annualized risk-free rate.
    bounds : list of tuple, optional
        Custom per-asset bounds [(min_0, max_0), ...].
    max_weight : float, default=1.0
        Upper bound cap per asset.
    min_weight : float, default=0.0
        Lower bound per asset (0.0 enforces no shorting).

    Returns
    -------
    weights : np.ndarray
        Optimal asset weights.
    info : dict
        Optimization status, expected return, volatility, Sharpe ratio.
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    n_assets = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = np.dot(w, mu)
        var = np.dot(w.T, np.dot(sigma, w))
        vol = np.sqrt(max(var, 1e-12))
        return float(-(ret - risk_free_rate) / vol)

    if bounds is None:
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    init_guess = np.ones(n_assets) / n_assets

    res = minimize(
        neg_sharpe,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    weights = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
    weights = weights / np.sum(weights)

    ret, vol, sharpe = portfolio_performance(weights, mu, sigma, risk_free_rate)

    info = {
        "success": bool(res.success),
        "message": res.message,
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }
    return weights, info


def compute_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_points: int = 50,
    bounds: Optional[List[Tuple[float, float]]] = None,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> Dict[str, Any]:
    """Trace out the Markowitz Efficient Frontier by sweeping target returns.

    For a sequence of target returns between minimum variance return and maximum
    possible asset return, solves:
        min_w  w^T Sigma w
        s.t.   w^T mu = target_return, sum(w) = 1, min_weight <= w_i <= max_weight

    Parameters
    ----------
    expected_returns : np.ndarray
        Annualized expected returns of shape (N,).
    cov_matrix : np.ndarray
        Annualized covariance matrix of shape (N, N).
    num_points : int, default=50
        Number of points along the efficient frontier.
    bounds : list of tuple, optional
        Custom per-asset bounds.
    max_weight : float, default=1.0
        Upper bound cap per asset.
    min_weight : float, default=0.0
        Lower bound cap per asset.

    Returns
    -------
    frontier : dict
        Keys: 'returns', 'volatilities', 'sharpes', 'weights', 'min_var', 'max_sharpe'.
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    n_assets = len(mu)

    if bounds is None:
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    # 1. Compute Minimum Variance Portfolio
    min_var_w, min_var_info = optimize_minimum_variance(sigma, bounds=bounds)
    min_ret, min_vol, min_sharpe = portfolio_performance(min_var_w, mu, sigma)

    # 2. Compute Maximum Sharpe Ratio Portfolio
    max_sharpe_w, max_sharpe_info = optimize_maximum_sharpe(mu, sigma, bounds=bounds)
    ms_ret, ms_vol, ms_sharpe = portfolio_performance(max_sharpe_w, mu, sigma)

    # Determine frontier return target bounds
    max_possible_return = float(np.max(mu))
    min_target = float(min_ret)
    max_target = max(max_possible_return * 0.99, float(ms_ret))

    target_returns = np.linspace(min_target, max_target, num_points)

    frontier_returns: List[float] = []
    frontier_vols: List[float] = []
    frontier_sharpes: List[float] = []
    frontier_weights: List[np.ndarray] = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mu) - t},
        ]
        res = minimize(
            lambda w: float(np.dot(w.T, np.dot(sigma, w))),
            np.ones(n_assets) / n_assets,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if res.success:
            w_opt = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            w_opt = w_opt / np.sum(w_opt)
            r, v, s = portfolio_performance(w_opt, mu, sigma)
            frontier_returns.append(r)
            frontier_vols.append(v)
            frontier_sharpes.append(s)
            frontier_weights.append(w_opt)

    return {
        "returns": np.array(frontier_returns),
        "volatilities": np.array(frontier_vols),
        "sharpes": np.array(frontier_sharpes),
        "weights": np.array(frontier_weights),
        "min_variance": {
            "weights": min_var_w,
            "return": min_ret,
            "volatility": min_vol,
            "sharpe": min_sharpe,
            "info": min_var_info,
        },
        "max_sharpe": {
            "weights": max_sharpe_w,
            "return": ms_ret,
            "volatility": ms_vol,
            "sharpe": ms_sharpe,
            "info": max_sharpe_info,
        },
    }


def run_rolling_mean_variance_rebalance(
    returns_df: pd.DataFrame,
    signals_df: Optional[pd.DataFrame] = None,
    lookback_window: int = 126,
    rebalance_freq: str = "M",
    cov_method: str = "ledoit_wolf",
    max_weight: float = 0.5,
    min_weight: float = 0.0,
    risk_free_rate: float = 0.0,
    objective_type: str = "max_sharpe",
) -> Dict[str, Any]:
    """Run walk-forward rolling mean-variance portfolio rebalancing.

    At each rebalance date, uses trailing historical returns (up to t) to estimate
    covariance (via sample or Ledoit-Wolf shrinkage) and expected returns (either
    from directional ML signals or trailing mean returns).
    Portfolio weights are held constant until the next rebalance date. Out-of-sample
    daily portfolio returns are realized strictly forward.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns for universe assets with DatetimeIndex.
    signals_df : pd.DataFrame, optional
        Directional probability signals for universe assets. If None, uses trailing
        historical mean returns.
    lookback_window : int, default=126
        Lookback window in trading days for covariance/return estimation (e.g., 6 months).
    rebalance_freq : str, default='M'
        Rebalance frequency string ('M' for monthly, 'W' for weekly, 'Q' for quarterly).
    cov_method : str, default='ledoit_wolf'
        Covariance estimation method: 'ledoit_wolf' or 'sample'.
    max_weight : float, default=0.5
        Maximum weight cap per asset (respecting Kelly-derived diversification limits).
    min_weight : float, default=0.0
        Minimum weight per asset (0.0 for long-only).
    risk_free_rate : float, default=0.0
        Annualized risk-free rate.
    objective_type : str, default='max_sharpe'
        'max_sharpe' or 'min_variance'.

    Returns
    -------
    results : dict
        Contains 'weights' (DataFrame of daily weights), 'rebalance_weights'
        (DataFrame at rebalance dates), 'returns' (Series of daily realized portfolio returns),
        'shrinkages' (Series of Ledoit-Wolf shrinkage parameters over time).
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

    # Pre-allocate daily weight matrix
    weights_matrix = np.zeros((n_days, n_assets))

    rebal_set = set(rebal_dates)
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    for i in range(lookback_window, n_days):
        current_date = clean_returns.index[i]

        if current_date in rebal_set or i == lookback_window:
            # Trailing window strictly up to yesterday (i - 1) to avoid lookahead
            hist_slice = clean_returns.iloc[i - lookback_window : i]
            cov, delta = estimate_covariance(hist_slice, method=cov_method, annualize=True)
            if delta is not None:
                shrinkage_history[current_date] = delta

            # Expected returns
            if signals_df is not None and current_date in signals_df.index:
                # Use ML probabilities
                probs = signals_df.loc[current_date, tickers].values
                vols = hist_slice.std().values * np.sqrt(252)
                exp_ret = compute_model_expected_returns(probs, vols)
            else:
                # Fallback: annualized sample mean
                exp_ret = hist_slice.mean().values * 252

            if objective_type == "min_variance":
                opt_w, _ = optimize_minimum_variance(cov, bounds=bounds)
            else:
                opt_w, _ = optimize_maximum_sharpe(
                    exp_ret, cov, risk_free_rate=risk_free_rate, bounds=bounds
                )

            current_weights = opt_w
            rebalance_weights[current_date] = current_weights

        weights_matrix[i] = current_weights

    # Slice out the out-of-sample period
    oos_index = clean_returns.index[lookback_window:]
    weights_df = pd.DataFrame(weights_matrix[lookback_window:], index=oos_index, columns=tickers)

    rebal_weights_df = pd.DataFrame.from_dict(rebalance_weights, orient="index", columns=tickers)

    # Realized daily portfolio return
    oos_returns = clean_returns.loc[oos_index]
    daily_port_returns = (weights_df * oos_returns).sum(axis=1)

    return {
        "weights": weights_df,
        "rebalance_weights": rebal_weights_df,
        "returns": daily_port_returns,
        "shrinkages": pd.Series(shrinkage_history, name="shrinkage_intensity"),
    }
