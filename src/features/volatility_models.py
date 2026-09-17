# ============================================================
# Volatility Modeling & GARCH Dynamics (Phase 14)
# ============================================================
"""
Production-grade volatility modeling capturing volatility clustering and leverage effects
via GARCH(1,1), GJR-GARCH(1,1,1), and EGARCH(1,1,1).

DATA LEAKAGE SAFEGUARDS (MANDATORY REQUIREMENT):
------------------------------------------------
1. Parameter Fitting Over Time (Strict Temporal Integrity):
   In historical feature generation and backtesting, fitting a single full-sample GARCH model
   leaks future volatility regimes and crisis events back into historical rows.
2. Rolling/Expanding Refits:
   Features for downstream ML/backtesting must be generated via `rolling_garch_features`,
   where model parameters are periodically refit strictly using past observations [t-window, t],
   projecting conditional volatility out-of-sample until the next refit.

Theoretical References:
-----------------------
- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the
  Variance of United Kingdom Inflation. Econometrica, 50(4), 987-1007.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
  Journal of Econometrics, 31(3), 307-327.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the Relation between the
  Expected Value and the Volatility of the Nominal Excess Return on Stocks.
  The Journal of Finance, 48(5), 1779-1801.
- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach.
  Econometrica, 59(2), 347-370.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

# Compatibility patch for arch under pandas >= 3.0 vs pandas 2.x
import inspect
import pandas.util._decorators as _pud

_orig_deprecate_kwarg = _pud.deprecate_kwarg
_deprecate_params = list(inspect.signature(_orig_deprecate_kwarg).parameters.keys())


def _compat_deprecate_kwarg(*args, **kwargs):
    # In pandas >= 3.0, the first parameter is 'klass' (e.g. FutureWarning).
    # In pandas < 3.0, the first parameter is 'old_arg_name'.
    if _deprecate_params and _deprecate_params[0] == "klass":
        if len(args) >= 2 and isinstance(args[0], str):
            return _orig_deprecate_kwarg(FutureWarning, *args, **kwargs)
    return _orig_deprecate_kwarg(*args, **kwargs)


_pud.deprecate_kwarg = _compat_deprecate_kwarg

from arch import arch_model  # noqa: E402

from src.features.base import FeatureBase  # noqa: E402
from src.features.feature_registry import feature_registry  # noqa: E402
from src.features.volatility_analysis import arch_lm_test  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ============================================================
# Data Container for Model Results
# ============================================================


@dataclass
class GARCHModelResult:
    """Structured container for fitted ARCH/GARCH model parameters and diagnostics."""

    model_name: str
    params: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    persistence: float
    is_stationary: bool
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    arch_lm_stat: float
    arch_lm_pvalue: float
    remaining_arch_effects: bool
    fitted_model: Any = field(repr=False, default=None)

    def summary_dict(self) -> dict[str, Any]:
        """Return serializable summary of model properties."""
        return {
            "model_name": self.model_name,
            "aic": round(self.aic, 2),
            "bic": round(self.bic, 2),
            "log_likelihood": round(self.log_likelihood, 2),
            "persistence": round(self.persistence, 4),
            "is_stationary": self.is_stationary,
            "arch_lm_stat": round(self.arch_lm_stat, 2),
            "arch_lm_pvalue": round(self.arch_lm_pvalue, 4),
            "remaining_arch_effects": self.remaining_arch_effects,
            "params": {k: round(v, 6) for k, v in self.params.items()},
        }


# ============================================================
# Helpers
# ============================================================


def _prepare_returns_series(
    data: pd.Series | pd.DataFrame | np.ndarray,
    price_col: str = "close",
) -> pd.Series:
    """Extract clean log returns from price series or return array."""
    if isinstance(data, pd.DataFrame):
        if price_col in data.columns:
            p = data[price_col].astype(float)
            r = np.log(p / p.shift(1)).dropna()
            r.index = data.index[1:] if len(r) == len(data) - 1 else r.index
        elif "return" in data.columns:
            r = data["return"].dropna().astype(float)
        else:
            raise ValueError(f"DataFrame must contain '{price_col}' or 'return' column.")
    elif isinstance(data, pd.Series):
        # If series values look like prices (all positive, mean >> 1), compute returns
        if (data > 0).all() and data.mean() > 5.0 and data.std() > 1.0:
            r = np.log(data / data.shift(1)).dropna()
        else:
            r = data.dropna().astype(float)
    else:
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
        r = pd.Series(arr)

    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 30:
        raise ValueError(
            f"Returns series length ({len(r)}) is too short for GARCH modeling. Minimum is 30."
        )
    return r


# ============================================================
# 1. Plain GARCH(1,1) Model Fitting & Diagnostics
# ============================================================


def fit_garch(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    rescale: bool = True,
    price_col: str = "close",
) -> GARCHModelResult:
    """Fit a symmetric GARCH(p, q) model with rigorous stationarity and residual diagnostics.

    Theoretical Foundation:
    -----------------------
    The standard GARCH(1,1) process (Bollerslev 1986) models the conditional variance as:
        r_t = mu + epsilon_t,   epsilon_t = sigma_t * z_t,   z_t ~ i.i.d.(0, 1)
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    Stationarity & Persistence:
    ---------------------------
    - The unconditional (long-run) variance is:
          sigma_L^2 = omega / (1 - (alpha + beta))
    - Stationarity Requirement: alpha + beta < 1.0.
      If alpha + beta >= 1.0, the process is integrated (IGARCH if = 1) or explosive (> 1),
      meaning past volatility shocks persist permanently or compound infinitely.
    - Persistence is measured by alpha + beta. In equity markets, persistence typically
      ranges between 0.90 and 0.99.

    Residual Diagnostics:
    ---------------------
    If the GARCH model successfully captures the empirical volatility clustering identified
    in Phase 10, the standardized residuals:
        z_t = epsilon_t / sigma_t
    must be conditionally homoskedastic (i.e. Engle's ARCH-LM test on z_t should fail
    to reject the null hypothesis of no ARCH effects, p > 0.05).

    Parameters
    ----------
    returns : pd.Series | pd.DataFrame | np.ndarray
        Price series or returns.
    p : int, default 1
        Lag order of symmetric innovation (ARCH term).
    q : int, default 1
        Lag order of lagged variance (GARCH term).
    dist : str, default 'normal'
        Residual distribution ('normal', 't', 'skewt').
    rescale : bool, default True
        If True, scales returns by 100 during optimization for numerical stability,
        then rescales conditional volatility back to original scale.
    price_col : str, default 'close'
        Column name if DataFrame is passed.

    Returns
    -------
    GARCHModelResult
        Container with parameters, information criteria, conditional volatility, and diagnostics.
    """
    r_series = _prepare_returns_series(returns, price_col=price_col)

    # Scale returns by 100 for numerical optimization stability in arch library
    scale = 100.0 if rescale else 1.0
    r_scaled = r_series * scale

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        am = arch_model(r_scaled, mean="Constant", vol="Garch", p=p, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False)

    params_dict = dict(res.params)
    # Conditional volatility on original scale
    cond_vol = (res.conditional_volatility / scale).copy()
    cond_vol.index = r_series.index

    # Standardized residuals: epsilon_t / sigma_t
    std_resid = res.std_resid.copy()
    std_resid.index = r_series.index

    # Check stationarity: alpha + beta < 1
    alpha = params_dict.get("alpha[1]", 0.0)
    beta = params_dict.get("beta[1]", 0.0)
    persistence = float(alpha + beta)
    is_stationary = bool(persistence < 1.0)

    # Re-run ARCH-LM test on standardized residuals
    lm_res = arch_lm_test(std_resid.dropna(), lags=5)
    remaining_arch = lm_res.has_arch_effects(alpha=0.05)

    return GARCHModelResult(
        model_name="GARCH(1,1)",
        params=params_dict,
        log_likelihood=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        persistence=persistence,
        is_stationary=is_stationary,
        conditional_volatility=cond_vol,
        standardized_residuals=std_resid,
        arch_lm_stat=float(lm_res.lm_stat),
        arch_lm_pvalue=float(lm_res.lm_pvalue),
        remaining_arch_effects=remaining_arch,
        fitted_model=res,
    )


# ============================================================
# 2. Asymmetric Volatility Models (GJR-GARCH & EGARCH)
# ============================================================


def fit_gjr_garch(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "normal",
    rescale: bool = True,
    price_col: str = "close",
) -> GARCHModelResult:
    """Fit a GJR-GARCH(p, o, q) model capturing asymmetric leverage effects.

    Theoretical Foundation:
    -----------------------
    In equity markets, bad news (negative returns) generates significantly higher volatility
    than good news of equal magnitude. Glosten, Jagannathan, and Runkle (1993) model this via:
        sigma_t^2 = omega + (alpha + gamma * I_{epsilon_{t-1} < 0}) * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    where:
        I_{epsilon_{t-1} < 0} = 1 if epsilon_{t-1} < 0, else 0.
        gamma > 0 denotes the presence of an asymmetric "leverage effect".

    Stationarity Condition:
        alpha + beta + gamma / 2 < 1.0.
    """
    r_series = _prepare_returns_series(returns, price_col=price_col)
    scale = 100.0 if rescale else 1.0
    r_scaled = r_series * scale

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        am = arch_model(r_scaled, mean="Constant", vol="Garch", p=p, o=o, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False)

    params_dict = dict(res.params)
    cond_vol = (res.conditional_volatility / scale).copy()
    cond_vol.index = r_series.index
    std_resid = res.std_resid.copy()
    std_resid.index = r_series.index

    alpha = params_dict.get("alpha[1]", 0.0)
    gamma = params_dict.get("gamma[1]", 0.0)
    beta = params_dict.get("beta[1]", 0.0)
    persistence = float(alpha + beta + gamma / 2.0)
    is_stationary = bool(persistence < 1.0)

    lm_res = arch_lm_test(std_resid.dropna(), lags=5)
    remaining_arch = lm_res.has_arch_effects(alpha=0.05)

    return GARCHModelResult(
        model_name="GJR-GARCH(1,1,1)",
        params=params_dict,
        log_likelihood=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        persistence=persistence,
        is_stationary=is_stationary,
        conditional_volatility=cond_vol,
        standardized_residuals=std_resid,
        arch_lm_stat=float(lm_res.lm_stat),
        arch_lm_pvalue=float(lm_res.lm_pvalue),
        remaining_arch_effects=remaining_arch,
        fitted_model=res,
    )


def fit_egarch(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "normal",
    rescale: bool = True,
    price_col: str = "close",
) -> GARCHModelResult:
    """Fit Nelson's Exponential GARCH (EGARCH) model.

    Theoretical Foundation:
    -----------------------
    Developed by Daniel Nelson (1991), EGARCH models log conditional variance:
        ln(sigma_t^2) = omega + alpha * (|z_{t-1}| - E|z_{t-1}|) + gamma * z_{t-1} + beta * ln(sigma_{t-1}^2)
    where:
        gamma < 0 (or gamma > 0 depending on parameterization) captures asymmetry.
        Since ln(sigma_t^2) is modeled, sigma_t^2 is guaranteed positive without imposing
        non-negativity restrictions on parameters.
    """
    r_series = _prepare_returns_series(returns, price_col=price_col)
    scale = 100.0 if rescale else 1.0
    r_scaled = r_series * scale

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        am = arch_model(r_scaled, mean="Constant", vol="EGARCH", p=p, o=o, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False)

    params_dict = dict(res.params)
    cond_vol = (res.conditional_volatility / scale).copy()
    cond_vol.index = r_series.index
    std_resid = res.std_resid.copy()
    std_resid.index = r_series.index

    beta = params_dict.get("beta[1]", 0.0)
    persistence = float(abs(beta))
    is_stationary = bool(persistence < 1.0)

    lm_res = arch_lm_test(std_resid.dropna(), lags=5)
    remaining_arch = lm_res.has_arch_effects(alpha=0.05)

    return GARCHModelResult(
        model_name="EGARCH(1,1,1)",
        params=params_dict,
        log_likelihood=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        persistence=persistence,
        is_stationary=is_stationary,
        conditional_volatility=cond_vol,
        standardized_residuals=std_resid,
        arch_lm_stat=float(lm_res.lm_stat),
        arch_lm_pvalue=float(lm_res.lm_pvalue),
        remaining_arch_effects=remaining_arch,
        fitted_model=res,
    )


# ============================================================
# 3. Model Comparison & Selection
# ============================================================


def select_best_volatility_model(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    criterion: Literal["aic", "bic"] = "aic",
    price_col: str = "close",
) -> tuple[GARCHModelResult, pd.DataFrame]:
    """Fit GARCH(1,1), GJR-GARCH(1,1,1), and EGARCH(1,1,1), returning the best model.

    Parameters
    ----------
    returns : pd.Series | pd.DataFrame | np.ndarray
        Market data.
    criterion : 'aic' | 'bic', default 'aic'
        Information criterion for selection.
    price_col : str, default 'close'
        Column name if DataFrame is passed.

    Returns
    -------
    tuple[GARCHModelResult, pd.DataFrame]
        (best_model_result, comparison_table_dataframe)
    """
    models = {
        "GARCH(1,1)": fit_garch(returns, price_col=price_col),
        "GJR-GARCH(1,1,1)": fit_gjr_garch(returns, price_col=price_col),
        "EGARCH(1,1,1)": fit_egarch(returns, price_col=price_col),
    }

    rows = []
    for name, m in models.items():
        rows.append(
            {
                "model": name,
                "aic": m.aic,
                "bic": m.bic,
                "log_likelihood": m.log_likelihood,
                "persistence": m.persistence,
                "is_stationary": m.is_stationary,
                "arch_lm_pvalue": m.arch_lm_pvalue,
                "has_remaining_arch": m.remaining_arch_effects,
            }
        )

    df_comp = pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)
    best_name = str(df_comp.iloc[0]["model"])
    best_model = models[best_name]

    logger.info(
        "Model selection complete. Winner: %s (%s=%.2f)",
        best_name,
        criterion.upper(),
        df_comp.iloc[0][criterion],
    )
    return best_model, df_comp


# ============================================================
# 4. Volatility Forecasting
# ============================================================


def forecast_volatility(
    fitted_result: GARCHModelResult,
    horizon: int = 5,
    method: Literal["analytic", "simulation", "bootstrap"] | None = None,
) -> pd.Series:
    """Compute N-step-ahead conditional volatility forecasts from a fitted GARCH model.

    Parameters
    ----------
    fitted_result : GARCHModelResult
        Result from fit_garch, fit_gjr_garch, or fit_egarch.
    horizon : int, default 5
        Forecast horizon in periods (trading days).
    method : 'analytic' | 'simulation' | 'bootstrap' | None, default None
        Forecasting methodology. If None, automatically selects 'simulation'
        for non-linear models (such as EGARCH) when horizon > 1, and 'analytic' otherwise.

    Returns
    -------
    pd.Series
        Forecasted standard deviation (volatility) for step 1..horizon on original scale.
    """
    if fitted_result.fitted_model is None:
        raise ValueError("Model result does not contain a fitted arch model object.")

    if method is None:
        if fitted_result.model_name.startswith("EGARCH") and horizon > 1:
            method = "simulation"
        else:
            method = "analytic"

    forecasts = fitted_result.fitted_model.forecast(horizon=horizon, method=method)
    # Variance forecast is in variance space, scaled by (scale^2)
    # arch models fit on r * 100 have variance on 100^2 scale
    var_forecast = forecasts.variance.iloc[-1].values
    vol_forecast = np.sqrt(var_forecast) / 100.0

    steps = [f"h_{i+1}" for i in range(horizon)]
    return pd.Series(
        vol_forecast,
        index=steps,
        name=f"forecast_vol_{fitted_result.model_name}",
    )


# ============================================================
# 5. Anti-Leakage Rolling Out-Of-Sample Feature Generation
# ============================================================


def rolling_garch_features(
    data: pd.DataFrame | pd.Series,
    window: int = 252,
    refit_frequency: int = 20,
    model_type: Literal["garch", "gjr", "egarch"] = "garch",
    price_col: str = "close",
) -> pd.DataFrame:
    """Generate conditional volatility features using rolling out-of-sample refits (Zero Lookahead).

    Why Rolling Refits are Essential (Anti-Leakage):
    ------------------------------------------------
    Full-sample GARCH estimation fits the model over the ENTIRE time series [0..T], meaning
    parameters (omega, alpha, beta) at day 50 were optimized using volatility data from day 2000.
    In real trading, future volatility regimes (e.g. 2020 COVID crash) cannot be known in 2018.

    To eliminate lookahead bias:
    1. At rebalance date t, fit model strictly on past window [t - window, t].
    2. Use the fitted parameters to project 1-step ahead conditional volatility out-of-sample
       for the interval [t, t + refit_frequency].
    3. Advance window by refit_frequency and repeat.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Market data or price series.
    window : int, default 252
        Estimation window in trading days (e.g. 1 year).
    refit_frequency : int, default 20
        Number of bars between model parameter re-estimations (e.g. monthly).
    model_type : 'garch' | 'gjr' | 'egarch', default 'garch'
        GARCH specification to fit.
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'garch_vol' and 'garch_vol_annualized' aligned with input index.
    """
    if window < 60:
        raise ValueError(f"Rolling window ({window}) must be >= 60 for reliable GARCH fitting.")

    r_series = _prepare_returns_series(data, price_col=price_col)
    n = len(r_series)

    # Initialize output array
    out_vol = np.full(n, np.nan, dtype=float)

    # Fit functions mapping
    fit_fn = {
        "garch": fit_garch,
        "gjr": fit_gjr_garch,
        "egarch": fit_egarch,
    }.get(model_type.lower(), fit_garch)

    # Rolling refit loop
    current_model_res: GARCHModelResult | None = None

    for i in range(window, n):
        # Time to refit?
        if (i - window) % refit_frequency == 0 or current_model_res is None:
            sub_r = r_series.iloc[i - window : i]
            try:
                current_model_res = fit_fn(sub_r)
            except Exception as exc:
                logger.debug("Refit failed at index %d: %s. Reusing previous model.", i, exc)

        if current_model_res is not None:
            # 1-step ahead conditional volatility forecast for observation i
            try:
                f_vol = forecast_volatility(current_model_res, horizon=1).iloc[0]
                out_vol[i] = f_vol
            except Exception:
                # Fallback to rolling std if forecast fails
                out_vol[i] = r_series.iloc[max(0, i - 20) : i].std()

    s_vol = pd.Series(out_vol, index=r_series.index)

    # Align with original input DataFrame index if DataFrame was passed
    if isinstance(data, pd.DataFrame):
        full_vol = pd.Series(np.nan, index=data.index)
        full_vol.loc[s_vol.index] = s_vol.values
        vol_series = full_vol
    else:
        vol_series = s_vol

    ann_vol = vol_series * np.sqrt(252.0)

    return pd.DataFrame(
        {
            "garch_vol": vol_series,
            "garch_vol_annualized": ann_vol,
        },
        index=data.index if isinstance(data, pd.DataFrame) else r_series.index,
    )


# ============================================================
# 6. FeatureBase Extractor
# ============================================================


class GARCHVolatilityFeatureExtractor(FeatureBase):
    """Unified GARCH volatility extractor adhering to FeatureBase interface.

    Features Produced:
    - garch_vol: Daily conditional volatility from out-of-sample rolling GARCH(1,1).
    - garch_vol_annualized: Annualized conditional volatility (garch_vol * sqrt(252)).
    """

    def __init__(
        self,
        name: str = "garch_volatility_suite",
        window: int = 252,
        refit_frequency: int = 20,
        model_type: Literal["garch", "gjr", "egarch"] = "garch",
        price_col: str = "close",
    ) -> None:
        super().__init__(
            name=name,
            category="volatility",
            required_columns=[price_col],
        )
        self.window = window
        self.refit_frequency = refit_frequency
        self.model_type = model_type
        self.price_col = price_col

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling GARCH volatility features strictly past-only."""
        return rolling_garch_features(
            data=df,
            window=self.window,
            refit_frequency=self.refit_frequency,
            model_type=self.model_type,
            price_col=self.price_col,
        )


# ============================================================
# 7. Central Registry Registration
# ============================================================


def _register_garch_features() -> None:
    """Register GARCH features in the global feature_registry."""
    feature_registry.register(
        name="garch_vol",
        category="volatility",
        description="Daily conditional volatility from rolling out-of-sample GARCH(1,1)",
        lookback_horizon=252,
        compute_fn=lambda df: rolling_garch_features(df, window=252, refit_frequency=20)[
            ["garch_vol"]
        ],
        tags=["garch", "conditional_volatility", "clustering"],
    )
    feature_registry.register(
        name="garch_vol_annualized",
        category="volatility",
        description="Annualized conditional volatility from rolling out-of-sample GARCH(1,1)",
        lookback_horizon=252,
        compute_fn=lambda df: rolling_garch_features(df, window=252, refit_frequency=20)[
            ["garch_vol_annualized"]
        ],
        tags=["garch", "annualized", "risk_scaling"],
    )


_register_garch_features()
