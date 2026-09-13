# ============================================================
# Rolling GARCH Anti-Leakage & Lookahead Verification (Phase 14)
# ============================================================
"""
Explicit automated test confirming that rolling_garch_features at time T
is strictly unaffected by perturbing or corrupting future data (t > T).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.volatility_models import rolling_garch_features


def test_rolling_garch_no_lookahead():
    """Verify that future price/volatility shocks do NOT leak into historical GARCH features."""
    np.random.seed(42)
    n = 150
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({"close": prices}, index=dates)

    cutoff_idx = 110
    window = 70
    refit_freq = 20

    base_features = rolling_garch_features(
        df,
        window=window,
        refit_frequency=refit_freq,
        price_col="close",
    )

    # Severely perturb all future prices with 1000x shock
    df_corrupted = df.copy()
    df_corrupted.iloc[cutoff_idx + 1 :, df.columns.get_loc("close")] *= 1000.0

    perturbed_features = rolling_garch_features(
        df_corrupted,
        window=window,
        refit_frequency=refit_freq,
        price_col="close",
    )

    # For all rows up to and including cutoff_idx, values must be strictly identical
    for col in base_features.columns:
        s_base = base_features[col].iloc[: cutoff_idx + 1]
        s_pert = perturbed_features[col].iloc[: cutoff_idx + 1]
        np.testing.assert_allclose(
            s_base.values,
            s_pert.values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"Lookahead detected in rolling_garch_features for column {col}!",
        )
