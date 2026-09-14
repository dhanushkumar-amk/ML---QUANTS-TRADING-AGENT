# ============================================================
# Feature Selection & Target Engineering Module (Phase 18)
# ============================================================
"""
Unified feature selection, target engineering, and collinearity diagnostics.

=============================================================================
CRITICAL METHODOLOGICAL NOTICE: ANTI-LEAKAGE & WALK-FORWARD PRINCIPLE
-----------------------------------------------------------------------------
Feature selection (correlation, mutual information, tree importances, VIF)
MUST be performed strictly on TRAINING-PERIOD data. Running feature selection
on the entire dataset (or including out-of-sample validation/test partitions)
leaks future price movements and distribution statistics into feature choice,
leading to severe selection bias and non-reproducible backtest results.
=============================================================================
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Prediction Target Engineering (Strict No-Lookahead)
# ============================================================


def make_target(
    df: pd.DataFrame,
    horizon: int = 1,
    task_type: Literal["classification", "regression"] = "classification",
    price_col: str = "close",
) -> pd.Series:
    """Create forward-looking prediction target with strict no-lookahead validation.

    Parameters
    ----------
    df : pd.DataFrame
        Market price history DataFrame containing `price_col`.
    horizon : int, default 1
        Prediction horizon in bars (e.g., 1 for next-day, 5 for next 5-day).
        Must be >= 1.
    task_type : {'classification', 'regression'}, default 'classification'
        - 'classification': 1.0 if forward return > 0, else 0.0 (binary direction).
        - 'regression': forward percentage return (P_{t+h} - P_t) / P_t.
    price_col : str, default 'close'
        Price column used to calculate forward returns.

    Returns
    -------
    pd.Series
        Target series aligned with `df.index`.
        The final `horizon` rows are strictly `NaN` because future outcomes are
        unobserved at the historical series boundary.

    Raises
    ------
    ValueError
        If horizon < 1, df is empty, or price_col is missing.
    """
    if horizon < 1:
        raise ValueError(f"Target horizon must be >= 1, got {horizon}.")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

    prices = df[price_col].astype(float)

    # Negative shift looks strictly forward in time:
    # index t receives price from index t + horizon
    future_prices = prices.shift(-horizon)

    # Forward relative return: R_{t, t+h} = (P_{t+h} - P_t) / P_t
    fwd_returns = (future_prices - prices) / prices

    # Strict No-Lookahead Assertions:
    # 1. The last `horizon` rows must be NaN (future unobserved)
    if not fwd_returns.iloc[-horizon:].isna().all():
        raise RuntimeError(
            f"Lookahead validation failed: last {horizon} rows of target are not NaN."
        )

    # 2. Row t must not equal row t return (no same-day identity)
    if len(prices) > horizon:
        # Check that row 0 target matches actual return from price 0 to price horizon
        expected_ret_0 = (prices.iloc[horizon] - prices.iloc[0]) / prices.iloc[0]
        actual_ret_0 = fwd_returns.iloc[0]
        if not np.isclose(expected_ret_0, actual_ret_0, rtol=1e-8, atol=1e-8):
            raise RuntimeError("Lookahead validation failed: forward return alignment corrupted.")

    if task_type == "regression":
        target = fwd_returns.copy()
        target.name = f"target_ret_{horizon}d"
    elif task_type == "classification":
        # Binary direction: 1.0 for positive return, 0.0 for zero or negative return.
        # Preserve NaNs in unobserved tail.
        target = pd.Series(np.nan, index=fwd_returns.index, name=f"target_dir_{horizon}d")
        valid_mask = fwd_returns.notna()
        target[valid_mask] = (fwd_returns[valid_mask] > 0.0).astype(float)
    else:
        raise ValueError(
            f"Invalid task_type: '{task_type}'. Must be 'classification' or 'regression'."
        )

    return target


# ============================================================
# 2. Univariate Feature Filtering
# ============================================================


def compute_univariate_metrics(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    task_type: Literal["classification", "regression"] = "classification",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute univariate correlation and mutual information against target.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Scaled or unscaled feature matrix.
    target : pd.Series
        Prediction target series.
    task_type : {'classification', 'regression'}, default 'classification'
        Task type for mutual information estimation.
    random_state : int, default 42
        Seed for mutual information estimation.

    Returns
    -------
    pd.DataFrame
        Table with index=features and columns:
        ['pearson_corr', 'spearman_corr', 'abs_spearman_corr', 'mutual_info'].
    """
    # Align by common index and drop NaNs
    common_idx = feature_matrix.dropna().index.intersection(target.dropna().index)
    if len(common_idx) < 30:
        raise ValueError(f"Insufficient aligned observations ({len(common_idx)}) for metrics.")

    X = feature_matrix.loc[common_idx]
    y = target.loc[common_idx]

    results: list[dict[str, float | str]] = []

    # Mutual information
    if task_type == "classification":
        mi_scores = mutual_info_classif(
            X.values, y.values.astype(int), discrete_features=False, random_state=random_state
        )
    else:
        mi_scores = mutual_info_regression(
            X.values, y.values, discrete_features=False, random_state=random_state
        )

    for i, col in enumerate(X.columns):
        s_feat = X[col]
        # Pearson linear correlation
        p_corr = float(s_feat.corr(y, method="pearson"))
        # Spearman rank monotonic correlation
        sp_corr = float(s_feat.corr(y, method="spearman"))

        results.append(
            {
                "feature": col,
                "pearson_corr": p_corr if not np.isnan(p_corr) else 0.0,
                "spearman_corr": sp_corr if not np.isnan(sp_corr) else 0.0,
                "abs_spearman_corr": abs(sp_corr) if not np.isnan(sp_corr) else 0.0,
                "mutual_info": float(mi_scores[i]),
            }
        )

    res_df = pd.DataFrame(results).set_index("feature")
    return res_df


# ============================================================
# 3. Multicollinearity & Correlation Clustering
# ============================================================


def compute_multicollinearity(
    feature_matrix: pd.DataFrame,
    corr_threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """Compute Variance Inflation Factor (VIF) and identify redundant collinear pairs.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Clean feature matrix (no NaNs or Infs).
    corr_threshold : float, default 0.85
        Absolute correlation threshold above which two features are flagged as redundant.

    Returns
    -------
    vif_df : pd.DataFrame
        DataFrame indexed by feature with column 'vif'.
    collinear_pairs : list of (feature_1, feature_2, corr)
        List of highly correlated feature pairs (|r| >= corr_threshold).
    """
    clean_X = feature_matrix.dropna().copy()
    if clean_X.empty:
        raise ValueError("Feature matrix is empty after dropping NaNs.")

    # Drop any zero-variance / constant features
    stds = clean_X.std(ddof=0)
    valid_cols = stds[stds > 1e-8].index.tolist()
    if len(valid_cols) == 0:
        raise ValueError("All features have zero variance.")

    clean_X = clean_X[valid_cols]

    # Pairwise correlation matrix
    corr_mat = clean_X.corr().abs()
    collinear_pairs: list[tuple[str, str, float]] = []

    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            c1, c2 = valid_cols[i], valid_cols[j]
            r = float(corr_mat.loc[c1, c2])
            if r >= corr_threshold:
                collinear_pairs.append((c1, c2, r))

    # Sort collinear pairs by highest correlation descending
    collinear_pairs.sort(key=lambda x: x[2], reverse=True)

    # Compute Variance Inflation Factor (VIF)
    # Add constant for standard uncentered VIF
    import warnings

    try:
        X_with_const = sm.add_constant(clean_X, has_constant="add")
        vif_data = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, col in enumerate(clean_X.columns):
                # i + 1 because column 0 is the added constant
                try:
                    val = variance_inflation_factor(X_with_const.values, i + 1)
                except Exception:
                    val = np.inf
                vif_data.append({"feature": col, "vif": float(val)})
        vif_df = pd.DataFrame(vif_data).set_index("feature")
    except Exception as exc:
        logger.warning("VIF calculation failed: %s; falling back to inf.", exc)
        vif_df = pd.DataFrame({"vif": [np.inf] * len(valid_cols)}, index=valid_cols)

    return vif_df, collinear_pairs


# ============================================================
# 4. Embedded Selection (Baseline Tree Feature Importance)
# ============================================================


def compute_tree_importance(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    task_type: Literal["classification", "regression"] = "classification",
    max_depth: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """Compute tree-based Gini/MDI feature importance strictly for ranking.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Features aligned with target.
    target : pd.Series
        Target labels or returns.
    task_type : {'classification', 'regression'}, default 'classification'
    max_depth : int, default 5
        Constrained tree depth to prevent overfitting noisy financial data.
    random_state : int, default 42

    Returns
    -------
    pd.Series
        Feature importances summing to 1.0, indexed by feature name.
    """
    common_idx = feature_matrix.dropna().index.intersection(target.dropna().index)
    X = feature_matrix.loc[common_idx]
    y = target.loc[common_idx]

    if task_type == "classification":
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X.values, y.values.astype(int))
    else:
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X.values, y.values)

    importances = pd.Series(model.feature_importances_, index=X.columns, name="tree_importance")
    return importances


# ============================================================
# 5. Full Feature Selection Report & Recommended Shortlist
# ============================================================


def feature_selection_report(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    task_type: Literal["classification", "regression"] = "classification",
    top_k: int = 15,
    max_vif: float = 10.0,
    corr_threshold: float = 0.85,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Produce comprehensive ranked feature table and recommended non-collinear shortlist.

    NOTE: THIS FUNCTION MUST ONLY BE CALLED ON TRAINING-PERIOD DATA TO PREVENT
    FORWARD LOOKAHEAD LEAKAGE.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Training feature matrix.
    target : pd.Series
        Training target series.
    task_type : {'classification', 'regression'}, default 'classification'
    top_k : int, default 15
        Maximum number of shortlisted features to recommend.
    max_vif : float, default 10.0
        Threshold above which features are flagged for severe multicollinearity.
    corr_threshold : float, default 0.85
        Pairwise correlation threshold to filter redundant features.
    random_state : int, default 42

    Returns
    -------
    report_df : pd.DataFrame
        Ranked feature table containing:
        ['spearman_corr', 'mutual_info', 'tree_importance', 'vif', 'composite_rank', 'selected']
    shortlist : list[str]
        List of recommended feature names, pruned of collinearity.
    """
    logger.info("Generating feature selection report on %d features...", feature_matrix.shape[1])

    # 1. Univariate metrics
    uni_df = compute_univariate_metrics(
        feature_matrix, target, task_type=task_type, random_state=random_state
    )

    # 2. Multicollinearity (VIF + Collinear Pairs)
    vif_df, collinear_pairs = compute_multicollinearity(
        feature_matrix, corr_threshold=corr_threshold
    )

    # 3. Tree importance
    tree_s = compute_tree_importance(
        feature_matrix, target, task_type=task_type, random_state=random_state
    )

    # Assemble comprehensive report
    report_df = pd.DataFrame(index=feature_matrix.columns)
    report_df["spearman_corr"] = uni_df["spearman_corr"]
    report_df["abs_spearman"] = uni_df["abs_spearman_corr"]
    report_df["mutual_info"] = uni_df["mutual_info"]
    report_df["tree_importance"] = tree_s
    report_df["vif"] = vif_df["vif"].reindex(report_df.index).fillna(np.inf)

    # 4. Composite ranking
    # Normalize each metric to percentile rank [0, 1]
    rank_mi = report_df["mutual_info"].rank(pct=True)
    rank_sp = report_df["abs_spearman"].rank(pct=True)
    rank_tr = report_df["tree_importance"].rank(pct=True)

    # Composite score: equal weight of MI, Monotonic Correlation, and Tree Importance
    report_df["composite_score"] = (rank_mi + rank_sp + rank_tr) / 3.0
    report_df = report_df.sort_values("composite_score", ascending=False)
    report_df["composite_rank"] = range(1, len(report_df) + 1)

    # 5. Greedy non-collinear shortlist selection
    # Iteratively select highest-ranked features while skipping those
    # collinear with already selected features (|r| >= corr_threshold).
    clean_X = feature_matrix.dropna()
    corr_mat = clean_X.corr().abs()

    shortlist: list[str] = []
    dropped_reasons: dict[str, str] = {}

    for feat in report_df.index:
        if len(shortlist) >= top_k:
            break

        # Check pairwise correlation with already selected features
        is_collinear = False
        redundant_with = None
        for sel_feat in shortlist:
            if sel_feat in corr_mat.index and feat in corr_mat.columns:
                r_val = corr_mat.loc[sel_feat, feat]
                if r_val >= corr_threshold:
                    is_collinear = True
                    redundant_with = (sel_feat, r_val)
                    break

        if is_collinear:
            dropped_reasons[feat] = (
                f"Redundant with '{redundant_with[0]}' (|r|={redundant_with[1]:.2f})"
            )
            continue

        shortlist.append(feat)

    report_df["selected"] = report_df.index.isin(shortlist)
    report_df["prune_reason"] = [
        dropped_reasons.get(f, "" if f in shortlist else "Rank below top_k cutoff")
        for f in report_df.index
    ]

    logger.info("Feature selection complete: %d features shortlisted.", len(shortlist))
    return report_df, shortlist
