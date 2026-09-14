# ============================================================
# Model Interpretability & Tree SHAP Engine (Phase 24)
# ============================================================
"""
Model interpretability and Tree SHAP diagnostic module for quantitative trading.

=============================================================================
THEORETICAL FOUNDATION: SHAP (SHAPLEY ADDITIVE EXPLANATIONS) IN FINANCE
-----------------------------------------------------------------------------
In quantitative finance, machine learning models are frequently criticized as
'black boxes' that cannot be audited by risk managers or investment committees.

SHAP grounds feature attribution in cooperative game theory (Lloyd Shapley, 1953):
Each feature value of an instance acts as a 'player' in a coalition, and the
prediction is the payout. The Shapley value is the unique attribution method
guaranteed to satisfy four foundational efficiency axioms:
1. Efficiency: Sum of attributions equals the difference between model prediction
   and the expected baseline prediction: sum(phi_i) = f(x) - E[f(x)].
2. Symmetry: Identical contributors receive identical attributions.
3. Dummy/Null Player: Features with zero marginal contribution receive zero attribution.
4. Additivity: Feature attributions across ensembles sum linearly.

In this module, Tree SHAP is computed natively via the Lundberg & Lee (2018)
polynomial-time tree algorithm embedded within XGBoost's C++ core (`pred_contribs=True`),
bypassing external Python DLL compilation bottlenecks and providing:
- Exact local and global feature attribution
- Non-linear interaction analysis (`pred_interactions=True`)
- Force / waterfall breakdown of highest-conviction winning vs. losing trades
- Cross-referencing against Phase 18 univariate filters (Correlation & Mutual Information)
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.gradient_boosting_model import GradientBoostingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Tree SHAP Explainer Wrapper
# ============================================================


class TreeSHAPExplainer:
    """Production Tree SHAP explanation engine for gradient boosted models.

    Computes exact Shapley values and feature interaction matrices using
    the native tree traversal algorithm.

    Parameters
    ----------
    model : GradientBoostingModel
        Fitted GradientBoostingModel instance.
    """

    def __init__(self, model: GradientBoostingModel) -> None:
        if not model.is_fitted_ or model.model_ is None:
            raise ValueError("Model must be fitted before initializing TreeSHAPExplainer.")
        self.model_wrapper = model
        self.feature_names = model.feature_names_
        self.booster = model.model_.get_booster()

    def compute_shap_values(
        self,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, float]:
        """Compute exact Tree SHAP values and base value for input dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix matching model's feature names.

        Returns
        -------
        shap_values : np.ndarray of shape (N, D)
            Local Shapley values for each sample and feature.
        base_value : float
            Expected baseline model output in margin space: E[f(x)].
        """
        dmat = xgb.DMatrix(X[self.feature_names].values, feature_names=self.feature_names)
        contribs = self.booster.predict(dmat, pred_contribs=True)
        # contribs: shape (N, D + 1), where the final column is the bias/base value
        shap_values = contribs[:, :-1]
        base_value = float(contribs[0, -1])
        return shap_values, base_value

    def compute_shap_interactions(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Compute pairwise Tree SHAP interaction values matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (N, D, D)
            Interaction tensors across all samples.
        """
        dmat = xgb.DMatrix(X[self.feature_names].values, feature_names=self.feature_names)
        # contribs: shape (N, D + 1, D + 1)
        interactions = self.booster.predict(dmat, pred_interactions=True)
        return interactions[:, :-1, :-1]

    def get_feature_importance_ranking(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute mean absolute SHAP feature importance ranking.

        Parameters
        ----------
        X : pd.DataFrame
            Evaluation dataset.

        Returns
        -------
        pd.DataFrame
            Ranking table sorted by mean absolute SHAP value.
        """
        shap_values, _ = self.compute_shap_values(X)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        ranking_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        total = ranking_df["mean_abs_shap"].sum()
        ranking_df["importance_share"] = ranking_df["mean_abs_shap"] / total if total > 0 else 0.0
        return ranking_df.reset_index(drop=True)


# ============================================================
# 2. SHAP Visualizations (Global & Local)
# ============================================================


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    max_features: int = 15,
    title: str = "SHAP Feature Importance Summary",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Generate global SHAP summary plot (custom beeswarm / violin scatter).

    Displays feature ranking by mean absolute SHAP value with colored scatter
    points indicating whether high or low feature values drove positive vs.
    negative directional predictions.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (N, D).
    X : pd.DataFrame
        Raw or scaled feature matrix of shape (N, D).
    feature_names : list[str]
    max_features : int, default 15
    title : str
    output_path : Path or str, optional

    Returns
    -------
    plt.Figure
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_features]
    top_indices = top_indices[::-1]  # reverse for ascending y-axis

    fig, ax = plt.subplots(figsize=(10, max(5, int(len(top_indices) * 0.45))))

    y_positions = np.arange(len(top_indices))

    for y_idx, f_idx in enumerate(top_indices):
        f_name = feature_names[f_idx]
        vals = shap_values[:, f_idx]
        feat_raw = X[f_name].values

        # Normalize feature values to [0, 1] for colormap
        f_min, f_max = np.nanmin(feat_raw), np.nanmax(feat_raw)
        if f_max > f_min:
            norm_vals = (feat_raw - f_min) / (f_max - f_min)
        else:
            norm_vals = np.full_like(feat_raw, 0.5)

        # Add slight jitter to y_position for beeswarm clarity
        jitter = np.random.normal(0, 0.08, size=len(vals))
        scatter = ax.scatter(
            vals,
            y_positions[y_idx] + jitter,
            c=norm_vals,
            cmap="coolwarm",
            alpha=0.6,
            s=16,
            edgecolors="none",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=10, fontweight="bold")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("SHAP Value (Impact on Model Log-Odds Margin)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.02, shrink=0.7)
    cbar.set_label("Feature Value (Low → High)", fontsize=9)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low", "High"])

    plt.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        logger.info("Saved SHAP summary plot to %s", out)

    return fig


def plot_shap_waterfall(
    sample_shap: np.ndarray,
    sample_features: pd.Series,
    base_value: float,
    sample_idx: Any,
    actual_label: int,
    predicted_prob: float,
    max_display: int = 10,
    title: str | None = None,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Generate local waterfall plot explaining a single trade prediction.

    Parameters
    ----------
    sample_shap : np.ndarray
        1D vector of SHAP values for this specific observation.
    sample_features : pd.Series
        Feature values for this observation.
    base_value : float
        Model expected baseline margin.
    sample_idx : Any
        Identifier or timestamp for this trade.
    actual_label : int
        Actual ground truth class (1 for UP, 0 for DOWN).
    predicted_prob : float
        Model predicted probability of UP.
    max_display : int, default 10
    title : str, optional
    output_path : Path or str, optional

    Returns
    -------
    plt.Figure
    """
    feat_names = list(sample_features.index)
    order = np.argsort(np.abs(sample_shap))[::-1][:max_display]
    order = order[::-1]  # ascending for horizontal bars

    fig, ax = plt.subplots(figsize=(10, max(5, int(len(order) * 0.45))))

    running_sum = base_value
    y_positions = np.arange(len(order))

    for idx, f_idx in enumerate(order):
        feat_names[f_idx]
        val = sample_shap[f_idx]
        sample_features.iloc[f_idx]
        color = "#2ca02c" if val >= 0 else "#d62728"

        ax.barh(
            y_positions[idx],
            val,
            left=running_sum,
            color=color,
            edgecolor="black",
            height=0.6,
            alpha=0.85,
        )
        running_sum += val

    labels = [f"{feat_names[i]} = {sample_features.iloc[i]:.2f}" for i in order]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(base_value, color="gray", linestyle=":", label=f"Base Value ({base_value:.2f})")
    ax.axvline(
        running_sum,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Final Margin ({running_sum:.2f})",
    )

    t = (
        title
        or f"SHAP Local Explanation: Bar {sample_idx} (Actual={actual_label}, Pred P(UP)={predicted_prob:.1%})"
    )
    ax.set_title(t, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Cumulative Model Output (Margin Space)", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        logger.info("Saved SHAP waterfall plot to %s", out)

    return fig


def plot_shap_dependence(
    feature_name: str,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    interaction_feature: str | None = None,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Generate partial dependence scatter plot checking for non-linear relationships.

    Parameters
    ----------
    feature_name : str
    shap_values : np.ndarray
    X : pd.DataFrame
    interaction_feature : str, optional
    output_path : Path or str, optional

    Returns
    -------
    plt.Figure
    """
    f_idx = list(X.columns).index(feature_name)
    feat_x = X[feature_name].values
    shap_y = shap_values[:, f_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    if interaction_feature and interaction_feature in X.columns:
        color_data = X[interaction_feature].values
        scatter = ax.scatter(
            feat_x, shap_y, c=color_data, cmap="viridis", alpha=0.7, edgecolors="none"
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(interaction_feature, fontsize=10)
    else:
        ax.scatter(feat_x, shap_y, color="#1f77b4", alpha=0.6, edgecolors="none")

    ax.axhline(0.0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel(feature_name, fontsize=11, fontweight="bold")
    ax.set_ylabel(f"SHAP Value for {feature_name}", fontsize=11, fontweight="bold")
    ax.set_title(
        f"SHAP Dependence: {feature_name}"
        + (f" (Colored by {interaction_feature})" if interaction_feature else ""),
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        logger.info("Saved SHAP dependence plot to %s", out)

    return fig


# ============================================================
# 3. Cross-Reference with Univariate Feature Selection (Phase 18)
# ============================================================


def cross_reference_shap_vs_univariate(
    shap_ranking_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Compare SHAP non-linear tree feature ranking against Phase 18 univariate metrics.

    Parameters
    ----------
    shap_ranking_df : pd.DataFrame
        Table with columns 'feature', 'mean_abs_shap'.
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    comparison_df : pd.DataFrame
        Table comparing:
        - SHAP Rank
        - Correlation Rank
        - Mutual Information Rank
        - Absolute Rank Divergence (identifies features valuable only in non-linear combinations)
    """
    from sklearn.feature_selection import mutual_info_classif

    clean_idx = X.dropna().index.intersection(y.dropna().index)
    X_clean = X.loc[clean_idx]
    y_clean = y.loc[clean_idx]

    corrs = {col: abs(float(np.corrcoef(X_clean[col], y_clean)[0, 1])) for col in X_clean.columns}
    mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
    mis = {col: float(score) for col, score in zip(X_clean.columns, mi_scores, strict=False)}

    df = shap_ranking_df.copy()
    df["pearson_corr_abs"] = df["feature"].map(corrs)
    df["mutual_info"] = df["feature"].map(mis)

    df["shap_rank"] = np.arange(1, len(df) + 1)
    df["corr_rank"] = df["pearson_corr_abs"].rank(ascending=False, method="dense").astype(int)
    df["mi_rank"] = df["mutual_info"].rank(ascending=False, method="dense").astype(int)

    # Divergence: If SHAP rank is much higher than univariate rank, tree found non-linear interaction
    df["interaction_lift"] = df["corr_rank"] - df["shap_rank"]

    logger.info("Completed SHAP vs. Univariate cross-reference comparison.")
    return df
