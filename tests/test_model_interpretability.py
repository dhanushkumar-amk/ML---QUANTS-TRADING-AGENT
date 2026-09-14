# ============================================================
# Unit Tests for Model Interpretability & SHAP (Phase 24)
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.model_interpretability import (
    TreeSHAPExplainer,
    cross_reference_shap_vs_univariate,
    plot_shap_dependence,
    plot_shap_summary,
    plot_shap_waterfall,
)


@pytest.fixture
def synthetic_shap_data():
    """Generate synthetic dataset where f1 has direct positive effect."""
    np.random.seed(42)
    n = 150
    f1 = np.random.normal(0, 1, n)
    f2 = np.random.normal(0, 1, n)
    f3 = np.random.normal(0, 1, n)

    logits = 1.2 * f1 - 0.8 * f2 + 0.1 * f3
    y = (logits > 0).astype(int)

    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})
    y_series = pd.Series(y, name="target")
    return X, y_series


class TestTreeSHAPExplainer:
    """Test Tree SHAP values computation and consistency properties."""

    def test_shap_values_sum_to_margin(self, synthetic_shap_data):
        """Confirm efficiency axiom: sum of SHAP values + base value = raw output margin."""
        X, y = synthetic_shap_data
        model = GradientBoostingModel(
            backend="xgboost",
            task_type="classification",
            n_estimators=25,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)

        explainer = TreeSHAPExplainer(model)
        shap_values, base_value = explainer.compute_shap_values(X)

        assert shap_values.shape == X.shape
        assert isinstance(base_value, float)

        # Get booster raw output margins
        import xgboost as xgb

        dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
        raw_margins = model.model_.get_booster().predict(dmat, output_margin=True)

        # Sum of SHAP attributions + base value
        reconstructed_margins = shap_values.sum(axis=1) + base_value
        np.testing.assert_allclose(reconstructed_margins, raw_margins, rtol=1e-5, atol=1e-5)

    def test_feature_importance_ranking(self, synthetic_shap_data):
        """Confirm top feature according to SHAP aligns with dominant synthetic signal."""
        X, y = synthetic_shap_data
        model = GradientBoostingModel(n_estimators=30, max_depth=2, random_state=42)
        model.fit(X, y)

        explainer = TreeSHAPExplainer(model)
        ranking_df = explainer.get_feature_importance_ranking(X)

        assert "feature" in ranking_df.columns
        assert "mean_abs_shap" in ranking_df.columns
        assert len(ranking_df) == 3
        # f1 and f2 should dominate f3
        top_features = list(ranking_df["feature"].iloc[:2])
        assert "f1" in top_features
        assert "f2" in top_features

    def test_shap_interactions(self, synthetic_shap_data):
        """Confirm interaction matrix shape is (N, D, D)."""
        X, y = synthetic_shap_data
        model = GradientBoostingModel(n_estimators=15, max_depth=2, random_state=42)
        model.fit(X, y)

        explainer = TreeSHAPExplainer(model)
        interactions = explainer.compute_shap_interactions(X)
        assert interactions.shape == (len(X), 3, 3)

    def test_shap_plots_and_cross_reference(self, synthetic_shap_data, tmp_path):
        """Test summary, waterfall, dependence plots and cross-reference analysis."""
        X, y = synthetic_shap_data
        model = GradientBoostingModel(n_estimators=20, max_depth=2, random_state=42)
        model.fit(X, y)

        explainer = TreeSHAPExplainer(model)
        shap_vals, base_val = explainer.compute_shap_values(X)
        ranking_df = explainer.get_feature_importance_ranking(X)

        # 1. Summary plot
        plot_shap_summary(
            shap_vals,
            X,
            feature_names=list(X.columns),
            output_path=tmp_path / "summary.png",
        )
        assert (tmp_path / "summary.png").exists()

        # 2. Waterfall plot
        plot_shap_waterfall(
            sample_shap=shap_vals[0],
            sample_features=X.iloc[0],
            base_value=base_val,
            sample_idx=0,
            actual_label=int(y.iloc[0]),
            predicted_prob=float(model.predict_proba(X.iloc[[0]])[0, 1]),
            output_path=tmp_path / "waterfall.png",
        )
        assert (tmp_path / "waterfall.png").exists()

        # 3. Dependence plot
        plot_shap_dependence(
            feature_name="f1",
            shap_values=shap_vals,
            X=X,
            interaction_feature="f2",
            output_path=tmp_path / "dependence.png",
        )
        assert (tmp_path / "dependence.png").exists()

        # 4. Cross reference
        comp_df = cross_reference_shap_vs_univariate(ranking_df, X, y)
        assert "interaction_lift" in comp_df.columns
        assert len(comp_df) == 3
