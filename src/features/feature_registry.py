# ============================================================
# Central Feature Registry (Phase 12)
# ============================================================
"""
Central feature registry tracking all engineered quantitative features by name,
category, lookback horizon, and required columns.

Enables programmatic feature selection, modular pipeline construction,
and auditing in downstream ML modeling (Phases 18+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata container for an engineered quantitative feature."""

    name: str
    category: str
    description: str
    lookback_horizon: int
    required_columns: list[str] = field(default_factory=lambda: ["close"])
    compute_fn: Callable[..., pd.DataFrame] | None = None
    tags: list[str] = field(default_factory=list)
    status: Literal["production", "experimental", "excluded"] = "production"
    exclusion_reason: str = ""


class FeatureRegistry:
    """Central registry tracking all available quantitative features."""

    def __init__(self) -> None:
        self._registry: dict[str, FeatureMetadata] = {}

    def register(
        self,
        name: str,
        category: str,
        description: str,
        lookback_horizon: int,
        required_columns: Sequence[str] | None = None,
        compute_fn: Callable[..., pd.DataFrame] | None = None,
        tags: Sequence[str] | None = None,
        status: Literal["production", "experimental", "excluded"] = "production",
        exclusion_reason: str = "",
    ) -> FeatureMetadata:
        """Register a feature specification in the registry.

        Parameters
        ----------
        name : str
            Unique feature identifier (e.g., 'mom_20d', 'rsi_14').
        category : str
            Feature category ('momentum', 'mean_reversion', 'volatility', etc.).
        description : str
            Detailed explanation of economic or statistical intuition.
        lookback_horizon : int
            Lookback window in periods required to compute this feature.
        required_columns : Sequence[str] | None
            Required market data columns.
        compute_fn : Callable | None
            Function taking DataFrame and returning DataFrame with this feature.
        tags : Sequence[str] | None
            Optional search/grouping tags.
        status : {'production', 'experimental', 'excluded'}, default 'production'
        exclusion_reason : str, default ''
            Documented rationale if the feature is excluded from production models.

        Returns
        -------
        FeatureMetadata
            Registered metadata object.
        """
        meta = FeatureMetadata(
            name=name,
            category=category,
            description=description,
            lookback_horizon=lookback_horizon,
            required_columns=list(required_columns or ["close"]),
            compute_fn=compute_fn,
            tags=list(tags or []),
            status=status,
            exclusion_reason=exclusion_reason,
        )
        self._registry[name] = meta
        logger.debug("Registered feature: %s [%s] (status: %s)", name, category, status)
        return meta

    def mark_production(self, name: str) -> None:
        """Mark a registered feature as vetted for production."""
        meta = self.get(name)
        meta.status = "production"
        meta.exclusion_reason = ""
        logger.info("Marked feature '%s' as production.", name)

    def mark_excluded(self, name: str, reason: str) -> None:
        """Mark a feature as excluded from production with documented rationale."""
        meta = self.get(name)
        meta.status = "excluded"
        meta.exclusion_reason = reason
        logger.info("Marked feature '%s' as excluded: %s", name, reason)

    def list_production_features(self) -> list[str]:
        """Return list of feature names vetted for production models."""
        return sorted(
            [name for name, meta in self._registry.items() if meta.status == "production"]
        )

    def list_excluded_features(self) -> dict[str, str]:
        """Return mapping of excluded feature names to their documented exclusion reasons."""
        return {
            name: meta.exclusion_reason
            for name, meta in self._registry.items()
            if meta.status == "excluded"
        }

    def get(self, name: str) -> FeatureMetadata:
        """Retrieve metadata for a registered feature.

        Raises
        ------
        KeyError
            If feature name is not found.
        """
        if name not in self._registry:
            raise KeyError(
                f"Feature '{name}' not found in registry. Registered: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def contains(self, name: str) -> bool:
        """Check if feature name is registered."""
        return name in self._registry

    def list_features(
        self,
        category: str | None = None,
        tag: str | None = None,
        status: str | None = None,
    ) -> list[str]:
        """List registered feature names matching optional category, tag, and status filters."""
        names = []
        for name, meta in self._registry.items():
            if category and meta.category.lower() != category.lower():
                continue
            if tag and tag.lower() not in [t.lower() for t in meta.tags]:
                continue
            if status and meta.status.lower() != status.lower():
                continue
            names.append(name)
        return sorted(names)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all registered features and their metadata as a pandas DataFrame."""
        rows = [
            {
                "name": m.name,
                "category": m.category,
                "status": m.status,
                "exclusion_reason": m.exclusion_reason,
                "lookback_horizon": m.lookback_horizon,
                "required_columns": ", ".join(m.required_columns),
                "description": m.description,
                "tags": ", ".join(m.tags),
                "has_compute_fn": m.compute_fn is not None,
            }
            for m in self._registry.values()
        ]
        return pd.DataFrame(rows)

    def compute_features(
        self,
        df: pd.DataFrame,
        feature_names: Sequence[str] | None = None,
        category: str | None = None,
    ) -> pd.DataFrame:
        """Compute multiple registered features on an input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Market data.
        feature_names : Sequence[str] | None
            Specific feature names to compute. If None, computes all matching category.
        category : str | None
            If specified, filters features by category.

        Returns
        -------
        pd.DataFrame
            DataFrame containing computed feature columns aligned with input index.
        """
        target_names = (
            list(feature_names) if feature_names else self.list_features(category=category)
        )
        if not target_names:
            raise ValueError("No matching features found to compute.")

        result_cols: dict[str, Any] = {}
        for name in target_names:
            meta = self.get(name)
            if meta.compute_fn is None:
                logger.warning("Feature '%s' has no compute_fn defined, skipping.", name)
                continue
            feat_df = meta.compute_fn(df)
            for c in feat_df.columns:
                result_cols[c] = feat_df[c].values

        return pd.DataFrame(result_cols, index=df.index)

    def clear(self) -> None:
        """Clear all registered features (useful for test isolation)."""
        self._registry.clear()


# Global default feature registry singleton
feature_registry = FeatureRegistry()


def register_feature(
    name: str,
    category: str,
    description: str,
    lookback_horizon: int,
    required_columns: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    registry: FeatureRegistry | None = None,
) -> Callable:
    """Decorator to register a feature calculation function.

    Example
    -------
    @register_feature(
        name="mom_20d",
        category="momentum",
        description="20-day simple price momentum",
        lookback_horizon=20,
    )
    def compute_mom_20d(df: pd.DataFrame) -> pd.DataFrame:
        ...
    """
    target_reg = registry or feature_registry

    def decorator(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        target_reg.register(
            name=name,
            category=category,
            description=description,
            lookback_horizon=lookback_horizon,
            required_columns=required_columns,
            compute_fn=fn,
            tags=tags,
        )
        return fn

    return decorator
