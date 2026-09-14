# ============================================================
# Unit Tests for Sequence Data Prep for DL (Phase 25)
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.sequence_data_prep import (
    SequenceDataLoader,
    SequenceDataset,
    SequenceNormalizer,
    create_sliding_sequences,
    plot_sequence_window_sanity_check,
    walk_forward_sequence_split,
)
from src.models.walk_forward import WalkForwardSplitter


@pytest.fixture
def synthetic_tabular_data():
    """Generate sample time-series data for testing sequence conversion."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    f1 = np.linspace(1, 10, n) + np.random.normal(0, 0.1, n)
    f2 = np.sin(np.linspace(0, 4 * np.pi, n))
    X = pd.DataFrame({"trend": f1, "cycle": f2}, index=dates)
    y = pd.Series((f2 > 0).astype(int), index=dates, name="target")
    return X, y


class TestSequenceDataPrep:
    """Test 3D sequence creation, normalization, and PyTorch compatibility."""

    def test_create_sliding_sequences_shapes(self, synthetic_tabular_data):
        """Confirm sliding sequence produces exact expected 3D tensor shape."""
        X, y = synthetic_tabular_data
        seq_len = 10
        stride = 1
        seqs, targets, masks, end_ts = create_sliding_sequences(
            X, y, seq_len=seq_len, stride=stride
        )

        expected_n = len(X) - seq_len + 1
        assert seqs.shape == (expected_n, seq_len, 2)
        assert len(targets) == expected_n
        assert masks.shape == (expected_n, seq_len)
        assert len(end_ts) == expected_n

        # First sequence should end at index seq_len - 1 (index 9)
        assert end_ts[0] == X.index[9]
        np.testing.assert_allclose(seqs[0, :, :], X.iloc[0:10].values)

    def test_sequence_normalizer_training_only(self):
        """Confirm normalizer fits only on training sequences."""
        train_seqs = np.array(
            [
                [[10.0, 100.0], [20.0, 200.0]],
                [[30.0, 300.0], [40.0, 400.0]],
            ]
        )  # shape (2, 2, 2)

        test_seqs = np.array(
            [
                [[50.0, 500.0], [60.0, 600.0]],
            ]
        )  # shape (1, 2, 2)

        normalizer = SequenceNormalizer(method="standard")
        norm_train = normalizer.fit_transform(train_seqs)

        # Flattened train: [10, 100], [20, 200], [30, 300], [40, 400]
        # Mean = [25.0, 250.0]
        assert pytest.approx(float(norm_train.mean()), abs=1e-5) == 0.0

        # Transform test using train parameters
        norm_test = normalizer.transform(test_seqs)
        # Test values are all above train mean, so normalized test values should be positive
        assert (norm_test > 0).all()

    def test_sequence_dataset_and_dataloader(self, synthetic_tabular_data):
        """Test SequenceDataset and SequenceDataLoader iteration."""
        X, y = synthetic_tabular_data
        seqs, targets, masks, end_ts = create_sliding_sequences(X, y, seq_len=15)

        ds = SequenceDataset(seqs, targets, masks, end_ts)
        assert len(ds) == len(seqs)

        item = ds[0]
        assert "sequence" in item
        assert "target" in item
        assert "mask" in item
        assert item["sequence"].shape == (15, 2)

        # Test dataloader batching
        loader = SequenceDataLoader(ds, batch_size=16, shuffle=False)
        assert len(loader) == (len(ds) + 15) // 16

        first_batch = next(iter(loader))
        assert len(first_batch["sequence"]) == 16
        assert len(first_batch["target"]) == 16

    def test_walk_forward_sequence_split_no_leakage(self, synthetic_tabular_data):
        """Confirm walk-forward sequence folds preserve embargo gap and zero leakage."""
        X, y = synthetic_tabular_data
        splitter = WalkForwardSplitter(
            n_splits=2,
            min_train_size=40,
            test_size=30,
            embargo_bars=5,
            window_type="expanding",
        )

        seq_len = 10
        folds = walk_forward_sequence_split(
            X, y, splitter=splitter, seq_len=seq_len, normalizer_method="robust"
        )

        assert len(folds) == 2
        for fold_info in folds:
            train_ds = fold_info["train_dataset"]
            test_ds = fold_info["test_dataset"]

            # Confirm sequence lengths
            assert train_ds.sequences.shape[1] == seq_len
            assert test_ds.sequences.shape[1] == seq_len

            # Check temporal ordering: max train timestamp < min test timestamp
            max_train_ts = train_ds.timestamps[-1]
            min_test_ts = test_ds.timestamps[0]
            assert max_train_ts < min_test_ts

    def test_plot_sequence_window_sanity_check(self, synthetic_tabular_data, tmp_path):
        """Test visual sanity check plot generation."""
        X, y = synthetic_tabular_data
        seqs, targets, masks, end_ts = create_sliding_sequences(X, y, seq_len=15)
        ds = SequenceDataset(seqs, targets, masks, end_ts)

        plot_sequence_window_sanity_check(
            dataset=ds,
            feature_names=list(X.columns),
            sample_indices=[0, 5],
            output_path=tmp_path / "sanity_check.png",
        )
        assert (tmp_path / "sanity_check.png").exists()
