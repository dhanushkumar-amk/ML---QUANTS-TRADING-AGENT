# ============================================================
# Unit Tests for LSTM Model (Phase 26)
# ============================================================
from __future__ import annotations

import numpy as np
import pytest

from src.models.lstm_model import LSTMModel, NumPyLSTMNetwork
from src.models.sequence_data_prep import SequenceDataLoader, SequenceDataset


@pytest.fixture
def synthetic_dl_data():
    """Create small synthetic sequences for fast CI testing."""
    np.random.seed(42)
    N, T, D = 40, 10, 4
    X = np.random.randn(N, T, D).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)

    train_ds = SequenceDataset(X[:25], y[:25])
    val_ds = SequenceDataset(X[25:], y[25:])
    train_loader = SequenceDataLoader(train_ds, batch_size=8, shuffle=False)
    val_loader = SequenceDataLoader(val_ds, batch_size=8, shuffle=False)
    return X, y, train_loader, val_loader


class TestLSTMModel:
    """Test LSTM forward, backward, training loop, and checkpointing."""

    def test_lstm_forward_shape(self, synthetic_dl_data):
        """Confirm LSTM forward pass produces correct logits shape."""
        X, y, _, _ = synthetic_dl_data
        net = NumPyLSTMNetwork(input_size=4, hidden_size=16, task_type="classification")
        logits, cache = net.forward(X[:5], training=False)

        assert logits.shape == (5, 2)
        assert "h_states" in cache
        assert "c_states" in cache

    def test_lstm_training_loop_and_early_stopping(self, synthetic_dl_data):
        """Confirm training loop runs without error and tracks loss history."""
        _, _, train_loader, val_loader = synthetic_dl_data

        model = LSTMModel(
            input_size=4,
            hidden_size=16,
            dropout=0.1,
            task_type="classification",
            learning_rate=0.01,
            random_state=42,
        )

        model.fit(train_loader, val_loader=val_loader, epochs=4, patience=2)

        assert model.is_fitted_
        assert len(model.train_losses_) >= 2
        assert len(model.val_losses_) == len(model.train_losses_)

        # Test inference
        preds = model.predict(val_loader.dataset.sequences)
        probs = model.predict_proba(val_loader.dataset.sequences)

        assert len(preds) == len(val_loader.dataset)
        assert probs.shape == (len(val_loader.dataset), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_lstm_checkpoint_roundtrip(self, synthetic_dl_data, tmp_path):
        """Confirm checkpoint save and load preserves parameters and predictions."""
        _, _, train_loader, val_loader = synthetic_dl_data

        model = LSTMModel(input_size=4, hidden_size=8, task_type="classification")
        model.fit(train_loader, epochs=2)

        orig_preds = model.predict_proba(val_loader.dataset.sequences)

        save_path = tmp_path / "lstm_checkpoint.json"
        model.save_checkpoint(save_path, metadata={"test": "ok"})
        assert save_path.exists()

        loaded_model = LSTMModel.load_checkpoint(save_path)
        assert loaded_model.is_fitted_
        assert loaded_model.training_metadata_["test"] == "ok"

        loaded_preds = loaded_model.predict_proba(val_loader.dataset.sequences)
        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5, atol=1e-5)
