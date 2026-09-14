# ============================================================
# Unit Tests for Transformer Model (Phase 27)
# ============================================================
from __future__ import annotations

import numpy as np
import pytest

from src.models.sequence_data_prep import SequenceDataLoader, SequenceDataset
from src.models.transformer_model import (
    NumPyTransformerNetwork,
    TransformerModel,
    plot_attention_weights,
)


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


class TestTransformerModel:
    """Test Transformer forward pass, attention extraction, training loop, and checkpointing."""

    def test_transformer_forward_and_attention_shape(self, synthetic_dl_data):
        """Confirm Transformer forward pass and attention weights have exact shapes."""
        X, y, _, _ = synthetic_dl_data
        net = NumPyTransformerNetwork(
            input_size=4, d_model=16, n_heads=2, d_ff=32, task_type="classification"
        )
        logits, cache = net.forward(X[:5], training=False)

        assert logits.shape == (5, 2)
        assert "attn_weights" in cache
        assert cache["attn_weights"].shape == (5, 2, 10, 10)  # (B, n_heads, T, T)

    def test_transformer_training_and_inference(self, synthetic_dl_data):
        """Confirm training loop runs without error and makes predictions."""
        _, _, train_loader, val_loader = synthetic_dl_data

        model = TransformerModel(
            input_size=4,
            d_model=16,
            n_heads=2,
            d_ff=32,
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

        # Test attention weights extraction
        attn_matrix = model.get_attention_weights(val_loader.dataset.sequences)
        assert attn_matrix.shape == (len(val_loader.dataset), 2, 10, 10)

    def test_transformer_checkpoint_and_plotting(self, synthetic_dl_data, tmp_path):
        """Confirm checkpoint save/load and attention heatmap plotting."""
        _, _, train_loader, val_loader = synthetic_dl_data

        model = TransformerModel(input_size=4, d_model=16, n_heads=2, task_type="classification")
        model.fit(train_loader, epochs=2)

        orig_probs = model.predict_proba(val_loader.dataset.sequences)

        # Test Checkpoint Save/Load
        ckpt_path = tmp_path / "transformer_ckpt.json"
        model.save_checkpoint(ckpt_path, metadata={"test": "ok"})
        assert ckpt_path.exists()

        loaded_model = TransformerModel.load_checkpoint(ckpt_path)
        assert loaded_model.is_fitted_
        loaded_probs = loaded_model.predict_proba(val_loader.dataset.sequences)
        np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5, atol=1e-5)

        # Test Attention Weight Plotting
        sample_seq = val_loader.dataset.sequences[0]
        plot_path = tmp_path / "attention_test.png"
        plot_attention_weights(model, sample_seq, head_idx=0, output_path=plot_path)
        assert plot_path.exists()
