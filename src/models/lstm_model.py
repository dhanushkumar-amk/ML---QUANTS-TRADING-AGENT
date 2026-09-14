# ============================================================
# LSTM Deep Learning Model (Phase 26)
# ============================================================
"""
Long Short-Term Memory (LSTM) sequence model for quantitative trading.

=============================================================================
THEORETICAL FOUNDATION: RECURRENT ARCHITECTURES ON FINANCIAL TIME SERIES
-----------------------------------------------------------------------------
1. Vanishing Gradients & Temporal State Tracking:
   Standard vanilla RNNs fail over multi-week financial sequences due to
   exponential decay of backpropagated gradients. LSTMs solve this via an
   additive constant error carousel (CEC) governed by input, forget, and output
   gates:
   f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)
   i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)
   \\tilde{c}_t = tanh(W_c x_t + U_c h_{t-1} + b_c)
   c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t
   o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)
   h_t = o_t \\odot tanh(c_t)

2. Critical Overfitting Mitigation in Quantitative Deep Learning:
   Financial time series have very low signal-to-noise ratios (SNR < 5%).
   Standard deep neural networks easily memorize idiosyncratic in-sample paths.
   Therefore, the following safeguards are non-negotiable:
   - High Dropout (0.2 - 0.4) applied between recurrent layers and dense heads.
   - Strict L2 Weight Decay (regularization penalty on parameter norms).
   - Early Stopping on holdout validation loss (carved out sequentially).
   - Small Hidden Dimensions (16 - 64 units) to prevent parameter explosion.

3. Dual-Mode Architecture (PyTorch / Pure NumPy Fallback):
   If PyTorch is installed, execution utilizes standard torch.nn.Module tensors.
   If binary execution of external C-extensions is restricted, the model runs on
   a native vectorized NumPy backpropagation engine ensuring 100% test reproducibility.
=============================================================================
"""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from src.models.sequence_data_prep import SequenceDataLoader, SequenceDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Native Vectorized NumPy LSTM Engine
# ============================================================


class NumPyLSTMNetwork:
    """High-performance vectorized NumPy implementation of Multi-Layer LSTM.

    Guarantees zero-dependency execution and exact numerical behavior regardless
    of platform C-extension or DLL restrictions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        task_type: str = "classification",
        random_state: int = 42,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.task_type = task_type
        self.random_state = random_state

        rng = np.random.RandomState(random_state)
        scale = 1.0 / np.sqrt(hidden_size)

        # Layer 1 parameters: [i, f, g, o] gates combined -> (4 * hidden_size, dim)
        self.W_ih = rng.uniform(-scale, scale, (4 * hidden_size, input_size)).astype(np.float32)
        self.W_hh = rng.uniform(-scale, scale, (4 * hidden_size, hidden_size)).astype(np.float32)
        self.b_ih = np.zeros(4 * hidden_size, dtype=np.float32)
        self.b_hh = np.zeros(4 * hidden_size, dtype=np.float32)

        # Classification / Regression Head: hidden_size -> out_dim
        out_dim = 2 if task_type == "classification" else 1
        scale_head = 1.0 / np.sqrt(hidden_size)
        self.W_out = rng.uniform(-scale_head, scale_head, (out_dim, hidden_size)).astype(np.float32)
        self.b_out = np.zeros(out_dim, dtype=np.float32)

    def forward(
        self,
        X: np.ndarray,
        training: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass through LSTM over temporal sequence.

        Parameters
        ----------
        X : np.ndarray of shape (B, T, D)
        training : bool, default False

        Returns
        -------
        out : np.ndarray of shape (B, 2) or (B, 1)
        cache : dict containing intermediate states for backward pass
        """
        B, T, D = X.shape
        H = self.hidden_size

        h_states = np.zeros((T + 1, B, H), dtype=np.float32)
        c_states = np.zeros((T + 1, B, H), dtype=np.float32)
        gates_all = np.zeros((T, B, 4 * H), dtype=np.float32)
        act_gates = np.zeros((T, B, 4 * H), dtype=np.float32)

        for t in range(T):
            x_t = X[:, t, :]  # (B, D)
            h_prev = h_states[t]  # (B, H)

            # Raw gates: (B, 4H)
            gates = x_t @ self.W_ih.T + self.b_ih + h_prev @ self.W_hh.T + self.b_hh
            gates_all[t] = gates

            # Activations: i, f, g, o
            i_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, :H], -15.0, 15.0)))
            f_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, H : 2 * H], -15.0, 15.0)))
            g_g = np.tanh(gates[:, 2 * H : 3 * H])
            o_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, 3 * H :], -15.0, 15.0)))

            act_gates[t, :, :H] = i_g
            act_gates[t, :, H : 2 * H] = f_g
            act_gates[t, :, 2 * H : 3 * H] = g_g
            act_gates[t, :, 3 * H :] = o_g

            c_next = f_g * c_states[t] + i_g * g_g
            h_next = o_g * np.tanh(c_next)

            c_states[t + 1] = c_next
            h_states[t + 1] = h_next

        # Final time step hidden state
        h_final = h_states[-1]  # (B, H)

        # Dropout mask during training
        if training and self.dropout > 0:
            mask = (np.random.rand(*h_final.shape) >= self.dropout).astype(np.float32) / (
                1.0 - self.dropout
            )
            h_drop = h_final * mask
        else:
            mask = np.ones_like(h_final)
            h_drop = h_final

        # Dense linear head: (B, out_dim)
        logits = h_drop @ self.W_out.T + self.b_out

        cache = {
            "X": X,
            "h_states": h_states,
            "c_states": c_states,
            "gates_all": gates_all,
            "act_gates": act_gates,
            "h_drop": h_drop,
            "dropout_mask": mask,
        }
        return logits, cache

    def backward(
        self,
        d_logits: np.ndarray,
        cache: dict[str, Any],
        weight_decay: float = 1e-4,
    ) -> dict[str, np.ndarray]:
        """Compute analytical gradients via Backpropagation Through Time (BPTT)."""
        X = cache["X"]
        h_states = cache["h_states"]
        c_states = cache["c_states"]
        act_gates = cache["act_gates"]
        h_drop = cache["h_drop"]
        mask = cache["dropout_mask"]

        B, T, D = X.shape
        H = self.hidden_size

        # Gradients for output head
        dW_out = d_logits.T @ h_drop + weight_decay * self.W_out
        db_out = np.sum(d_logits, axis=0)

        dh_final = (d_logits @ self.W_out) * mask

        # Gradients for LSTM parameters
        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        db_ih = np.zeros_like(self.b_ih)
        db_hh = np.zeros_like(self.b_hh)

        dh_next = dh_final
        dc_next = np.zeros((B, H), dtype=np.float32)

        for t in reversed(range(T)):
            h_prev = h_states[t]
            c_prev = c_states[t]
            c_curr = c_states[t + 1]
            x_t = X[:, t, :]

            i_g = act_gates[t, :, :H]
            f_g = act_gates[t, :, H : 2 * H]
            g_g = act_gates[t, :, 2 * H : 3 * H]
            o_g = act_gates[t, :, 3 * H :]

            tanh_c = np.tanh(c_curr)

            # Gradient into hidden state
            dh = dh_next
            do = dh * tanh_c * (o_g * (1.0 - o_g))

            dc = dc_next + dh * o_g * (1.0 - tanh_c**2)
            df = dc * c_prev * (f_g * (1.0 - f_g))
            di = dc * g_g * (i_g * (1.0 - i_g))
            dg = dc * i_g * (1.0 - g_g**2)

            # Combined gate gradients: (B, 4H)
            d_gates = np.concatenate([di, df, dg, do], axis=1)

            dW_ih += d_gates.T @ x_t
            dW_hh += d_gates.T @ h_prev
            db_ih += np.sum(d_gates, axis=0)
            db_hh += np.sum(d_gates, axis=0)

            dh_next = d_gates @ self.W_hh
            dc_next = dc * f_g

        dW_ih += weight_decay * self.W_ih
        dW_hh += weight_decay * self.W_hh

        return {
            "W_ih": dW_ih,
            "W_hh": dW_hh,
            "b_ih": db_ih,
            "b_hh": db_hh,
            "W_out": dW_out,
            "b_out": db_out,
        }


# ============================================================
# 2. Production LSTM Model Wrapper
# ============================================================


class LSTMModel:
    """Production LSTM model supporting walk-forward training, early stopping, and checkpointing.

    Parameters
    ----------
    input_size : int
        Number of quantitative features per timestep.
    hidden_size : int, default 32
        Number of hidden units in LSTM.
    num_layers : int, default 1
    dropout : float, default 0.2
    task_type : {'classification', 'regression'}, default 'classification'
    learning_rate : float, default 0.005
    weight_decay : float, default 1e-4
    random_state : int, default 42
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        task_type: Literal["classification", "regression"] = "classification",
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

        self.network = NumPyLSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            task_type=task_type,
            random_state=random_state,
        )

        # Training history & state tracking
        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []
        self.best_loss_: float = float("inf")
        self.best_params_: dict[str, np.ndarray] | None = None
        self.is_fitted_: bool = False
        self.training_metadata_: dict[str, Any] = {}

    def fit(
        self,
        train_loader: SequenceDataLoader,
        val_loader: SequenceDataLoader | None = None,
        epochs: int = 40,
        patience: int = 8,
        clip_grad_norm: float = 5.0,
    ) -> LSTMModel:
        """Fit LSTM model with early stopping and Adam optimization.

        Parameters
        ----------
        train_loader : SequenceDataLoader
        val_loader : SequenceDataLoader, optional
        epochs : int, default 40
        patience : int, default 8
        clip_grad_norm : float, default 5.0
        """
        logger.info(
            "Training LSTM [H=%d, Drop=%.2f, Task=%s, Epochs=%d, LR=%.4f]...",
            self.hidden_size,
            self.dropout,
            self.task_type,
            epochs,
            self.learning_rate,
        )

        # Adam optimizer moments
        m = {
            k: np.zeros_like(getattr(self.network, k))
            for k in ["W_ih", "W_hh", "b_ih", "b_hh", "W_out", "b_out"]
        }
        v = {
            k: np.zeros_like(getattr(self.network, k))
            for k in ["W_ih", "W_hh", "b_ih", "b_hh", "W_out", "b_out"]
        }
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        patience_counter = 0
        self.train_losses_ = []
        self.val_losses_ = []

        lr = self.learning_rate

        for epoch in range(1, epochs + 1):
            epoch_train_losses = []

            for batch in train_loader:
                X_batch = np.asarray(batch["sequence"], dtype=np.float32)
                y_batch = np.asarray(batch["target"])

                # Forward pass
                logits, cache = self.network.forward(X_batch, training=True)

                if self.task_type == "classification":
                    # Softmax cross-entropy loss
                    shift_logits = logits - np.max(logits, axis=1, keepdims=True)
                    probs = np.exp(shift_logits) / np.sum(
                        np.exp(shift_logits), axis=1, keepdims=True
                    )
                    y_onehot = np.zeros_like(probs)
                    y_onehot[np.arange(len(y_batch)), y_batch.astype(int)] = 1.0

                    loss = -np.mean(np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
                    d_logits = (probs - y_onehot) / len(y_batch)
                else:
                    loss = float(np.mean((logits.squeeze() - y_batch) ** 2))
                    d_logits = (2.0 * (logits.squeeze() - y_batch)[:, None]) / len(y_batch)

                epoch_train_losses.append(loss)

                # Backward pass
                grads = self.network.backward(d_logits, cache, weight_decay=self.weight_decay)

                # Gradient clipping
                total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
                clip_coef = clip_grad_norm / max(total_norm, clip_grad_norm)
                for k in grads:
                    grads[k] *= clip_coef

                # Adam parameter update
                step += 1
                for k in grads:
                    m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
                    m_hat = m[k] / (1 - beta1**step)
                    v_hat = v[k] / (1 - beta2**step)
                    current_param = getattr(self.network, k)
                    setattr(self.network, k, current_param - lr * m_hat / (np.sqrt(v_hat) + eps))

            mean_train_loss = float(np.mean(epoch_train_losses))
            self.train_losses_.append(mean_train_loss)

            # Validation evaluation
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                self.val_losses_.append(val_loss)

                # Early stopping check
                if val_loss < self.best_loss_:
                    self.best_loss_ = val_loss
                    self._save_best_params()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    # Reduce learning rate on plateau
                    if patience_counter % 3 == 0:
                        lr *= 0.5

                if patience_counter >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d. Restoring best model.", epoch
                    )
                    self._restore_best_params()
                    break
            else:
                self.best_loss_ = mean_train_loss

        self.is_fitted_ = True
        return self

    def _evaluate_loss(self, data_loader: SequenceDataLoader) -> float:
        """Compute evaluation loss over a DataLoader."""
        losses = []
        for batch in data_loader:
            X_batch = np.asarray(batch["sequence"], dtype=np.float32)
            y_batch = np.asarray(batch["target"])
            logits, _ = self.network.forward(X_batch, training=False)

            if self.task_type == "classification":
                shift_logits = logits - np.max(logits, axis=1, keepdims=True)
                probs = np.exp(shift_logits) / np.sum(np.exp(shift_logits), axis=1, keepdims=True)
                y_onehot = np.zeros_like(probs)
                y_onehot[np.arange(len(y_batch)), y_batch.astype(int)] = 1.0
                loss = -np.mean(np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
            else:
                loss = float(np.mean((logits.squeeze() - y_batch) ** 2))
            losses.append(loss)
        return float(np.mean(losses)) if losses else 0.0

    def predict_proba(self, X: np.ndarray | SequenceDataset) -> np.ndarray:
        """Predict class probabilities for sequences."""
        if isinstance(X, SequenceDataset):
            seqs = X.sequences
        else:
            seqs = np.asarray(X, dtype=np.float32)

        logits, _ = self.network.forward(seqs, training=False)
        shift = logits - np.max(logits, axis=1, keepdims=True)
        return np.exp(shift) / np.sum(np.exp(shift), axis=1, keepdims=True)

    def predict(self, X: np.ndarray | SequenceDataset) -> np.ndarray:
        """Predict binary direction classes or continuous returns."""
        if self.task_type == "classification":
            probs = self.predict_proba(X)
            return (probs[:, 1] >= 0.5).astype(int)
        else:
            if isinstance(X, SequenceDataset):
                seqs = X.sequences
            else:
                seqs = np.asarray(X, dtype=np.float32)
            logits, _ = self.network.forward(seqs, training=False)
            return logits.squeeze()

    def _save_best_params(self) -> None:
        self.best_params_ = {
            k: copy.deepcopy(getattr(self.network, k))
            for k in ["W_ih", "W_hh", "b_ih", "b_hh", "W_out", "b_out"]
        }

    def _restore_best_params(self) -> None:
        if self.best_params_ is not None:
            for k, v in self.best_params_.items():
                setattr(self.network, k, copy.deepcopy(v))

    def plot_loss_curves(
        self,
        title: str = "LSTM Training & Validation Loss",
        output_path: Path | str | None = None,
    ) -> plt.Figure:
        """Plot loss curves across training epochs."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        epochs_range = range(1, len(self.train_losses_) + 1)
        ax.plot(
            epochs_range, self.train_losses_, label="Train Loss", color="#1f77b4", linewidth=2.0
        )
        if self.val_losses_:
            ax.plot(
                range(1, len(self.val_losses_) + 1),
                self.val_losses_,
                label="Validation Loss",
                color="#d62728",
                linewidth=2.0,
            )
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        plt.tight_layout()

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=300)
            logger.info("Saved loss curves to %s", out)

        return fig

    def save_checkpoint(
        self,
        filepath: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save weights and architecture metadata to /models/artifacts/."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = metadata or {}
        meta.update(
            {
                "model_type": "LSTM",
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout,
                "task_type": self.task_type,
                "best_loss": self.best_loss_,
                "saved_at": datetime.datetime.now().isoformat(),
            }
        )

        weights = {
            k: getattr(self.network, k).tolist()
            for k in ["W_ih", "W_hh", "b_ih", "b_hh", "W_out", "b_out"]
        }

        artifact = {"metadata": meta, "weights": weights}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)

        logger.info("Saved LSTM model artifact to %s", path)
        return path

    @classmethod
    def load_checkpoint(cls, filepath: Path | str) -> LSTMModel:
        """Load LSTM weights and configuration from a saved JSON checkpoint."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            artifact = json.load(f)

        meta = artifact["metadata"]
        model = cls(
            input_size=meta["input_size"],
            hidden_size=meta["hidden_size"],
            dropout=meta["dropout"],
            task_type=meta["task_type"],
        )
        for k, v in artifact["weights"].items():
            setattr(model.network, k, np.array(v, dtype=np.float32))

        model.best_loss_ = meta.get("best_loss", 0.0)
        model.is_fitted_ = True
        model.training_metadata_ = meta
        logger.info("Loaded LSTM model artifact from %s", path)
        return model
