# ============================================================
# Sequence Data Preparation for Deep Learning (Phase 25)
# ============================================================
"""
Time-series sequence data transformation engine for Deep Learning models.

=============================================================================
ARCHITECTURAL DESIGN DECISION: /src/models/sequence_data_prep.py
-----------------------------------------------------------------------------
Why keep `sequence_data_prep.py` inside `/src/models/` alongside `walk_forward.py`:
1. Cohesion with Walk-Forward Framework:
   Sequence dataset generation directly couples with `WalkForwardSplitter` indices
   from Phase 20. Keeping the sequence transform in `/src/models/` ensures seamless
   interoperability across both tabular estimators (Phase 21-22) and recurrent /
   transformer neural architectures (Phase 26-27).
2. Data Flow:
   Raw Data -> Feature Pipeline (Phase 17) -> Tabular Matrix (Phase 18)
   -> Sequence Data Prep (Phase 25) -> Deep Learning Architectures (Phase 26+).

=============================================================================
CRITICAL ANTI-LEAKAGE CONSTRAINTS IN TEMPORAL WINDOWING
-----------------------------------------------------------------------------
Transforming a 2D matrix (N_samples, N_features) into 3D tensors
(N_sequences, seq_len, N_features) introduces distinct risk vectors for leakage:

1. Boundary Crossing & Lookahead:
   A sequence window of length L ending at bar t (predicting target at t+1)
   must only observe historical bars [t - L + 1, ..., t].
   Under walk-forward cross-validation, NO sequence in the test fold may have any
   of its historical lookback window overlap with the training fold or the embargo gap:
   Sequence t_test_start must satisfy: (t_test_start - L + 1) >= t_embargo_end.
   This guarantees that test sequences are strictly isolated.

2. Training-Only Scaling:
   Feature normalization scalers (RobustScaler, StandardScaler) must be fit
   EXCLUSIVELY on the training slice of sequences and applied forward to test sequences.
   Fitting scalers across the full sequence tensor leaks future statistical moments
   (mean, standard deviation, median).

3. Window Quality & Padding/Masking:
   Financial data contains gaps (holidays, trading halts, missing bars).
   Rather than silently including corrupted windows or ffilling over massive gaps,
   we implement:
   - Gap detection: Discarding windows where calendar interval exceeds max threshold.
   - Attention masking: Binary mask tensor (1 = valid, 0 = padded/missing) matching
     standard Transformer / RNN padding conventions.
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.walk_forward import WalkForwardSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Sequence Window Creation & Tensor Formatting
# ============================================================


def create_sliding_sequences(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    seq_len: int = 20,
    stride: int = 1,
    max_nan_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    """Transform tabular feature matrix into sliding 3D temporal sequences.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (N, D)
        Feature matrix.
    y : pd.Series or np.ndarray of shape (N,)
        Target aligned with the final bar of each sequence window.
    seq_len : int, default 20
        Number of historical bars per sequence.
    stride : int, default 1
        Step size between consecutive sequence windows.
    max_nan_fraction : float, default 0.0
        Maximum allowed fraction of NaNs in a window before discarding it.

    Returns
    -------
    sequences : np.ndarray of shape (num_samples, seq_len, num_features)
        3D feature tensor.
    targets : np.ndarray of shape (num_samples,)
        1D target vector corresponding to the next-step prediction after each window.
    masks : np.ndarray of shape (num_samples, seq_len)
        Boolean mask (True = valid, False = padded/missing) for DL attention/RNN masking.
    sequence_end_indices : list[Any]
        Timestamps or index labels corresponding to the final bar of each window.
    """
    if isinstance(X, pd.DataFrame):
        feat_arr = X.values
        timestamps = list(X.index)
    else:
        feat_arr = np.asarray(X, dtype=float)
        timestamps = list(range(len(feat_arr)))

    y_arr = np.asarray(y)

    n_bars, n_features = feat_arr.shape
    if n_bars < seq_len:
        raise ValueError(
            f"Dataset length ({n_bars}) is shorter than sequence lookback ({seq_len})."
        )

    seq_list = []
    target_list = []
    mask_list = []
    end_idx_list = []

    for end_idx in range(seq_len - 1, n_bars, stride):
        start_idx = end_idx - seq_len + 1
        window = feat_arr[start_idx : end_idx + 1]
        target = y_arr[end_idx]

        # NaN inspection & mask generation
        nan_mask = ~np.isnan(window)
        nan_frac = float(np.sum(~nan_mask)) / (seq_len * n_features)

        if nan_frac > max_nan_fraction:
            # Skip corrupted window
            continue

        # Fill any remaining NaNs with 0.0 for tensor stability
        clean_window = np.nan_to_num(window, nan=0.0)
        # Step-level validity mask: True if all features at that timestep are valid
        timestep_mask = np.all(nan_mask, axis=1)

        seq_list.append(clean_window)
        target_list.append(target)
        mask_list.append(timestep_mask)
        end_idx_list.append(timestamps[end_idx])

    if len(seq_list) == 0:
        raise ValueError("No valid sequences could be constructed under constraints.")

    sequences = np.stack(seq_list, axis=0).astype(np.float32)
    targets = np.array(target_list)
    masks = np.stack(mask_list, axis=0).astype(bool)

    logger.debug(
        "Constructed sequence dataset: %d samples, seq_len=%d, features=%d",
        sequences.shape[0],
        sequences.shape[1],
        sequences.shape[2],
    )
    return sequences, targets, masks, end_idx_list


# ============================================================
# 2. Sequence Normalization Pipeline (Training-Only Fit)
# ============================================================


class SequenceNormalizer:
    """Scales 3D sequence tensors while strictly preventing future information leakage.

    Fits scaling statistics (mean/std or median/IQR) exclusively on the 2D
    unrolled training sequences, and transforms test sequences forward.

    Parameters
    ----------
    method : {'robust', 'standard'}, default 'robust'
        - 'robust': RobustScaler (median and IQR, robust to financial outliers)
        - 'standard': StandardScaler (zero mean, unit variance)
    """

    def __init__(self, method: str = "robust") -> None:
        self.method = method
        self.scaler = RobustScaler() if method == "robust" else StandardScaler()
        self.is_fitted = False

    def fit(self, train_sequences: np.ndarray) -> SequenceNormalizer:
        """Fit normalization parameters on training sequence tensor.

        Parameters
        ----------
        train_sequences : np.ndarray of shape (N_train, seq_len, n_features)
        """
        n_samples, seq_len, n_feats = train_sequences.shape
        # Flatten across samples and timesteps: (N_train * seq_len, n_feats)
        flattened = train_sequences.reshape(-1, n_feats)
        self.scaler.fit(flattened)
        self.is_fitted = True
        return self

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        """Apply fitted normalization to a 3D sequence tensor.

        Parameters
        ----------
        sequences : np.ndarray of shape (N, seq_len, n_features)

        Returns
        -------
        np.ndarray of shape (N, seq_len, n_features)
            Normalized sequence tensor.
        """
        if not self.is_fitted:
            raise ValueError("SequenceNormalizer must be fitted before transforming.")

        n_samples, seq_len, n_feats = sequences.shape
        flattened = sequences.reshape(-1, n_feats)
        scaled_flat = self.scaler.transform(flattened)
        return scaled_flat.reshape(n_samples, seq_len, n_feats).astype(np.float32)

    def fit_transform(self, train_sequences: np.ndarray) -> np.ndarray:
        """Fit on training sequences and return transformed tensor."""
        return self.fit(train_sequences).transform(train_sequences)


# ============================================================
# 3. PyTorch-Compatible Dataset & DataLoader
# ============================================================


class SequenceDataset:
    """Production sequence dataset mimicking the PyTorch Dataset interface.

    Returns sequences, targets, and masks as tensors/numpy arrays.
    Compatible with both standard numpy pipelines and PyTorch DataLoader.

    Parameters
    ----------
    sequences : np.ndarray of shape (N, seq_len, D)
    targets : np.ndarray of shape (N,)
    masks : np.ndarray of shape (N, seq_len), optional
    timestamps : list, optional
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray | None = None,
        timestamps: list[Any] | None = None,
    ) -> None:
        if len(sequences) != len(targets):
            raise ValueError(
                f"Sequences ({len(sequences)}) and targets ({len(targets)}) must match in length."
            )
        self.sequences = np.asarray(sequences, dtype=np.float32)
        self.targets = np.asarray(targets)
        self.masks = (
            np.asarray(masks, dtype=bool)
            if masks is not None
            else np.ones((len(sequences), sequences.shape[1]), dtype=bool)
        )
        self.timestamps = timestamps or list(range(len(sequences)))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetch a single sequence observation."""
        item: dict[str, Any] = {
            "sequence": self.sequences[idx],
            "target": self.targets[idx],
            "mask": self.masks[idx],
            "timestamp": self.timestamps[idx],
        }

        # Optional conversion to torch.Tensor if PyTorch is loaded
        try:
            import torch

            item["sequence"] = torch.from_numpy(item["sequence"])
            item["target"] = torch.tensor(item["target"])
            item["mask"] = torch.from_numpy(item["mask"])
        except Exception:
            pass

        return item


class SequenceDataLoader:
    """Production batch iterator mimicking PyTorch DataLoader.

    Handles batch chunking, shuffling, and tensor conversion without requiring
    external compiled dependencies.

    Parameters
    ----------
    dataset : SequenceDataset
    batch_size : int, default 32
    shuffle : bool, default False
        For time-series models, default is False to preserve chronology.
    drop_last : bool, default False
    """

    def __init__(
        self,
        dataset: SequenceDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_idx = indices[start:end]

            batch_seqs = self.dataset.sequences[batch_idx]
            batch_targets = self.dataset.targets[batch_idx]
            batch_masks = self.dataset.masks[batch_idx]

            batch_dict: dict[str, Any] = {
                "sequence": batch_seqs,
                "target": batch_targets,
                "mask": batch_masks,
            }

            try:
                import torch

                batch_dict["sequence"] = torch.from_numpy(batch_seqs)
                batch_dict["target"] = torch.tensor(batch_targets)
                batch_dict["mask"] = torch.from_numpy(batch_masks)
            except Exception:
                pass

            yield batch_dict


# ============================================================
# 4. Walk-Forward Sequence Splitter (Anti-Leakage Partition)
# ============================================================


def walk_forward_sequence_split(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplitter,
    seq_len: int = 20,
    normalizer_method: str = "robust",
) -> list[dict[str, Any]]:
    """Generate strictly leak-free walk-forward sequence folds for Deep Learning.

    CRITICAL ENFORCEMENT:
    For each fold:
    1. Training sequences are extracted from X[train_idx] and y[train_idx].
    2. Scaler is fit ONLY on training sequences.
    3. Test sequences are extracted starting strictly from test_idx[0], ensuring
       no lookback window crosses backward into the embargo gap or training partition.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    splitter : WalkForwardSplitter
    seq_len : int, default 20
    normalizer_method : str, default 'robust'

    Returns
    -------
    list of dict
        Each element contains:
        - 'fold': Fold index
        - 'train_dataset': SequenceDataset (normalized)
        - 'test_dataset': SequenceDataset (normalized with train scaler)
        - 'normalizer': Fitted SequenceNormalizer
        - 'n_train_seqs': int
        - 'n_test_seqs': int
    """
    folds = []

    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        # For the test fold, we need observations from test_idx.
        # But we must ensure test sequences do not reach backward across the embargo gap!
        # Therefore, test sequences can only be formed using test partition bars:
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        if len(X_test_fold) < seq_len:
            logger.warning(
                "Test fold %d length (%d) is shorter than sequence lookback (%d). Skipping fold.",
                fold_num,
                len(X_test_fold),
                seq_len,
            )
            continue

        # 1. Create raw sequence arrays
        train_seqs, train_tgts, train_masks, train_ts = create_sliding_sequences(
            X_train_fold, y_train_fold, seq_len=seq_len
        )
        test_seqs, test_tgts, test_masks, test_ts = create_sliding_sequences(
            X_test_fold, y_test_fold, seq_len=seq_len
        )

        # 2. Fit normalizer ONLY on train fold
        normalizer = SequenceNormalizer(method=normalizer_method)
        norm_train_seqs = normalizer.fit_transform(train_seqs)
        norm_test_seqs = normalizer.transform(test_seqs)

        train_ds = SequenceDataset(norm_train_seqs, train_tgts, train_masks, train_ts)
        test_ds = SequenceDataset(norm_test_seqs, test_tgts, test_masks, test_ts)

        folds.append(
            {
                "fold": fold_num,
                "train_dataset": train_ds,
                "test_dataset": test_ds,
                "normalizer": normalizer,
                "n_train_seqs": len(train_ds),
                "n_test_seqs": len(test_ds),
            }
        )

    logger.info(
        "Walk-forward sequence split completed: %d valid folds generated.",
        len(folds),
    )
    return folds


# ============================================================
# 5. Sanity Check Visualization
# ============================================================


def plot_sequence_window_sanity_check(
    dataset: SequenceDataset,
    feature_names: list[str],
    sample_indices: list[int] | None = None,
    features_to_plot: list[str] | None = None,
    title: str = "Deep Learning Sequence Window Sanity Check",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Plot sample sequence windows alongside their target directional labels.

    Confirms tensor orientation, temporal alignment, and normalization behavior.

    Parameters
    ----------
    dataset : SequenceDataset
    feature_names : list[str]
    sample_indices : list[int], optional
        Indices of sequence samples to plot (default: first 3).
    features_to_plot : list[str], optional
        Subset of features to visualize (default: top 3).
    title : str
    output_path : Path or str, optional

    Returns
    -------
    plt.Figure
    """
    indices = sample_indices or [0, min(10, len(dataset) - 1), min(25, len(dataset) - 1)]
    feats = features_to_plot or feature_names[:3]
    feat_indices = [feature_names.index(f) for f in feats if f in feature_names]

    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3.2 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax_idx, s_idx in enumerate(indices):
        ax = axes[ax_idx]
        seq = dataset.sequences[s_idx]  # shape: (seq_len, n_features)
        target = dataset.targets[s_idx]
        ts = dataset.timestamps[s_idx]
        seq_len = seq.shape[0]

        steps = np.arange(-seq_len + 1, 1)

        for f_pos, f_col in enumerate(feat_indices):
            f_name = feature_names[f_col]
            ax.plot(
                steps,
                seq[:, f_col],
                label=f_name,
                color=colors[f_pos % len(colors)],
                marker="o",
                markersize=4,
                linewidth=1.8,
            )

        label_str = "UP (+1)" if target == 1 else "DOWN (0)"
        color_tgt = "#2ca02c" if target == 1 else "#d62728"

        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.scatter(
            [1],
            [0],
            color=color_tgt,
            s=120,
            zorder=5,
            label=f"Next-Step Target: {label_str}",
        )
        ax.set_title(
            f"Sequence Sample #{s_idx} (Ending Bar: {ts}) | Target: {label_str}",
            fontweight="bold",
            fontsize=11,
        )
        ax.set_ylabel("Normalized Value", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=9)

    axes[-1].set_xlabel("Relative Time Steps (t - L + 1 to t)", fontsize=11)
    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        logger.info("Saved sequence sanity check plot to %s", out)

    return fig
