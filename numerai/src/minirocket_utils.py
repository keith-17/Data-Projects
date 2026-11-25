"""minirocket_utils.py — MiniRocketNumeraiEstimator child class of BaseNumeraiEstimator.

Implements an OOM-proof, strictly causal MiniRocket regressor (Dempster et al., 2021) for
Numerai Classic v6.0 ("Unity") panel data, supporting:
  - Per-ticker causal multivariate sequences (`input_mode='ticker_sequence'`) with a leak-free
    training tail buffer for validation warm-up, plus cross-sectional `input_mode='feature_profile'`.
  - Vectorized pure-NumPy Dempster-2021 MiniRocket (84 fixed length-9 {-1, +2} kernels,
    exponentially spaced dilations, quantile-sampled biases, PPV pooling) with optional `sktime`
    backend (`backend='sktime'` / `'auto'`).
  - Single-pass batched analytical Ridge / RidgeCV (GCV) and batched SGD linear heads so large
    Numerai datasets (`medium` / `all`) never materialize an `(N, num_kernels)` matrix in RAM.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import SGDRegressor

from baseline_utils import BaseNumeraiEstimator
from utils import NumeraiExtractions, _decode_structured

try:
    from sktime.transformations.panel.rocket import MiniRocket as SktimeMiniRocket
    from sktime.transformations.panel.rocket import MiniRocketMultivariate as SktimeMiniRocketMultivariate

    SKTIME_MINIROCKET_AVAILABLE = True
except ImportError:
    SktimeMiniRocket = None  # type: ignore[assignment]
    SktimeMiniRocketMultivariate = None  # type: ignore[assignment]
    SKTIME_MINIROCKET_AVAILABLE = False


# Fixed 84 length-9 MiniRocket kernel index triplets (3 weights of +2, 6 weights of -1; sum = 0)
_MINIROCKET_KERNEL_TRIPLETS: np.ndarray = np.asarray(list(combinations(range(9), 3)), dtype=np.int64)
_NUM_BASE_KERNELS: int = len(_MINIROCKET_KERNEL_TRIPLETS)  # 84
_DEFAULT_RIDGE_CV_ALPHAS: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def _compute_dilations_and_counts(
    seq_len: int,
    num_kernels: int,
    max_dilations_per_kernel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exponentially spaced MiniRocket dilations and per-dilation feature counts."""
    n_features_per_kernel = max(1, int(num_kernels) // _NUM_BASE_KERNELS)
    true_max_dilations = min(max(1, int(max_dilations_per_kernel)), n_features_per_kernel)
    max_exponent = np.log2(max(1.0, (int(seq_len) - 1) / 8.0))
    raw_dilations = np.floor(np.logspace(0.0, max_exponent, num=true_max_dilations, base=2.0)).astype(np.int64)
    dilations = np.unique(np.maximum(1, raw_dilations))

    n_dilations = len(dilations)
    base_count = n_features_per_kernel // n_dilations
    remainder = n_features_per_kernel % n_dilations
    num_features_per_dilation = np.full(n_dilations, base_count, dtype=np.int64)
    num_features_per_dilation[:remainder] += 1
    return dilations, num_features_per_dilation


class _NumpyMiniRocketTransformer:
    """Vectorized pure-NumPy implementation of Dempster et al. (2021) MiniRocket / MiniRocketMultivariate."""

    def __init__(
        self,
        num_kernels: int = 840,
        max_dilations_per_kernel: int = 8,
        random_state: int = 42,
    ):
        self.num_kernels = max(_NUM_BASE_KERNELS, (int(num_kernels) // _NUM_BASE_KERNELS) * _NUM_BASE_KERNELS)
        self.max_dilations_per_kernel = max(1, int(max_dilations_per_kernel))
        self.random_state = int(random_state)

    def _convolve_batch_for_dilation(
        self,
        X_batch: np.ndarray,
        dilation: int,
        channel_combos: Sequence[np.ndarray],
        kernel_indices: np.ndarray,
        padding_flags: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute convolutions for one dilation over a batch `(B, C, L)`.

        Returns `(conv_out, valid_lengths)` where `conv_out` has shape `(B, n_k, out_len)` and
        `valid_lengths` has shape `(n_k,)` giving the exact number of valid convolution positions
        per kernel (preventing unpadded convolutions from biasing quantiles/PPV with dummy zeros).
        """
        b_size, _, seq_len = X_batch.shape
        span = 8 * dilation + 1
        pad = 4 * dilation
        if span > seq_len:
            pad_needed = max(pad, (span - seq_len + 1) // 2)
        else:
            pad_needed = pad

        X_padded = np.pad(X_batch, ((0, 0), (0, 0), (pad_needed, pad_needed)), mode="constant", constant_values=0.0)
        n_k = len(kernel_indices)
        out_len = seq_len if pad_needed == pad else (X_padded.shape[2] - span + 1)
        conv_out = np.empty((b_size, n_k, out_len), dtype=np.float32)
        valid_lengths = np.empty(n_k, dtype=np.int64)

        for idx in range(n_k):
            k_id = int(kernel_indices[idx])
            ch = channel_combos[idx]
            use_pad = bool(padding_flags[idx])
            if len(ch) == 1:
                series = X_padded[:, int(ch[0]), :] if use_pad else X_batch[:, int(ch[0]), :]
            else:
                series = X_padded[:, ch, :].sum(axis=1) if use_pad else X_batch[:, ch, :].sum(axis=1)

            cur_len = series.shape[1]
            if cur_len < span:
                extra = span - cur_len
                left = extra // 2
                series = np.pad(series, ((0, 0), (left, extra - left)), mode="constant", constant_values=0.0)
                cur_len = series.shape[1]

            # Extract 9 strided slices of length `n_pos`
            n_pos = cur_len - span + 1
            # 2 * sum(pos3) - 1 * sum(other6) = 3 * sum(pos3) - sum(all9)
            i0, i1, i2 = _MINIROCKET_KERNEL_TRIPLETS[k_id]
            s_all = np.zeros((b_size, n_pos), dtype=np.float32)
            for p in range(9):
                offset = p * dilation
                s_all += series[:, offset : offset + n_pos]
            s_pos = (
                series[:, i0 * dilation : i0 * dilation + n_pos]
                + series[:, i1 * dilation : i1 * dilation + n_pos]
                + series[:, i2 * dilation : i2 * dilation + n_pos]
            )
            res = 3.0 * s_pos - s_all
            if n_pos >= out_len:
                conv_out[:, idx, :] = res[:, :out_len]
                valid_lengths[idx] = out_len
            else:
                conv_out[:, idx, :n_pos] = res
                conv_out[:, idx, n_pos:] = np.nan
                valid_lengths[idx] = n_pos
        return conv_out, valid_lengths

    def fit(self, X_sample: np.ndarray) -> "_NumpyMiniRocketTransformer":
        X_arr = np.asarray(X_sample, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(f"Expected 3D tensor (N, C, L), got shape {X_arr.shape}.")
        n_samples, n_channels, seq_len = X_arr.shape
        rng = np.random.default_rng(self.random_state)

        dilations, num_features_per_dilation = _compute_dilations_and_counts(
            seq_len=seq_len,
            num_kernels=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
        )
        self.dilations_ = dilations
        self.num_features_per_dilation_ = num_features_per_dilation
        self.n_channels_ = int(n_channels)
        self.seq_len_ = int(seq_len)

        # Build per-dilation specifications (channel subsets, padding flags, and quantile biases)
        self.dilation_specs_: List[Dict[str, Any]] = []
        self.feature_meta_: List[Dict[str, Any]] = []

        max_ch_per_combo = min(n_channels, 9)
        kernel_ids = np.arange(_NUM_BASE_KERNELS, dtype=np.int64)
        total_features = 0

        for d_idx, (dilation, n_feat_d) in enumerate(zip(dilations, num_features_per_dilation)):
            # Sample channel combinations per base kernel (for univariate C=1, always [0])
            channel_combos: List[np.ndarray] = []
            for _ in range(_NUM_BASE_KERNELS):
                if n_channels == 1:
                    channel_combos.append(np.array([0], dtype=np.int64))
                else:
                    # MiniRocketMultivariate draws channel count from 2^U(0, log2(min(C, 9)))
                    k_ch = int(np.floor(2.0 ** rng.uniform(0.0, np.log2(max_ch_per_combo + 1e-9))))
                    k_ch = max(1, min(n_channels, k_ch))
                    channel_combos.append(np.sort(rng.choice(n_channels, size=k_ch, replace=False)).astype(np.int64))

            padding_flags = ((kernel_ids + d_idx) % 2 == 0)

            # Sample one random training example per kernel to draw quantile biases (exact MiniRocket rule)
            sample_indices = rng.integers(0, n_samples, size=_NUM_BASE_KERNELS)
            # Low-discrepancy quantile grid in (0, 1)
            quantiles = np.linspace(1.0 / (n_feat_d + 1), n_feat_d / (n_feat_d + 1), num=int(n_feat_d))

            # Compute convolutions on the sampled examples in one batch
            unique_sample_idx, inv_idx = np.unique(sample_indices, return_inverse=True)
            conv_unique, valid_lengths = self._convolve_batch_for_dilation(
                X_arr[unique_sample_idx],
                dilation=int(dilation),
                channel_combos=channel_combos,
                kernel_indices=kernel_ids,
                padding_flags=padding_flags,
            )  # (U, 84, L_out), (84,)

            biases = np.empty((_NUM_BASE_KERNELS, int(n_feat_d)), dtype=np.float32)
            for k_id in range(_NUM_BASE_KERNELS):
                v_len = int(valid_lengths[k_id])
                c_vec = conv_unique[inv_idx[k_id], k_id, :v_len]
                biases[k_id, :] = np.quantile(c_vec, quantiles).astype(np.float32)
                for b_idx in range(int(n_feat_d)):
                    self.feature_meta_.append({
                        "feature_idx": total_features,
                        "kernel_id": int(k_id),
                        "dilation": int(dilation),
                        "bias": float(biases[k_id, b_idx]),
                        "channels": tuple(int(c) for c in channel_combos[k_id]),
                    })
                    total_features += 1

            self.dilation_specs_.append({
                "dilation": int(dilation),
                "channel_combos": channel_combos,
                "kernel_indices": kernel_ids,
                "padding_flags": padding_flags,
                "biases": biases,
            })

        self.n_output_features_ = int(total_features)
        return self

    def transform(self, X_batch: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_batch, dtype=np.float32)
        b_size = X_arr.shape[0]
        out = np.empty((b_size, self.n_output_features_), dtype=np.float32)
        col_offset = 0
        for spec in self.dilation_specs_:
            conv, valid_lengths = self._convolve_batch_for_dilation(
                X_arr,
                dilation=spec["dilation"],
                channel_combos=spec["channel_combos"],
                kernel_indices=spec["kernel_indices"],
                padding_flags=spec["padding_flags"],
            )  # (B, 84, L_out), (84,)
            biases = spec["biases"]  # (84, n_feat_d)
            n_feat_d = biases.shape[1]
            # Compute PPV strictly over valid positions [:v_len] and emit columns in exact (k_id, b_idx) order
            # matching `self.feature_meta_` (fixes both unpadded zero-fill bias and column transposition!)
            ppv_all = np.empty((b_size, _NUM_BASE_KERNELS, n_feat_d), dtype=np.float32)
            for v_len in np.unique(valid_lengths):
                v_int = int(v_len)
                k_mask = valid_lengths == v_len
                c_sub = conv[:, k_mask, :v_int]  # (B, K_sub, v_int)
                b_sub = biases[k_mask, :]  # (K_sub, n_feat_d)
                # Broadcast (B, K_sub, 1, v_int) > (1, K_sub, n_feat_d, 1) -> mean over v_int
                ppv_all[:, k_mask, :] = (c_sub[:, :, None, :] > b_sub[None, :, :, None]).mean(axis=3, dtype=np.float32)
            width = _NUM_BASE_KERNELS * n_feat_d
            out[:, col_offset : col_offset + width] = ppv_all.reshape(b_size, width)
            col_offset += width
        return out


class MiniRocketNumeraiEstimator(BaseNumeraiEstimator):
    """Child Class — Causal MiniRocket Regressor inheriting `BaseNumeraiEstimator`.

    Builds strictly causal per-ticker multivariate sequences (with a training tail buffer for
    validation warm-up) or cross-sectional feature profiles, transforms them in OOM-safe batches
    via MiniRocket PPV features, and fits an analytical Ridge / RidgeCV (or batched SGD) head.
    """

    def __init__(
        self,
        extractor: Optional[NumeraiExtractions] = None,
        # BaseNumeraiEstimator shared parameters
        top_k: Optional[int] = None,
        correlation_method: Optional[str] = "pandas_mean",
        abs_corr: bool = True,
        era_weighting: Optional[str] = None,
        recency_decay: Optional[float] = 0.005,
        prediction_transform: Optional[str] = None,
        bias_correction: Optional[str] = None,
        walk_forward_step_eras: Optional[int] = None,
        walk_forward_window: str = "expanding",
        target_horizon_eras: int = 4,
        era_stride: int = 1,
        era_col: str = "era",
        target_col: str = "target",
        verbose: int = 0,
        random_state: int = 42,
        # MiniRocket sequence & transform parameters
        ticker_col: str = "numerai_ticker",
        input_mode: str = "ticker_sequence",
        seq_len: int = 12,
        max_channels: int = 16,
        pad_mode: str = "first",
        num_kernels: int = 840,
        max_dilations_per_kernel: int = 8,
        fit_sample_size: int = 2048,
        backend: str = "numpy",
        # Linear head parameters
        standardize_features: bool = True,
        head_type: str = "ridge_cv",
        alpha: float = 100.0,
        l1_ratio: float = 0.15,
        huber_epsilon: float = 1.35,
        batch_size: int = 16384,
        n_jobs: int = -1,
    ):
        super().__init__(
            extractor=extractor,
            top_k=top_k,
            correlation_method=correlation_method,
            abs_corr=abs_corr,
            era_weighting=era_weighting,
            recency_decay=recency_decay,
            prediction_transform=prediction_transform,
            bias_correction=bias_correction,
            walk_forward_step_eras=walk_forward_step_eras,
            walk_forward_window=walk_forward_window,
            target_horizon_eras=target_horizon_eras,
            era_stride=era_stride,
            era_col=era_col,
            target_col=target_col,
            verbose=verbose,
            random_state=random_state,
        )
        self.ticker_col = ticker_col
        self.input_mode = input_mode
        self.seq_len = seq_len
        self.max_channels = max_channels
        self.pad_mode = pad_mode
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.fit_sample_size = fit_sample_size
        self.backend = backend
        self.standardize_features = standardize_features
        self.head_type = head_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.huber_epsilon = huber_epsilon
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def _extract_tickers(self, X: Any, n_rows: int) -> Optional[np.ndarray]:
        if isinstance(X, pd.DataFrame) and self.ticker_col in X.columns:
            return X[self.ticker_col].astype(str).to_numpy()
        return None

    def _resolve_input_mode(self, tickers: Optional[np.ndarray]) -> str:
        mode = str(_decode_structured(self.input_mode) or "ticker_sequence").strip().lower()
        if mode not in ("ticker_sequence", "feature_profile"):
            raise ValueError(f"Unsupported input_mode '{self.input_mode}'; use 'ticker_sequence' or 'feature_profile'.")
        if mode == "ticker_sequence" and tickers is None:
            return "feature_profile"
        return mode

    def _build_sequence_tensor(
        self,
        feature_matrix: np.ndarray,
        era_values: Optional[np.ndarray],
        tickers: Optional[np.ndarray],
        *,
        mode: str,
        is_fit: bool,
    ) -> np.ndarray:
        """Construct 3D float32 tensor `(N, C, L)` strictly causally via vectorized NumPy indexing."""
        n_rows, n_cols = feature_matrix.shape
        clean_mat = np.nan_to_num(
            np.asarray(feature_matrix, dtype=np.float32),
            nan=2.0,
            posinf=4.0,
            neginf=0.0,
        )

        if mode == "feature_profile":
            # Each row's feature vector becomes a univariate sequence of length >= 9
            if n_cols < 9:
                pad_w = 9 - n_cols
                clean_mat = np.pad(clean_mat, ((0, 0), (0, pad_w)), mode="constant", constant_values=2.0)
            if is_fit:
                self.ticker_tail_buffer_ = {}
            return clean_mat[:, None, :]

        # mode == "ticker_sequence": causal per-ticker rolling window of length `seq_len`
        seq_len = max(9, int(self.seq_len))
        pad_mode = str(_decode_structured(self.pad_mode) or "first").strip().lower()
        eras_arr = (
            np.asarray(era_values, dtype=np.int64)
            if era_values is not None
            else np.arange(n_rows, dtype=np.int64)
        )
        tickers_arr = np.asarray(tickers, dtype=object)

        # Sort by (ticker, era) stably via lexsort (last key is primary)
        order = np.lexsort((eras_arr, tickers_arr))
        sorted_tickers = tickers_arr[order]
        sorted_eras = eras_arr[order]
        sorted_feats = clean_mat[order]

        # Detect physical era spacing during fit so denser validation eras (e.g. step=1 vs train stride=4)
        # sample the exact same physical lookback horizon without 4x time-scale compression
        if is_fit:
            uniq_fit_eras = np.unique(sorted_eras)
            if len(uniq_fit_eras) >= 2:
                e_diffs = np.diff(uniq_fit_eras)
                pos_diffs = e_diffs[e_diffs > 0]
                self.train_era_step_ = max(1, int(round(float(np.median(pos_diffs))))) if len(pos_diffs) else 1
            else:
                self.train_era_step_ = 1
        train_step = max(1, int(getattr(self, "train_era_step_", 1)))

        # Detect whether prediction query eras are denser than training era spacing
        if not is_fit:
            uniq_q_eras = np.unique(sorted_eras)
            if len(uniq_q_eras) >= 2:
                q_diffs = np.diff(uniq_q_eras)
                pos_q = q_diffs[q_diffs > 0]
                query_step = max(1, int(round(float(np.median(pos_q))))) if len(pos_q) else 1
            else:
                query_step = 1
            need_era_resampling = (train_step > 1) and (query_step < train_step)
        else:
            need_era_resampling = False

        # Find contiguous ticker boundaries
        change_mask = np.empty(n_rows, dtype=bool)
        change_mask[0] = True
        if n_rows > 1:
            change_mask[1:] = sorted_tickers[1:] != sorted_tickers[:-1]
        starts = np.flatnonzero(change_mask)
        ends = np.append(starts[1:], n_rows)

        out_tensor = np.empty((n_rows, n_cols, seq_len), dtype=np.float32)
        tail_buffer: Dict[str, Tuple[np.ndarray, np.ndarray]] = {} if is_fit else getattr(self, "ticker_tail_buffer_", {})
        lag_offsets = (np.arange(seq_len - 1, -1, -1, dtype=np.int64)) * int(train_step)

        for s, e in zip(starts, ends):
            t_key = str(sorted_tickers[s])
            grp_eras = sorted_eras[s:e]  # (T,)
            grp_feats = sorted_feats[s:e]  # (T, C)
            orig_positions = order[s:e]
            t_len = grp_feats.shape[0]

            hist_eras: Optional[np.ndarray] = None
            hist_feats: Optional[np.ndarray] = None
            if not is_fit and t_key in tail_buffer:
                raw_entry = tail_buffer[t_key]
                if isinstance(raw_entry, tuple) and len(raw_entry) == 2:
                    h_e, h_f = raw_entry
                    # Strictly causal: only prepend historical eras strictly earlier than the first query era
                    c_mask = h_e < int(grp_eras[0])
                    if bool(np.any(c_mask)):
                        hist_eras = h_e[c_mask]
                        hist_feats = h_f[c_mask]
                else:
                    hist_feats = np.asarray(raw_entry, dtype=np.float32)

            if need_era_resampling:
                # Era-aligned causal lookback: for each query era e_t, fetch observations at
                # [e_t - (seq_len-1)*train_step, ..., e_t - train_step, e_t] via strictly causal searchsorted
                if hist_feats is not None and hist_eras is not None:
                    comb_eras = np.concatenate([hist_eras, grp_eras])
                    comb_feats = np.vstack([hist_feats, grp_feats])
                else:
                    comb_eras = grp_eras
                    comb_feats = grp_feats

                target_eras = grp_eras[:, None] - lag_offsets[None, :]  # (T, seq_len)
                pos_idx = np.searchsorted(comb_eras, target_eras, side="right") - 1  # (T, seq_len)
                missing_mask = pos_idx < 0
                safe_idx = np.maximum(pos_idx, 0)
                gathered = comb_feats[safe_idx]  # (T, seq_len, C)
                if bool(np.any(missing_mask)):
                    if pad_mode == "neutral":
                        gathered[missing_mask] = 2.0
                    else:
                        first_val = comb_feats[0]  # (C,)
                        gathered[missing_mask] = first_val
                # Transpose (T, seq_len, C) -> (T, C, seq_len)
                out_tensor[orig_positions] = np.transpose(gathered, (0, 2, 1))
            else:
                if hist_feats is not None:
                    combined = np.vstack([hist_feats, grp_feats])
                    need_pad = max(0, (seq_len - 1) - hist_feats.shape[0])
                else:
                    combined = grp_feats
                    need_pad = seq_len - 1

                if need_pad > 0:
                    if pad_mode == "neutral":
                        pad_block = np.full((need_pad, n_cols), 2.0, dtype=np.float32)
                    else:
                        pad_block = np.repeat(combined[:1, :], need_pad, axis=0)
                    padded = np.vstack([pad_block, combined])
                else:
                    padded = combined

                # sliding_window_view over axis=0 -> (T, C, seq_len)
                windows = sliding_window_view(padded, window_shape=seq_len, axis=0)[-t_len:]
                out_tensor[orig_positions] = windows

            if is_fit:
                # Retain enough timestamped trailing history for full stride-aligned validation warm-up
                keep_tail = min(max(seq_len - 1, (seq_len - 1) + 2), t_len)
                if keep_tail > 0:
                    tail_buffer[t_key] = (grp_eras[-keep_tail:].copy(), grp_feats[-keep_tail:].copy())

        if is_fit:
            self.ticker_tail_buffer_ = tail_buffer
        return out_tensor

    def _build_rocket_transformer(self, n_channels: int) -> Tuple[Any, str]:
        req_backend = str(_decode_structured(self.backend) or "numpy").strip().lower()
        if req_backend not in ("numpy", "sktime", "auto"):
            raise ValueError(f"Unsupported backend '{self.backend}'; use 'numpy', 'sktime', or 'auto'.")
        use_sktime = (req_backend == "sktime") or (req_backend == "auto" and SKTIME_MINIROCKET_AVAILABLE)
        if use_sktime:
            if not SKTIME_MINIROCKET_AVAILABLE:
                raise ImportError("sktime is not installed; use backend='numpy' or install sktime.")
            n_kernels = max(_NUM_BASE_KERNELS, int(self.num_kernels))
            max_dil = max(1, int(self.max_dilations_per_kernel))
            if n_channels > 1:
                return (
                    SktimeMiniRocketMultivariate(
                        num_kernels=n_kernels,
                        max_dilations_per_kernel=max_dil,
                        n_jobs=int(self.n_jobs),
                        random_state=int(self.random_state),
                    ),
                    "sktime",
                )
            return (
                SktimeMiniRocket(
                    num_kernels=n_kernels,
                    max_dilations_per_kernel=max_dil,
                    n_jobs=int(self.n_jobs),
                    random_state=int(self.random_state),
                ),
                "sktime",
            )
        return (
            _NumpyMiniRocketTransformer(
                num_kernels=int(self.num_kernels),
                max_dilations_per_kernel=int(self.max_dilations_per_kernel),
                random_state=int(self.random_state),
            ),
            "numpy",
        )

    def _transform_batch(self, X_batch: np.ndarray) -> np.ndarray:
        if self.active_backend_ == "sktime":
            out = self.rocket_.transform(X_batch)
            return np.asarray(out, dtype=np.float32)
        return self.rocket_.transform(X_batch)

    def _fit_batched_ridge(
        self,
        X_tensor: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray],
        *,
        use_gcv: bool,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
        """Single-pass batched exact Ridge / RidgeCV (GCV) without materializing `(N, K)` in RAM."""
        n_rows = X_tensor.shape[0]
        bs = max(256, int(self.batch_size))
        y_f64 = np.asarray(y, dtype=np.float64).ravel()
        w_f64 = (
            np.asarray(sample_weights, dtype=np.float64).ravel()
            if sample_weights is not None
            else np.ones(n_rows, dtype=np.float64)
        )
        w_f64 = np.where(np.isfinite(w_f64) & (w_f64 > 0.0), w_f64, 0.0)
        w_sum = float(w_f64.sum())
        if w_sum <= 1e-12:
            w_f64 = np.ones(n_rows, dtype=np.float64)
            w_sum = float(n_rows)

        y_mean = float(np.dot(w_f64, y_f64) / w_sum)
        y_centered = y_f64 - y_mean

        sum_x: Optional[np.ndarray] = None
        sum_x2: Optional[np.ndarray] = None
        XtX: Optional[np.ndarray] = None
        Xty: Optional[np.ndarray] = None
        n_feat = 0

        for start in range(0, n_rows, bs):
            end = min(n_rows, start + bs)
            Z = self._transform_batch(X_tensor[start:end]).astype(np.float64, copy=False)
            if sum_x is None:
                n_feat = Z.shape[1]
                sum_x = np.zeros(n_feat, dtype=np.float64)
                sum_x2 = np.zeros(n_feat, dtype=np.float64)
                XtX = np.zeros((n_feat, n_feat), dtype=np.float64)
                Xty = np.zeros(n_feat, dtype=np.float64)
            wb = w_f64[start:end]
            yb = y_centered[start:end]
            Zw = Z * wb[:, None]
            sum_x += Zw.sum(axis=0)
            sum_x2 += (Zw * Z).sum(axis=0)
            XtX += Zw.T @ Z
            Xty += Z.T @ (wb * yb)
            del Z, Zw

        assert sum_x is not None and sum_x2 is not None and XtX is not None and Xty is not None
        mean_x = sum_x / w_sum
        var_x = np.maximum(0.0, (sum_x2 / w_sum) - mean_x * mean_x)
        if bool(self.standardize_features):
            scale_x = np.where(var_x > 1e-12, np.sqrt(var_x), 1.0)
        else:
            scale_x = np.ones(n_feat, dtype=np.float64)

        # Centered & standardized Gram matrix: (Z - mu)^T W (Z - mu) / (s_i * s_j)
        XtX_c = XtX - w_sum * np.outer(mean_x, mean_x)
        inv_scale = 1.0 / scale_x
        XtX_std = XtX_c * np.outer(inv_scale, inv_scale)
        XtX_std = 0.5 * (XtX_std + XtX_std.T)
        # Since y_centered is weighted-zero-mean, (Z - mu)^T (w * y_c) = Z^T (w * y_c)
        Xty_std = Xty * inv_scale

        if use_gcv:
            eigvals, eigvecs = np.linalg.eigh(XtX_std)
            eigvals = np.maximum(eigvals, 0.0)
            ut_xty = eigvecs.T @ Xty_std
            y_norm_sq = float(np.dot(w_f64, y_centered * y_centered))
            best_alpha = float(_DEFAULT_RIDGE_CV_ALPHAS[0])
            best_gcv = np.inf
            # Kish effective sample size for weighted observations + explicit +1.0 intercept df in GCV
            w_sq_sum = float(np.dot(w_f64, w_f64))
            n_eff = max((w_sum * w_sum) / max(w_sq_sum, 1e-12), 2.0)
            for cand_alpha in _DEFAULT_RIDGE_CV_ALPHAS:
                a_val = float(cand_alpha) * (w_sum / max(float(n_rows), 1.0))
                denom = eigvals + a_val
                coef_eig = ut_xty / np.maximum(denom, 1e-12)
                # RSS = ||y||^2 - 2 w^T Xty + w^T XtX w
                rss = max(
                    1e-12,
                    y_norm_sq - 2.0 * float(np.dot(coef_eig, ut_xty)) + float(np.dot(coef_eig * coef_eig, eigvals)),
                )
                df_trace = 1.0 + float(np.sum(eigvals / np.maximum(denom, 1e-12)))
                gcv_denom = max(1e-6, 1.0 - df_trace / n_eff)
                gcv_score = rss / (gcv_denom * gcv_denom)
                if gcv_score < best_gcv:
                    best_gcv = gcv_score
                    best_alpha = float(cand_alpha)
            chosen_alpha = best_alpha
            eff_alpha = chosen_alpha * (w_sum / max(float(n_rows), 1.0))
            coef_std = eigvecs @ (ut_xty / np.maximum(eigvals + eff_alpha, 1e-12))
        else:
            chosen_alpha = max(1e-8, float(self.alpha))
            eff_alpha = chosen_alpha * (w_sum / max(float(n_rows), 1.0))
            reg_mat = XtX_std + np.eye(n_feat, dtype=np.float64) * eff_alpha
            try:
                coef_std = np.linalg.solve(reg_mat, Xty_std)
            except np.linalg.LinAlgError:
                coef_std = np.linalg.lstsq(reg_mat, Xty_std, rcond=None)[0]

        coef_std = np.where(np.isfinite(coef_std), coef_std, 0.0)
        return coef_std, y_mean, mean_x, scale_x, chosen_alpha

    def _fit_batched_sgd(
        self,
        X_tensor: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray],
        *,
        head: str,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
        """Two-pass batched SGDRegressor (`sgd_elasticnet` or `sgd_huber`) with standardization."""
        n_rows = X_tensor.shape[0]
        bs = max(256, int(self.batch_size))
        y_f64 = np.asarray(y, dtype=np.float64).ravel()
        w_f64 = (
            np.asarray(sample_weights, dtype=np.float64).ravel()
            if sample_weights is not None
            else None
        )

        # Pass 1: feature mean/std
        sum_x: Optional[np.ndarray] = None
        sum_x2: Optional[np.ndarray] = None
        for start in range(0, n_rows, bs):
            end = min(n_rows, start + bs)
            Z = self._transform_batch(X_tensor[start:end]).astype(np.float64, copy=False)
            if sum_x is None:
                sum_x = np.zeros(Z.shape[1], dtype=np.float64)
                sum_x2 = np.zeros(Z.shape[1], dtype=np.float64)
            sum_x += Z.sum(axis=0)
            sum_x2 += (Z * Z).sum(axis=0)
            del Z

        assert sum_x is not None and sum_x2 is not None
        mean_x = sum_x / float(n_rows)
        var_x = np.maximum(0.0, (sum_x2 / float(n_rows)) - mean_x * mean_x)
        scale_x = np.where(var_x > 1e-12, np.sqrt(var_x), 1.0) if bool(self.standardize_features) else np.ones_like(mean_x)

        sgd_alpha = max(1e-7, float(self.alpha) / max(float(n_rows), 1.0))
        if head == "sgd_huber":
            sgd = SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=sgd_alpha,
                epsilon=max(1.0, float(self.huber_epsilon)),
                random_state=int(self.random_state),
            )
        else:
            sgd = SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                alpha=sgd_alpha,
                l1_ratio=float(np.clip(float(self.l1_ratio), 0.0, 1.0)),
                random_state=int(self.random_state),
            )

        # Pass 2: batched partial_fit (2 epochs with deterministic shuffling to avoid chronological recency bias)
        rng_sgd = np.random.default_rng(int(self.random_state))
        for _ in range(2):
            perm = rng_sgd.permutation(n_rows)
            for start in range(0, n_rows, bs):
                batch_idx = perm[start : min(n_rows, start + bs)]
                Z = self._transform_batch(X_tensor[batch_idx]).astype(np.float64, copy=False)
                Z_std = (Z - mean_x) / scale_x
                wb = w_f64[batch_idx] if w_f64 is not None else None
                sgd.partial_fit(Z_std, y_f64[batch_idx], sample_weight=wb)
                del Z, Z_std

        coef_std = np.asarray(sgd.coef_, dtype=np.float64).ravel()
        intercept = float(np.asarray(sgd.intercept_, dtype=np.float64).ravel()[0])
        return coef_std, intercept, mean_x, scale_x, float(self.alpha)

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
    ) -> "MiniRocketNumeraiEstimator":
        _, feature_matrix, target_values, _, sample_weights, era_values = self._prepare_fit_inputs(
            X, y, sample_weight=sample_weight, eras=eras, return_matrix=True, matrix_dtype=np.float32
        )
        assert feature_matrix is not None

        # Retain top `max_channels` selected features for sequence channels (or all for feature_profile)
        tickers = self._extract_tickers(X, len(target_values))
        if len(tickers) != len(target_values) if tickers is not None else False:
            # In-fit era_stride subsampling occurred inside `_prepare_fit_inputs`
            tickers = None
        if tickers is None and isinstance(X, pd.DataFrame) and self.ticker_col in X.columns:
            # Align tickers if in-fit era_stride subsampled X
            raw_eras = X[self.era_col].to_numpy() if self.era_col in X.columns else None
            if raw_eras is not None and int(self.era_stride or 1) > 1:
                kept_eras = sorted(pd.unique(raw_eras))[:: int(self.era_stride)]
                mask = np.isin(raw_eras, kept_eras)
                tickers = X.loc[mask, self.ticker_col].astype(str).to_numpy()
            else:
                tickers = X[self.ticker_col].astype(str).to_numpy()

        active_mode = self._resolve_input_mode(tickers)
        self.active_input_mode_ = active_mode

        sel_idx = np.asarray(self.selected_indices_, dtype=np.int64)
        if active_mode == "ticker_sequence":
            max_ch = max(1, int(self.max_channels))
            if len(sel_idx) > max_ch:
                sel_idx = sel_idx[:max_ch]
        self.channel_indices_ = sel_idx
        self.channel_names_ = [str(self.feature_names_in_[i]) for i in sel_idx]
        self.channel_corrs_ = np.asarray(self.feature_corrs_[sel_idx], dtype=np.float64)

        sub_matrix = feature_matrix[:, sel_idx]
        X_tensor = self._build_sequence_tensor(
            sub_matrix,
            era_values,
            tickers,
            mode=active_mode,
            is_fit=True,
        )
        n_rows, n_channels, seq_len = X_tensor.shape
        self.n_channels_in_ = int(n_channels)
        self.seq_len_in_ = int(seq_len)

        rocket, active_backend = self._build_rocket_transformer(n_channels)
        self.active_backend_ = active_backend

        # Fit MiniRocket quantile biases on a deterministic subsample of training sequences
        rng = np.random.default_rng(int(self.random_state))
        fit_n = min(n_rows, max(16, int(self.fit_sample_size)))
        if fit_n < n_rows:
            fit_idx = np.sort(rng.choice(n_rows, size=fit_n, replace=False))
            rocket.fit(X_tensor[fit_idx])
        else:
            rocket.fit(X_tensor)
        self.rocket_ = rocket

        head = str(_decode_structured(self.head_type) or "ridge_cv").strip().lower()
        if head not in ("ridge_cv", "ridge", "sgd_elasticnet", "sgd_huber"):
            raise ValueError(
                f"Unsupported head_type '{self.head_type}'; use 'ridge_cv', 'ridge', 'sgd_elasticnet', or 'sgd_huber'."
            )
        self.active_head_type_ = head

        if head in ("ridge_cv", "ridge"):
            coef_std, intercept, mean_x, scale_x, chosen_alpha = self._fit_batched_ridge(
                X_tensor, target_values, sample_weights, use_gcv=(head == "ridge_cv")
            )
        else:
            coef_std, intercept, mean_x, scale_x, chosen_alpha = self._fit_batched_sgd(
                X_tensor, target_values, sample_weights, head=head
            )

        self.coef_ = coef_std
        self.intercept_ = float(intercept)
        self.feature_mean_ = mean_x
        self.feature_scale_ = scale_x
        self.chosen_alpha_ = float(chosen_alpha)
        self.n_rocket_features_ = int(len(coef_std))

        # Build interpretability tables (top MiniRocket kernels & channel attributions)
        abs_coef = np.abs(coef_std)
        if active_backend == "numpy" and hasattr(rocket, "feature_meta_"):
            meta_rows = []
            ch_imp = np.zeros(len(self.channel_names_), dtype=np.float64)
            for idx_f, meta in enumerate(rocket.feature_meta_):
                w_val = float(coef_std[idx_f])
                aw_val = float(abs_coef[idx_f])
                chs = meta["channels"]
                if active_mode == "ticker_sequence" and len(chs) > 0:
                    share = aw_val / float(len(chs))
                    for c_i in chs:
                        if 0 <= c_i < len(ch_imp):
                            ch_imp[c_i] += share
                meta_rows.append({
                    "rocket_feature": f"k{meta['kernel_id']:02d}_d{meta['dilation']}_f{idx_f}",
                    "kernel_id": int(meta["kernel_id"]),
                    "dilation": int(meta["dilation"]),
                    "bias": float(meta["bias"]),
                    "coef": w_val,
                    "abs_coef": aw_val,
                    "ppv_mean": float(mean_x[idx_f]),
                    "ppv_std": float(scale_x[idx_f]),
                })
            self.kernel_importance_df_ = (
                pd.DataFrame(meta_rows)
                .sort_values("abs_coef", ascending=False, kind="stable")
                .reset_index(drop=True)
            )
        else:
            ch_imp = np.zeros(len(self.channel_names_), dtype=np.float64)
            self.kernel_importance_df_ = (
                pd.DataFrame({
                    "rocket_feature": [f"mr_f{i}" for i in range(len(coef_std))],
                    "kernel_id": np.arange(len(coef_std)) % _NUM_BASE_KERNELS,
                    "dilation": np.nan,
                    "bias": np.nan,
                    "coef": coef_std,
                    "abs_coef": abs_coef,
                    "ppv_mean": mean_x,
                    "ppv_std": scale_x,
                })
                .sort_values("abs_coef", ascending=False, kind="stable")
                .reset_index(drop=True)
            )

        self.channel_importance_df_ = (
            pd.DataFrame({
                "feature": self.channel_names_,
                "kernel_weight_sum": ch_imp,
                "train_corr": self.channel_corrs_,
            })
            .sort_values(["kernel_weight_sum", "train_corr"], ascending=False, kind="stable")
            .reset_index(drop=True)
        )

        if self.bias_correction not in (None, False, "None", "none", ""):
            raw_train_pred = self._predict_tensor(X_tensor)
            self._fit_bias_correction(target_values, raw_train_pred, era_values)
            del raw_train_pred
        else:
            self.learned_bias_offset_ = 0.0

        del X_tensor, sub_matrix, feature_matrix
        return self

    def _predict_tensor(self, X_tensor: np.ndarray) -> np.ndarray:
        n_rows = X_tensor.shape[0]
        bs = max(256, int(self.batch_size))
        preds = np.empty(n_rows, dtype=np.float64)
        # Fold standardization into effective weight vector for single BLAS matvec per batch
        eff_coef = self.coef_ / self.feature_scale_
        eff_intercept = self.intercept_ - float(np.dot(self.feature_mean_, eff_coef))
        for start in range(0, n_rows, bs):
            end = min(n_rows, start + bs)
            Z = self._transform_batch(X_tensor[start:end]).astype(np.float64, copy=False)
            preds[start:end] = Z @ eff_coef + eff_intercept
            del Z
        return preds

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "rocket_") or not hasattr(self, "coef_"):
            raise ValueError("MiniRocketNumeraiEstimator must be fitted before calling predict.")
        feature_matrix, era_values = self._prepare_predict_matrix(X)
        tickers = self._extract_tickers(X, len(feature_matrix))
        sub_matrix = np.asarray(feature_matrix[:, self.channel_indices_], dtype=np.float32)
        X_tensor = self._build_sequence_tensor(
            sub_matrix,
            era_values,
            tickers,
            mode=self.active_input_mode_,
            is_fit=False,
        )
        raw_pred = self._predict_tensor(X_tensor)
        del X_tensor, sub_matrix, feature_matrix
        return self._apply_prediction_transform(raw_pred, era_values)


MiniRocketEstimator = MiniRocketNumeraiEstimator
