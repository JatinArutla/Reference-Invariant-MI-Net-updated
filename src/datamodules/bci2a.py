import os
import numpy as np
import scipy.io as sio
from typing import Tuple, Dict, Any
from .transforms import ea_align_trials, standardize_pair, standardize_loso_block, apply_reference
from .channels import BCI2A_CH_NAMES, parse_keep_channels, subset_and_reorder, neighbors_to_index_list, name_to_index

FS = 250

def _load_session_mat(path: str, subject: int, training: bool) -> Dict[str, Any]:
    name = f"A0{subject}{'T' if training else 'E'}.mat"
    return sio.loadmat(os.path.join(path, name))["data"]

def load_bci2a_session(
    data_root: str,
    subject: int,
    training: bool,
    *,
    all_trials: bool = True,
    t1_sec: float = 2.0,
    t2_sec: float = 6.0,
    ref_mode: str = "native",
    keep_channels: str | None = None,
    ref_channel: str = "Cz",
    laplacian: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns X [N,C,T], y [N] for A0sT/A0sE cropped to t∈[t1_sec,t2_sec].
    """
    n_channels = 22
    n_tests = 6 * 48
    win_len = 7 * FS
    t1, t2 = int(t1_sec * FS), int(t2_sec * FS)

    X = np.zeros((n_tests, n_channels, win_len), dtype=np.float32)
    y = np.zeros(n_tests, dtype=np.int64)

    a_data = _load_session_mat(data_root, subject, training)
    k = 0
    for ii in range(a_data.size):
        d = a_data[0, ii][0, 0]
        a_X, a_trial, a_y, a_art = d[0], d[1], d[2], d[5]
        for t in range(a_trial.size):
            if a_art[t] != 0 and not all_trials:
                continue
            seg = a_X[int(a_trial[t]) : int(a_trial[t]) + win_len, :n_channels].T  # [C, win_len]
            X[k] = seg
            y[k] = int(a_y[t])
            k += 1

    X = X[:k, :, t1:t2]               # [N,C,T=1000]
    y = (y[:k] - 1).astype(np.int64)  # [0..3]

    # Optional: channel intersection/subset (e.g., to make CAR comparable across datasets)
    keep_idx = parse_keep_channels(keep_channels, all_names=BCI2A_CH_NAMES)
    keep_names = None
    if keep_idx is not None:
        keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx]
        X = subset_and_reorder(X, keep_idx)

    # Optional: re-reference before any EA/standardization
    # ref_idx is needed for:
    #   - explicit channel referencing (mode='ref')
    #   - bipolar root selection (tree + 2-cycle bipolar uses a root channel)
    ref_idx = None
    rm = (ref_mode or "").lower()
    if rm in ("ref", "cz_ref", "channel_ref", "bipolar", "bip", "bipolar_like"):
        # Map ref channel name to index in the current channel order
        current_names = keep_names if keep_names is not None else BCI2A_CH_NAMES
        ref_map = name_to_index(current_names)
        if ref_channel not in ref_map:
            raise ValueError(f"ref_channel '{ref_channel}' not in channels: {current_names}")
        ref_idx = int(ref_map[ref_channel])

    lap_neighbors = None
    # Some reference operators require a neighbor graph.
    if laplacian or (ref_mode or "").lower() in (
        "laplacian", "lap", "local",
        "bipolar", "bip", "bipolar_like",
        "bipolar_edges", "bip_edges", "edges_bipolar",
    ):
        lap_neighbors = neighbors_to_index_list(
            all_names=BCI2A_CH_NAMES,
            keep_names=keep_names,
            sort_by_distance=True,
        )

    X = apply_reference(X, mode=ref_mode, ref_idx=ref_idx, lap_neighbors=lap_neighbors)
    return X, y

def load_subject_dependent(
    data_root: str,
    subject: int,
    *,
    ea: bool = True,
    standardize: bool = True,
    t1_sec: float = 2.0,
    t2_sec: float = 6.0,
    ref_mode: str = "native",
    keep_channels: str | None = None,
    ref_channel: str = "Cz",
    laplacian: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    A0sT → train, A0sE → test. Optional EA and standardization (fit on train).
    """
    X_tr, y_tr = load_bci2a_session(
        data_root, subject, True,
        t1_sec=t1_sec, t2_sec=t2_sec,
        ref_mode=ref_mode, keep_channels=keep_channels,
        ref_channel=ref_channel, laplacian=laplacian,
    )
    X_te, y_te = load_bci2a_session(
        data_root, subject, False,
        t1_sec=t1_sec, t2_sec=t2_sec,
        ref_mode=ref_mode, keep_channels=keep_channels,
        ref_channel=ref_channel, laplacian=laplacian,
    )

    if ea:
        X_tr = ea_align_trials(X_tr)
        X_te = ea_align_trials(X_te)

    if standardize:
        X_tr, X_te = standardize_pair(X_tr, X_te)

    return (X_tr, y_tr), (X_te, y_te)

def load_LOSO_pool(
    data_root: str,
    target_sub: int,
    *,
    n_sub: int = 9,
    ea: bool = True,
    standardize: bool = True,
    per_block_standardize: bool = True,
    t1_sec: float = 2.0,
    t2_sec: float = 6.0,
    ref_mode: str = "native",
    keep_channels: str | None = None,
    ref_channel: str = "Cz",
    laplacian: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Pool T+E per subject. Target subject kept separate. If `ea`, apply per block.
    If `standardize`:
      - if `per_block_standardize`: standardize each subject block before pooling,
      - else: standardize after pooling using pooled-source stats.
    """
    blocks = {}
    for s in range(1, n_sub + 1):
        X1, y1 = load_bci2a_session(
            data_root, s, True,
            t1_sec=t1_sec, t2_sec=t2_sec,
            ref_mode=ref_mode, keep_channels=keep_channels,
            ref_channel=ref_channel, laplacian=laplacian,
        )
        X2, y2 = load_bci2a_session(
            data_root, s, False,
            t1_sec=t1_sec, t2_sec=t2_sec,
            ref_mode=ref_mode, keep_channels=keep_channels,
            ref_channel=ref_channel, laplacian=laplacian,
        )
        X = np.concatenate([X1, X2], axis=0)
        y = np.concatenate([y1, y2], axis=0)

        if ea:
            X = ea_align_trials(X)
        if standardize and per_block_standardize:
            X = standardize_loso_block(X)

        blocks[s] = (X, y)

    X_tgt, y_tgt = blocks[target_sub]
    src_keys = [k for k in blocks.keys() if k != target_sub]
    X_src = np.concatenate([blocks[k][0] for k in src_keys], axis=0)
    y_src = np.concatenate([blocks[k][1] for k in src_keys], axis=0)

    if standardize and not per_block_standardize:
        # Fit on pooled sources; apply to both sources and target.
        from .transforms import fit_standardizer, apply_standardizer
        mu, sd = fit_standardizer(X_src)
        X_src = apply_standardizer(X_src, mu, sd)
        X_tgt = apply_standardizer(X_tgt, mu, sd)

    return (X_src.astype(np.float32), y_src.astype(np.int64)), (X_tgt.astype(np.float32), y_tgt.astype(np.int64))