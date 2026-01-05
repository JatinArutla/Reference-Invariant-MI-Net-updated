import numpy as np

# These reference transforms are designed to be model-agnostic and dataset-agnostic
# within the constraints of multi-channel EEG. They operate on either:
#   - single trials  [C, T]
#   - batches        [N, C, T]
#
# The goal is to isolate "reference" as a controlled distribution shift.
# Keep these functions simple and deterministic.

# -------------------------
# Reference transforms
# -------------------------

def apply_car(X: np.ndarray, *, channel_axis: int = 1) -> np.ndarray:
    """Common Average Reference (CAR).

    Supports:
      - 3D: [N,C,T] with channel_axis=1 (default)
      - 2D: [C,T] with channel_axis=0
    """
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        if channel_axis != 1:
            raise ValueError("For 3D input, channel_axis must be 1 for [N,C,T].")
        mean = Xf.mean(axis=1, keepdims=True)  # [N,1,T]
        return (Xf - mean).astype(np.float32)
    if Xf.ndim == 2:
        if channel_axis != 0:
            raise ValueError("For 2D input, channel_axis must be 0 for [C,T].")
        mean = Xf.mean(axis=0, keepdims=True)  # [1,T]
        return (Xf - mean).astype(np.float32)
    raise ValueError(f"apply_car expects 2D or 3D array, got {Xf.ndim}D")


def apply_median_reference(X: np.ndarray) -> np.ndarray:
    """Median reference (robust alternative to CAR).

    For each timepoint t, subtract median across channels.
    """
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        med = np.median(Xf, axis=1, keepdims=True)  # [N,1,T]
        return (Xf - med).astype(np.float32)
    if Xf.ndim == 2:
        med = np.median(Xf, axis=0, keepdims=True)  # [1,T]
        return (Xf - med).astype(np.float32)
    raise ValueError(f"apply_median_reference expects 2D or 3D array, got {Xf.ndim}D")


def apply_ref_channel(X: np.ndarray, ref_idx: int) -> np.ndarray:
    """Re-reference to a recorded channel: X_new[c,t] = X[c,t] - X[ref,t]."""
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        ref = Xf[:, ref_idx:ref_idx + 1, :]  # [N,1,T]
        return (Xf - ref).astype(np.float32)
    if Xf.ndim == 2:
        ref = Xf[ref_idx:ref_idx + 1, :]     # [1,T]
        return (Xf - ref).astype(np.float32)
    raise ValueError(f"apply_ref_channel expects 2D or 3D array, got {Xf.ndim}D")


def apply_gram_schmidt(X: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Gram–Schmidt orthogonalization reference.

    For each channel i:
        r_i(t) = mean_{j!=i} x_j(t)
        alpha_i = <x_i, r_i> / (<r_i, r_i> + eps)
        y_i = x_i - alpha_i * r_i

    This is a robust alternative to CAR when some channels are corrupted.
    """
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        N, C, T = Xf.shape
        s = Xf.sum(axis=1, keepdims=True)  # [N,1,T]
        # r_i = (sum - x_i) / (C-1)
        r = (s - Xf) / max(C - 1, 1)
        num = np.sum(Xf * r, axis=2, keepdims=True)          # [N,C,1]
        den = np.sum(r * r, axis=2, keepdims=True) + eps     # [N,C,1]
        alpha = num / den
        return (Xf - alpha * r).astype(np.float32)
    if Xf.ndim == 2:
        C, T = Xf.shape
        s = Xf.sum(axis=0, keepdims=True)  # [1,T]
        r = (s - Xf) / max(C - 1, 1)
        num = np.sum(Xf * r, axis=1, keepdims=True)          # [C,1]
        den = np.sum(r * r, axis=1, keepdims=True) + eps     # [C,1]
        alpha = num / den
        return (Xf - alpha * r).astype(np.float32)
    raise ValueError(f"apply_gram_schmidt expects 2D or 3D array, got {Xf.ndim}D")


def apply_laplacian(X: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    """Simple Laplacian / local average reference.

    For each channel i:
      X_i <- X_i - mean(X_neighbors(i))

    neighbors is a list of lists with length C.
    """
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        N, C, T = Xf.shape
        if len(neighbors) != C:
            raise ValueError(f"neighbors length {len(neighbors)} != C {C}")
        Y = Xf.copy()
        for i, nei in enumerate(neighbors):
            if not nei:
                continue
            Y[:, i, :] = Xf[:, i, :] - Xf[:, nei, :].mean(axis=1)
        return Y.astype(np.float32)
    if Xf.ndim == 2:
        C, T = Xf.shape
        if len(neighbors) != C:
            raise ValueError(f"neighbors length {len(neighbors)} != C {C}")
        Y = Xf.copy()
        for i, nei in enumerate(neighbors):
            if not nei:
                continue
            Y[i, :] = Xf[i, :] - Xf[nei, :].mean(axis=0)
        return Y.astype(np.float32)
    raise ValueError(f"apply_laplacian expects 2D or 3D array, got {Xf.ndim}D")


def apply_bipolar(X: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    """Bipolar-like derivation that preserves channel count.

    For each channel i, subtract ONE neighbor (the first in neighbors[i])
    if available:
        y_i = x_i - x_{neighbors[i][0]}

    This approximates bipolar derivations while keeping a fixed input dimension.
    """
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        N, C, T = Xf.shape
        if len(neighbors) != C:
            raise ValueError(f"neighbors length {len(neighbors)} != C {C}")
        Y = Xf.copy()
        for i, nei in enumerate(neighbors):
            if not nei:
                continue
            j = int(nei[0])
            Y[:, i, :] = Xf[:, i, :] - Xf[:, j, :]
        return Y.astype(np.float32)
    if Xf.ndim == 2:
        C, T = Xf.shape
        if len(neighbors) != C:
            raise ValueError(f"neighbors length {len(neighbors)} != C {C}")
        Y = Xf.copy()
        for i, nei in enumerate(neighbors):
            if not nei:
                continue
            j = int(nei[0])
            Y[i, :] = Xf[i, :] - Xf[j, :]
        return Y.astype(np.float32)
    raise ValueError(f"apply_bipolar expects 2D or 3D array, got {Xf.ndim}D")


def apply_random_global_reference(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    dirichlet_alpha: float = 1.0,
) -> np.ndarray:
    """Subtract a random weighted average of channels.

    ref(t) = sum_c w_c * x_c(t), with w ~ Dirichlet(alpha).
    y_c(t) = x_c(t) - ref(t)

    Intended primarily as TRAIN-TIME augmentation to approximate arbitrary
    reference choices in the wild.
    """
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        N, C, T = Xf.shape
        w = rng.dirichlet(alpha=np.full(C, float(dirichlet_alpha), dtype=np.float64)).astype(np.float32)  # [C]
        ref = np.tensordot(Xf, w, axes=([1], [0]))  # [N,T]
        return (Xf - ref[:, None, :]).astype(np.float32)
    if Xf.ndim == 2:
        C, T = Xf.shape
        w = rng.dirichlet(alpha=np.full(C, float(dirichlet_alpha), dtype=np.float64)).astype(np.float32)
        ref = np.tensordot(w, Xf, axes=([0], [0]))  # [T]
        return (Xf - ref[None, :]).astype(np.float32)
    raise ValueError(f"apply_random_global_reference expects 2D or 3D array, got {Xf.ndim}D")


def apply_reference(
    X: np.ndarray,
    mode: str = "native",
    *,
    ref_idx: int | None = None,
    lap_neighbors: list[list[int]] | None = None,
    rng: np.random.Generator | None = None,
    randref_alpha: float = 1.0,
) -> np.ndarray:
    """Apply a reference transform to X.

    mode:
      - native: no-op
      - car: common average reference over available channels
      - ref: re-reference to a recorded channel (requires ref_idx)
      - laplacian: local reference (requires lap_neighbors)
    """
    m = (mode or "native").lower()
    if m in ("native", "none", ""):
        return X.astype(np.float32, copy=False)
    if m in ("car", "car_full", "car_intersection"):
        # Handle both [C,T] (single trial) and [N,C,T] (batched).
        if X.ndim == 2:
            return apply_car(X, channel_axis=0)
        return apply_car(X, channel_axis=1)
    if m in ("median", "median_ref", "median_reference"):
        return apply_median_reference(X)
    if m in ("ref", "cz_ref", "channel_ref"):
        if ref_idx is None:
            raise ValueError("ref_idx is required for mode='ref'")
        return apply_ref_channel(X, ref_idx)
    if m in ("gs", "gram_schmidt", "gram-schmidt"):
        return apply_gram_schmidt(X)
    if m in ("laplacian", "lap", "local"):
        if lap_neighbors is None:
            raise ValueError("lap_neighbors is required for mode='laplacian'")
        return apply_laplacian(X, lap_neighbors)
    if m in ("bipolar", "bip", "bipolar_like"):
        if lap_neighbors is None:
            raise ValueError("lap_neighbors is required for mode='bipolar'")
        return apply_bipolar(X, lap_neighbors)
    if m in ("randref", "random_ref", "random_global_ref"):
        if rng is None:
            rng = np.random.default_rng(0)
        return apply_random_global_reference(X, rng=rng, dirichlet_alpha=randref_alpha)
    raise ValueError(f"Unknown reference mode: {mode}")

def _eigh_inv_sqrt(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Stable inverse square-root for symmetric PSD-ish matrices.

    EA can produce tiny negative eigenvalues due to numerical noise.
    Clip eigenvalues to avoid NaNs.
    """
    Md = (M + M.T) * 0.5
    w, v = np.linalg.eigh(Md)
    w = np.clip(w, eps, None)
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T

def ea_align_trials(X: np.ndarray) -> np.ndarray:
    """EA for one subject/session block. X: [N,C,T] -> [N,C,T]."""
    covs = [x @ x.T / x.shape[1] for x in X]
    Rbar = np.mean(covs, axis=0)
    W = _eigh_inv_sqrt(Rbar.astype(np.float64)).astype(np.float32)
    return np.asarray([W @ x for x in X], dtype=np.float32)

def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Channel-wise z-score stats. 3D: mean/std over (N,T). 4D: over (N,M,T)."""
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        mu = Xf.mean(axis=(0, 2), keepdims=True)
        sd = Xf.std(axis=(0, 2), keepdims=True) + 1e-8
    elif Xf.ndim == 4:
        mu = Xf.mean(axis=(0, 1, 3), keepdims=True)
        sd = Xf.std(axis=(0, 1, 3), keepdims=True) + 1e-8
    else:
        raise ValueError(f"Expected 3D/4D array, got {Xf.ndim}D.")
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float32, copy=False) - mu) / sd).astype(np.float32)

def standardize_pair(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X_train.ndim != X_test.ndim:
        raise ValueError(f"X_train.ndim ({X_train.ndim}) != X_test.ndim ({X_test.ndim})")
    mu, sd = fit_standardizer(X_train)
    return apply_standardizer(X_train, mu, sd), apply_standardizer(X_test, mu, sd)

def standardize_loso_block(X: np.ndarray) -> np.ndarray:
    """Within-block standardization used when pooling subjects (LOSO helper)."""
    mu, sd = fit_standardizer(X)
    return apply_standardizer(X, mu, sd)