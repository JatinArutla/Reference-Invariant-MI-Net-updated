import numpy as np

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


def apply_reference(
    X: np.ndarray,
    mode: str = "native",
    *,
    ref_idx: int | None = None,
    lap_neighbors: list[list[int]] | None = None,
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
        # Note: "car_full" vs "car_intersection" is controlled by channel subsetting
        return apply_car(X)
    if m in ("ref", "cz_ref", "channel_ref"):
        if ref_idx is None:
            raise ValueError("ref_idx is required for mode='ref'")
        return apply_ref_channel(X, ref_idx)
    if m in ("laplacian", "lap", "local"):
        if lap_neighbors is None:
            raise ValueError("lap_neighbors is required for mode='laplacian'")
        return apply_laplacian(X, lap_neighbors)
    raise ValueError(f"Unknown reference mode: {mode}")

def _eigh_inv_sqrt(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w, v = np.linalg.eigh(M + eps * np.eye(M.shape[0], dtype=M.dtype))
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def ea_fit(X: np.ndarray) -> np.ndarray:
    """Fit Euclidean Alignment (EA) matrix for one block.

    EA is defined by the inverse square-root of the mean trial covariance.
    The returned matrix W is applied as: x_aligned = W @ x.

    X: [N,C,T]
    Returns: W [C,C]
    """
    covs = [x @ x.T / x.shape[1] for x in X]
    Rbar = np.mean(covs, axis=0)
    W = _eigh_inv_sqrt(Rbar.astype(np.float64)).astype(np.float32)
    return W


def ea_apply(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply a pre-fit EA matrix W to X.

    Supports:
      - X [N,C,T]
      - X [C,T]
    """
    if X.ndim == 3:
        return np.asarray([W @ x for x in X], dtype=np.float32)
    if X.ndim == 2:
        return (W @ X).astype(np.float32)
    raise ValueError(f"ea_apply expects 2D/3D array, got {X.ndim}D")

def ea_align_trials(X: np.ndarray) -> np.ndarray:
    """EA for one subject/session block. X: [N,C,T] -> [N,C,T]."""
    W = ea_fit(X)
    return ea_apply(X, W)

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