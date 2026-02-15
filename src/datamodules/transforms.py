import numpy as np
from collections import deque

# These reference transforms are designed to be model-agnostic and dataset-agnostic
# within the constraints of multi-channel EEG. They operate on either:
#   - single trials  [C, T]
#   - batches        [N, C, T]
#
# The goal is to isolate "reference" as a controlled distribution shift.
# Keep these functions simple and deterministic.

# -------------------------
# Canonical channel order (used for fixed bipolar permutation)
# -------------------------

# IMPORTANT: The fixed bipolar permutation below assumes the input channel order is exactly:
CANON_CHS_18 = [
    "Fz", "FC3", "FC1", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4", "Pz"
]

# Single-cycle partner map for CANON_CHS_18 in the exact order above.
# y[i] = x[i] - x[perm[i]]
# This is a single 18-cycle, so null_dim = 1, rank = 17, and no channel is forced to zero.
BIPOLAR_PERM_CANON18 = [2, 5, 7, 4, 10, 11, 1, 13, 0, 3, 16, 8, 6, 14, 15, 9, 17, 12]

# If True, and C==18, bipolar will use the fixed single-cycle permutation above.
# If False, bipolar will be built from the neighbor graph (tree + 2-cycle) like other C values.
USE_FIXED_BIPOLAR_CYCLE_CANON18 = True


# -------------------------
# Small helpers
# -------------------------

def _get_C(X: np.ndarray) -> int:
    if X.ndim == 2:
        return int(X.shape[0])
    if X.ndim == 3:
        return int(X.shape[1])
    raise ValueError(f"Expected 2D [C,T] or 3D [N,C,T], got {X.ndim}D.")


def _check_ref_idx(ref_idx: int, C: int) -> None:
    # Disallow negative indexing here to avoid silent bugs.
    if not isinstance(ref_idx, (int, np.integer)):
        raise TypeError(f"ref_idx must be int, got {type(ref_idx)}")
    if ref_idx < 0 or ref_idx >= C:
        raise ValueError(f"ref_idx {ref_idx} out of range for C={C}")


def _validate_neighbors(neighbors: list[list[int]], C: int) -> None:
    if len(neighbors) != C:
        raise ValueError(f"neighbors length {len(neighbors)} != C {C}")
    for i, ns in enumerate(neighbors):
        for j in ns:
            jj = int(j)
            if jj < 0 or jj >= C:
                raise ValueError(f"Neighbor index {jj} out of range for C={C} (at i={i})")
            if jj == i:
                # self-neighbors are pointless and can create degenerate outputs
                continue


def _is_single_cycle_perm(perm: np.ndarray) -> bool:
    # Checks perm is exactly one cycle that visits all nodes once.
    C = perm.shape[0]
    seen = np.zeros(C, dtype=bool)
    cur = 0
    for _ in range(C):
        if seen[cur]:
            return False
        seen[cur] = True
        cur = int(perm[cur])
    return (cur == 0) and bool(seen.all())


def make_bipolar_tree_2cycle_neighbors(
    neighbors_lists: list[list[int]],
    *,
    root_idx: int,
) -> list[list[int]]:
    """Build a deterministic parent map (tree + one 2-cycle) from an undirected neighbor graph.

    Output is a neighbor list of length C such that apply_bipolar() will compute:
        y[i] = x[i] - x[parent[i]]

    Construction:
      1) BFS spanning tree rooted at root_idx.
      2) Replace the root self-loop with a 2-cycle between (root_idx, root_partner).

    Guarantees (if the graph is connected and C>1):
      - exactly one directed cycle (length 2)
      - null_dim == 1, rank == C-1
      - no forced all-zero channel (unlike root subtracting itself)
    """
    C = len(neighbors_lists)
    if C == 0:
        raise ValueError("neighbors_lists is empty")
    if C == 1:
        return [[0]]

    _check_ref_idx(int(root_idx), C)
    _validate_neighbors(neighbors_lists, C)

    # Undirected adjacency
    adj: list[set[int]] = [set() for _ in range(C)]
    for i, ns in enumerate(neighbors_lists):
        for j in ns:
            jj = int(j)
            if jj == i:
                continue
            adj[i].add(jj)
            adj[jj].add(i)

    if not adj[root_idx]:
        # Extremely degenerate: root has no neighbors.
        # Fall back to a self-loop to keep things defined.
        return [[int(root_idx)] for _ in range(C)]

    # BFS spanning tree
    parent: list[int | None] = [None] * C
    parent[root_idx] = root_idx
    q: deque[int] = deque([int(root_idx)])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if parent[v] is None:
                parent[v] = u
                q.append(v)

    if any(p is None for p in parent):
        bad = [i for i, p in enumerate(parent) if p is None]
        raise RuntimeError(f"Neighbor graph disconnected; unreachable nodes: {bad}")

    # 2-cycle root trick to avoid a forced zero channel
    root_partner = sorted(adj[int(root_idx)])[0]
    parent[int(root_idx)] = int(root_partner)
    parent[int(root_partner)] = int(root_idx)

    return [[int(p)] for p in parent]


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
    C = _get_C(X)
    _check_ref_idx(ref_idx, C)

    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        ref = Xf[:, ref_idx:ref_idx + 1, :]  # [N,1,T]
        return (Xf - ref).astype(np.float32)
    if Xf.ndim == 2:
        ref = Xf[ref_idx:ref_idx + 1, :]     # [1,T]
        return (Xf - ref).astype(np.float32)
    raise ValueError(f"apply_ref_channel expects 2D or 3D array, got {Xf.ndim}D")


def apply_gram_schmidt(X: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Gram-Schmidt-style orthogonalization reference (data-dependent).

    For each channel i:
        r_i(t) = mean_{j!=i} x_j(t)
        alpha_i = <x_i, r_i> / (<r_i, r_i> + eps)
        y_i = x_i - alpha_i * r_i
    """
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        N, C, T = Xf.shape
        s = Xf.sum(axis=1, keepdims=True)  # [N,1,T]
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
      y_i <- x_i - mean(x_neighbors(i))
    """
    Xf = X.astype(np.float32, copy=False)

    if Xf.ndim == 3:
        N, C, T = Xf.shape
        _validate_neighbors(neighbors, C)
        Y = Xf.copy()
        for i, nei in enumerate(neighbors):
            if not nei:
                continue
            Y[:, i, :] = Xf[:, i, :] - Xf[:, nei, :].mean(axis=1)
        return Y.astype(np.float32)

    if Xf.ndim == 2:
        C, T = Xf.shape
        _validate_neighbors(neighbors, C)
        Y = Xf.copy()
        for i, nei in enumerate(neighbors):
            if not nei:
                continue
            Y[i, :] = Xf[i, :] - Xf[nei, :].mean(axis=0)
        return Y.astype(np.float32)

    raise ValueError(f"apply_laplacian expects 2D or 3D array, got {Xf.ndim}D")


def apply_bipolar(X: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    """Nearest-neighbor differencing that preserves channel count.

    For each channel i, subtract ONE neighbor (neighbors[i][0]) if available:
        y_i = x_i - x_{neighbors[i][0]}

    Note: if neighbors[i][0] induces multiple directed cycles, this operator can have
    null_dim > 1. For CANON_CHS_18, prefer apply_bipolar_perm() below.
    """
    Xf = X.astype(np.float32, copy=False)

    if Xf.ndim == 3:
        N, C, T = Xf.shape
        _validate_neighbors(neighbors, C)
        Y = Xf.copy()
        for i, nei in enumerate(neighbors):
            if not nei:
                continue
            j = int(nei[0])
            if j == i:
                continue
            Y[:, i, :] = Xf[:, i, :] - Xf[:, j, :]
        return Y.astype(np.float32)

    if Xf.ndim == 2:
        C, T = Xf.shape
        _validate_neighbors(neighbors, C)
        Y = Xf.copy()
        for i, nei in enumerate(neighbors):
            if not nei:
                continue
            j = int(nei[0])
            if j == i:
                continue
            Y[i, :] = Xf[i, :] - Xf[j, :]
        return Y.astype(np.float32)

    raise ValueError(f"apply_bipolar expects 2D or 3D array, got {Xf.ndim}D")


def apply_bipolar_perm(X: np.ndarray, perm: list[int] | np.ndarray) -> np.ndarray:
    """Bipolar-like differencing using a fixed partner permutation.

    y[i] = x[i] - x[perm[i]]

    If perm is a single cycle, then:
      - null_dim = 1
      - rank = C - 1
      - no dead channel by construction
    """
    Xf = X.astype(np.float32, copy=False)
    perm = np.asarray(perm, dtype=np.int64)

    if Xf.ndim == 3:
        _, C, _ = Xf.shape
    elif Xf.ndim == 2:
        C, _ = Xf.shape
    else:
        raise ValueError(f"apply_bipolar_perm expects 2D or 3D array, got {Xf.ndim}D")

    if perm.shape != (C,):
        raise ValueError(f"perm must have shape (C,), got {perm.shape} for C={C}")

    # Must be a permutation of 0..C-1
    if np.unique(perm).size != C or int(perm.min()) < 0 or int(perm.max()) >= C:
        raise ValueError("perm must be a permutation of [0..C-1]")

    # Enforce single cycle to avoid null_dim > 1
    if not _is_single_cycle_perm(perm):
        raise ValueError("perm is not a single cycle; this would create null_dim > 1.")

    if Xf.ndim == 3:
        return (Xf - Xf[:, perm, :]).astype(np.float32)
    return (Xf - Xf[perm, :]).astype(np.float32)


def _edge_list_from_neighbors(neighbors: list[list[int]]) -> list[tuple[int, int]]:
    """Deterministic undirected edge list (i<j) from neighbor index lists."""
    edges = set()
    C = len(neighbors)
    _validate_neighbors(neighbors, C)
    for i, ns in enumerate(neighbors):
        for j in ns:
            a, b = i, int(j)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return sorted(edges)


def apply_bipolar_edges(
    X: np.ndarray,
    neighbors: list[list[int]],
    *,
    edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Edge-bipolar representation (changes channel count).

    Builds an undirected edge list from the neighbor graph and returns a signal
    per edge:
        y_(i,j) = x_i - x_j

    Output shape:
      - input [N,C,T] -> [N,E,T]
      - input [C,T]   -> [E,T]
    """
    Xf = X.astype(np.float32, copy=False)
    C = _get_C(Xf)
    _validate_neighbors(neighbors, C)

    if edges is None:
        edges = _edge_list_from_neighbors(neighbors)

    if Xf.ndim == 3:
        N, C, T = Xf.shape
        Y = np.empty((N, len(edges), T), dtype=np.float32)
        for k, (i, j) in enumerate(edges):
            Y[:, k, :] = Xf[:, int(i), :] - Xf[:, int(j), :]
        return Y
    if Xf.ndim == 2:
        C, T = Xf.shape
        Y = np.empty((len(edges), T), dtype=np.float32)
        for k, (i, j) in enumerate(edges):
            Y[k, :] = Xf[int(i), :] - Xf[int(j), :]
        return Y
    raise ValueError(f"apply_bipolar_edges expects 2D or 3D array, got {Xf.ndim}D")


def apply_random_global_reference(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    dirichlet_alpha: float = 1.0,
) -> np.ndarray:
    """Subtract a random weighted average of channels.

    ref(t) = sum_c w_c * x_c(t), with w ~ Dirichlet(alpha).
    y_c(t) = x_c(t) - ref(t)
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
    """Apply a reference transform to X."""
    m = (mode or "native").lower()

    if m in ("native", "none", ""):
        return X.astype(np.float32, copy=False)

    if m in ("car", "car_full", "car_intersection"):
        if X.ndim == 2:
            return apply_car(X, channel_axis=0)
        return apply_car(X, channel_axis=1)

    if m in ("median", "median_ref", "median_reference"):
        return apply_median_reference(X)

    if m in ("ref", "cz_ref", "channel_ref"):
        if ref_idx is None:
            raise ValueError("ref_idx is required for mode='ref'")
        return apply_ref_channel(X, int(ref_idx))

    if m in ("gs", "gram_schmidt", "gram-schmidt"):
        return apply_gram_schmidt(X)

    if m in ("laplacian", "lap", "local"):
        if lap_neighbors is None:
            raise ValueError("lap_neighbors is required for mode='laplacian'")
        return apply_laplacian(X, lap_neighbors)

    if m in ("bipolar", "bip", "bipolar_like"):
        C = _get_C(X)

        # For CANON_CHS_18 (C==18), optionally use the fixed single-cycle permutation.
        if C == 18 and USE_FIXED_BIPOLAR_CYCLE_CANON18:
            return apply_bipolar_perm(X, BIPOLAR_PERM_CANON18)

        # General case (including 22ch): build a tree + 2-cycle parent map from the neighbor graph.
        if lap_neighbors is None:
            raise ValueError("lap_neighbors is required for mode='bipolar'")
        if ref_idx is None:
            raise ValueError("ref_idx is required for mode='bipolar' (root channel)")
        bip_neighbors = make_bipolar_tree_2cycle_neighbors(lap_neighbors, root_idx=int(ref_idx))
        return apply_bipolar(X, bip_neighbors)

    if m in ("bipolar_edges", "bip_edges", "edges_bipolar"):
        if lap_neighbors is None:
            raise ValueError("lap_neighbors is required for mode='bipolar_edges'")
        return apply_bipolar_edges(X, lap_neighbors)

    if m in ("randref", "random_ref", "random_global_ref"):
        # Training should pass rng explicitly. This fallback is deterministic on purpose.
        if rng is None:
            rng = np.random.default_rng(0)
        return apply_random_global_reference(X, rng=rng, dirichlet_alpha=randref_alpha)

    raise ValueError(f"Unknown reference mode: {mode}")


# -------------------------
# EA (Euclidean Alignment)
# -------------------------

def _eigh_inv_sqrt(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Stable inverse square-root for symmetric PSD-ish matrices."""
    Md = (M + M.T) * 0.5
    w, v = np.linalg.eigh(Md)
    w = np.clip(w, eps, None)
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def ea_align_trials(X: np.ndarray) -> np.ndarray:
    """EA for one subject/session block. X: [N,C,T] -> [N,C,T]."""
    if X.ndim != 3:
        raise ValueError(f"ea_align_trials expects 3D [N,C,T], got {X.ndim}D")
    covs = [x @ x.T / x.shape[1] for x in X]
    Rbar = np.mean(covs, axis=0)
    W = _eigh_inv_sqrt(Rbar.astype(np.float64)).astype(np.float32)
    return np.asarray([W @ x for x in X], dtype=np.float32)


# -------------------------
# Standardization
# -------------------------

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


def standardize_instance(
    X: np.ndarray,
    *,
    eps: float = 1e-8,
    robust: bool = False,
) -> np.ndarray:
    """Per-trial, per-channel standardization over time.

    Supports:
      - [N,C,T] -> [N,C,T]
      - [C,T]   -> [C,T]
    """
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        if robust:
            med = np.median(Xf, axis=2, keepdims=True)
            mad = np.median(np.abs(Xf - med), axis=2, keepdims=True)
            sd = 1.4826 * mad + eps
            return ((Xf - med) / sd).astype(np.float32)
        mu = Xf.mean(axis=2, keepdims=True)
        sd = Xf.std(axis=2, keepdims=True) + eps
        return ((Xf - mu) / sd).astype(np.float32)
    if Xf.ndim == 2:
        if robust:
            med = np.median(Xf, axis=1, keepdims=True)
            mad = np.median(np.abs(Xf - med), axis=1, keepdims=True)
            sd = 1.4826 * mad + eps
            return ((Xf - med) / sd).astype(np.float32)
        mu = Xf.mean(axis=1, keepdims=True)
        sd = Xf.std(axis=1, keepdims=True) + eps
        return ((Xf - mu) / sd).astype(np.float32)
    raise ValueError(f"standardize_instance expects 2D or 3D array, got {Xf.ndim}D")


def standardize_pair(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X_train.ndim != X_test.ndim:
        raise ValueError(f"X_train.ndim ({X_train.ndim}) != X_test.ndim ({X_test.ndim})")
    mu, sd = fit_standardizer(X_train)
    return apply_standardizer(X_train, mu, sd), apply_standardizer(X_test, mu, sd)


def standardize_loso_block(X: np.ndarray) -> np.ndarray:
    """Within-block standardization used when pooling subjects (LOSO helper)."""
    mu, sd = fit_standardizer(X)
    return apply_standardizer(X, mu, sd)