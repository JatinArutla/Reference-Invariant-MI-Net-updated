import numpy as np
from tensorflow.keras.utils import Sequence

from src.datamodules.transforms import apply_reference, standardize_instance
from src.datamodules.channels import BCI2A_CH_NAMES, neighbors_to_index_list, parse_keep_channels

class RefJitterSequence(Sequence):
    def __init__(
        self,
        X, y,
        batch_size,
        ref_modes,
        ref_channel="Cz",
        laplacian=False,
        keep_channels="",
        mu=None, sd=None,
        standardize_mode: str = "train",
        instance_robust: bool = False,
        shuffle=True,
        seed: int = 1,
        randref_alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = int(seed)
        self.epoch = 0
        self.X = X.astype(np.float32, copy=False)   # [N,C,T]
        self.y = y
        self.bs = int(batch_size)
        self.ref_modes = [m.strip().lower() for m in ref_modes if m.strip()]

        # This mode changes channel count; we do not support mixing it in jitter-mix.
        if any(m in ("bipolar_edges", "bip_edges", "edges_bipolar") for m in self.ref_modes):
            raise ValueError(
                "RefJitterSequence does not support bipolar_edges because it changes channel count. "
                "Run bipolar_edges in single-mode training/eval only."
            )
        self.ref_channel = ref_channel
        self.shuffle = shuffle
        self.rng = np.random.default_rng(self.seed)
        self.randref_alpha = float(randref_alpha)

        # keep_names in CURRENT order
        # keep_names = [c.strip() for c in keep_channels.split(",") if c.strip()] or None
        # self.current_names = keep_names if keep_names is not None else BCI2A_CH_NAMES

        # keep_names in CURRENT order
        keep_idx = parse_keep_channels(keep_channels, all_names=BCI2A_CH_NAMES)
        keep_names = None if keep_idx is None else [BCI2A_CH_NAMES[i] for i in keep_idx]
        self.current_names = keep_names if keep_names is not None else BCI2A_CH_NAMES

        need_lap = laplacian or any(
            m in (
                "laplacian", "lap", "local",
                "bipolar", "bip", "bipolar_like",
                "bipolar_edges", "bip_edges", "edges_bipolar",
            )
            for m in self.ref_modes
        )
        self.lap_neighbors = (
            neighbors_to_index_list(
                all_names=BCI2A_CH_NAMES,
                keep_names=self.current_names,
                sort_by_distance=True,
            )
            if need_lap
            else None
        )

        # map ref_channel to index in current channel list
        # needed for:
        #   - mode='ref'
        #   - mode='bipolar' root selection (tree + 2-cycle)
        self.ref_idx = None
        modes_lower = [m.lower() for m in self.ref_modes]
        need_root = any(
            mm in ("ref", "cz_ref", "channel_ref", "bipolar", "bip", "bipolar_like")
            for mm in modes_lower
        )
        if need_root:
            name_to_i = {n: i for i, n in enumerate(self.current_names)}
            if ref_channel not in name_to_i:
                raise ValueError(f"ref_channel '{ref_channel}' not in current channel set")
            self.ref_idx = int(name_to_i[ref_channel])

        # Optional standardization applied AFTER reference transform.
        # - train: use provided mu/sd (computed on training split)
        # - instance: per-trial standardization over time
        # - none: no standardization
        self.standardize_mode = (standardize_mode or "train").lower()
        if self.standardize_mode not in ("train", "instance", "none"):
            raise ValueError("standardize_mode must be one of: train, instance, none")
        self.instance_robust = bool(instance_robust)
        self.mu = mu
        self.sd = sd

        self.idx = np.arange(len(self.y))
        if self.shuffle:
            self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def on_epoch_end(self):
        self.epoch += 1
        if self.shuffle:
            self.rng.shuffle(self.idx)

    def _pick_mode(self, sample_index: int) -> str:
        # Deterministic “random” choice that depends only on seed, epoch, and sample index
        h = (sample_index * 1103515245 + self.epoch * 12345 + self.seed) & 0x7fffffff
        return self.ref_modes[h % len(self.ref_modes)]

    def _sample_rng(self, sample_index: int) -> np.random.Generator:
        """Deterministic RNG per (seed, epoch, sample_index) for randref."""
        h = (sample_index * 2654435761 + self.epoch * 97531 + self.seed * 1009) & 0xFFFFFFFF
        return np.random.default_rng(int(h))

    def __getitem__(self, k):
        b = self.idx[k*self.bs:(k+1)*self.bs]
        Xb = self.X[b]   # [B,C,T]
        yb = self.y[b]

        out = np.empty_like(Xb, dtype=np.float32)
        for i in range(len(b)):
            m = self._pick_mode(int(b[i]))
            rng = self._sample_rng(int(b[i])) if m in ("randref", "random_ref", "random_global_ref") else None
            xi = apply_reference(
                Xb[i:i+1], mode=m,
                ref_idx=self.ref_idx,
                lap_neighbors=self.lap_neighbors,
                rng=rng,
                randref_alpha=self.randref_alpha,
            )[0]
            if self.standardize_mode == "instance":
                xi = standardize_instance(xi, robust=self.instance_robust)
            elif self.standardize_mode == "train" and self.mu is not None:
                # mu,sd are [1,C,1]
                xi = (xi - self.mu[0, :, 0:1]) / self.sd[0, :, 0:1]
            out[i] = xi

        # model expects [B,1,C,T]
        return out[:, None, :, :], yb