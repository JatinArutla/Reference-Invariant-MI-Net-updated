import numpy as np
from tensorflow.keras.utils import Sequence

from src.datamodules.transforms import apply_reference
from src.datamodules.channels import BCI2A_CH_NAMES, neighbors_to_index_list

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
        shuffle=True,
        seed: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seed = int(seed)
        self.epoch = 0
        self.X = X.astype(np.float32, copy=False)   # [N,C,T]
        self.y = y
        self.bs = int(batch_size)
        self.ref_modes = [m.strip().lower() for m in ref_modes if m.strip()]
        self.ref_channel = ref_channel
        self.shuffle = shuffle
        self.rng = np.random.default_rng(self.seed)

        # keep_names in CURRENT order
        keep_names = [c.strip() for c in keep_channels.split(",") if c.strip()] or None
        self.current_names = keep_names if keep_names is not None else BCI2A_CH_NAMES

        need_lap = laplacian or any(m in ("laplacian", "lap", "local") for m in self.ref_modes)
        self.lap_neighbors = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=self.current_names) if need_lap else None

        # map ref_channel to index in current channel list
        self.ref_idx = None
        if "ref" in self.ref_modes:
            name_to_i = {n: i for i, n in enumerate(self.current_names)}
            if ref_channel not in name_to_i:
                raise ValueError(f"ref_channel '{ref_channel}' not in current channel set")
            self.ref_idx = name_to_i[ref_channel]

        # optional standardization applied AFTER reference transform
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

    def __getitem__(self, k):
        b = self.idx[k*self.bs:(k+1)*self.bs]
        Xb = self.X[b]   # [B,C,T]
        yb = self.y[b]

        out = np.empty_like(Xb, dtype=np.float32)
        for i in range(len(b)):
            m = self._pick_mode(int(b[i]))
            xi = apply_reference(
                Xb[i:i+1], mode=m,
                ref_idx=self.ref_idx,
                lap_neighbors=self.lap_neighbors,
            )[0]
            if self.mu is not None:
                # mu,sd are [1,C,1]
                xi = (xi - self.mu[0, :, 0:1]) / self.sd[0, :, 0:1]
            out[i] = xi

        # model expects [B,1,C,T]
        return out[:, None, :, :], yb