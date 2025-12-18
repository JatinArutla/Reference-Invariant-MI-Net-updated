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
        ea_ws=None,
        group_ids=None,
        mu=None, sd=None,
        shuffle=True,
        seed=1,
    ):
        self.X = X.astype(np.float32, copy=False)   # [N,C,T]
        self.y = y
        self.bs = int(batch_size)
        self.ref_modes = [m.strip().lower() for m in ref_modes if m.strip()]
        self.ref_channel = ref_channel
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        # keep_names in CURRENT order
        keep_names = [c.strip() for c in keep_channels.split(",") if c.strip()] or None
        self.current_names = keep_names if keep_names is not None else BCI2A_CH_NAMES

        # laplacian neighbors aligned to current channel order
        self.lap_neighbors = None
        if laplacian:
            self.lap_neighbors = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=self.current_names)

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

        # optional group id per sample (e.g., subject id in LOSO pooling)
        self.group_ids = None
        if group_ids is not None:
            gids = np.asarray(group_ids)
            if gids.shape[0] != self.X.shape[0]:
                raise ValueError(f"group_ids length {gids.shape[0]} != N {self.X.shape[0]}")
            self.group_ids = gids

        # optional EA matrices per mode: {mode: W} with W [C,C]
        # If provided, EA is applied AFTER reference transform.
        self.ea_ws = None
        if ea_ws is not None:
            # ea_ws can be either:
            #  - {mode: W} where W is [C,C]
            #  - {mode: {group_id: W}} for pooled settings (e.g., LOSO)
            norm = {}
            for k, v in dict(ea_ws).items():
                mk = str(k).lower()
                if isinstance(v, dict):
                    norm[mk] = {int(g): W.astype(np.float32) for g, W in v.items()}
                else:
                    norm[mk] = np.asarray(v, dtype=np.float32)
            self.ea_ws = norm

        self.idx = np.arange(len(self.y), dtype=np.int64)
        self._rng = np.random.RandomState(self.seed)
        if self.shuffle:
            self._rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def on_epoch_end(self):
        if self.shuffle:
            self.epoch += 1
            self._rng = np.random.RandomState(self.seed + self.epoch)
            self._rng.shuffle(self.idx)

    def _pick_mode(self, sample_id: int) -> str:
        """Deterministic per-sample mode selection.

        Ensures that running the same command yields identical batches.
        The mapping changes across epochs if shuffle=True due to idx shuffling.
        """
        m = len(self.ref_modes)
        if m == 1:
            return self.ref_modes[0]
        # simple hash based on sample_id and seed
        j = (int(sample_id) * 1315423911 + self.seed * 2654435761) & 0x7FFFFFFF
        return self.ref_modes[j % m]

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

            if self.ea_ws is not None and m in self.ea_ws:
                Wm = self.ea_ws[m]
                if isinstance(Wm, dict):
                    if self.group_ids is None:
                        raise ValueError("ea_ws provides per-group matrices but group_ids is None")
                    gid = int(self.group_ids[int(b[i])])
                    if gid in Wm:
                        xi = (Wm[gid] @ xi).astype(np.float32)
                else:
                    xi = (Wm @ xi).astype(np.float32)

            if self.mu is not None:
                # mu,sd are [1,C,1]
                xi = (xi - self.mu[0, :, 0:1]) / self.sd[0, :, 0:1]
            out[i] = xi

        # model expects [B,1,C,T]
        return out[:, None, :, :], yb