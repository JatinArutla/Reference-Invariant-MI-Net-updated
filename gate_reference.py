"""gate_reference.py

Reference-invariance gate tests for MI-Net.

What it does
- Train a supervised ATCNet model under a chosen training reference scheme.
- Evaluate the same trained weights under multiple test reference schemes.
- Optional stronger baselines:
  - mixed-reference training (concatenation of multiple reference modes)
  - reference-jitter training (per-sample re-referencing within a batch)

Paper-clean fixes integrated here
- One stratified train/val split per subject, reused across all conditions.
- Checkpoint selection monitors the right metric:
  - single-mode: val_accuracy
  - mix/jitter: val_refmean_acc (equal-weight mean across test modes)
- Jitter/mix scalers are fit on the training split only (no leakage).
"""

import os
import argparse
import json
import glob
from typing import Dict, List, Tuple

# Determinism knobs (must be set before TF import)
os.environ["TF_DISABLE_LAYOUT_OPTIMIZER"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf

tf.keras.backend.set_image_data_format("channels_last")
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, cohen_kappa_score

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence as KSequence

from src.datamodules.bci2a import load_LOSO_pool, load_subject_dependent
from src.datamodules.transforms import fit_standardizer, apply_standardizer, apply_reference
from src.datamodules.channels import BCI2A_CH_NAMES, parse_keep_channels, neighbors_to_index_list, name_to_index
from src.datamodules.ref_jitter import RefJitterSequence
from src.models.model import build_atcnet


def set_seed(seed: int = 1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _reshape_for_model(X: np.ndarray) -> np.ndarray:
    """[N,C,T] -> [N,1,C,T]"""
    N, C, T = X.shape
    return X.reshape(N, 1, C, T).astype(np.float32, copy=False)


def build_model(args) -> tf.keras.Model:
    return build_atcnet(
        n_classes=args.n_classes,
        in_chans=args.n_channels,
        in_samples=args.in_samples,
        n_windows=args.n_windows,
        attention=args.attention,
        eegn_F1=args.eegn_F1,
        eegn_D=args.eegn_D,
        eegn_kernel=args.eegn_kernel,
        eegn_pool=args.eegn_pool,
        eegn_dropout=args.eegn_dropout,
        tcn_depth=args.tcn_depth,
        tcn_kernel=args.tcn_kernel,
        tcn_filters=args.tcn_filters,
        tcn_dropout=args.tcn_dropout,
        tcn_activation=args.tcn_activation,
        fuse=args.fuse,
        from_logits=args.from_logits,
        return_ssl_feat=False,
    )



def _resolve_ssl_path(ssl_template: str, subject: int) -> str:
    """Resolve a subject-specific SSL weights path.

    Supports:
    - a direct file path
    - a template with '{sub}' or '{subject}' (format())
    - a directory path (we pick the latest matching '*sub{subject}_epoch*.weights.h5')
    """
    if not ssl_template:
        return ""
    try:
        path = ssl_template.format(sub=subject, subject=subject)
    except Exception:
        path = ssl_template

    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, f"*sub{subject}_epoch*.weights.h5")))
        if not cands:
            cands = sorted(glob.glob(os.path.join(path, "*.weights.h5")))
        return cands[-1] if cands else ""

    return path if os.path.exists(path) else ""


def _maybe_load_ssl(model: tf.keras.Model, ssl_template: str, subject: int) -> str:
    """Load SSL-pretrained weights into `model` if provided. Returns resolved path (or '')."""
    wpath = _resolve_ssl_path(ssl_template, subject)
    if not wpath:
        return ""
    # model.load_weights(wpath, skip_mismatch=True)
    model.load_weights(wpath)
    print('SSL weights loaded - _maybe_load_ssl')
    return wpath

def _load_train_and_test(
    args,
    sub: int,
    *,
    ref_mode_train: str,
    ref_mode_test: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return (X_train,y_train), (X_test,y_test).

    - EA applied if args.ea
    - NO standardization (we do it outside so test modes share train stats)
    """
    if args.loso:
        (X_src_tr, y_src_tr), _ = load_LOSO_pool(
            args.data_root,
            sub,
            n_sub=args.n_sub,
            ea=args.ea,
            standardize=False,
            per_block_standardize=args.per_block_standardize,
            t1_sec=args.t1_sec,
            t2_sec=args.t2_sec,
            ref_mode=ref_mode_train,
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=args.laplacian,
        )
        (_, _), (X_tgt_te, y_tgt_te) = load_LOSO_pool(
            args.data_root,
            sub,
            n_sub=args.n_sub,
            ea=args.ea,
            standardize=False,
            per_block_standardize=args.per_block_standardize,
            t1_sec=args.t1_sec,
            t2_sec=args.t2_sec,
            ref_mode=ref_mode_test,
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=args.laplacian,
        )
        return (X_src_tr, y_src_tr), (X_tgt_te, y_tgt_te)

    (X_tr, y_tr), _ = load_subject_dependent(
        args.data_root,
        sub,
        ea=args.ea,
        standardize=False,
        t1_sec=args.t1_sec,
        t2_sec=args.t2_sec,
        ref_mode=ref_mode_train,
        keep_channels=args.keep_channels,
        ref_channel=args.ref_channel,
        laplacian=args.laplacian,
    )
    (_, _), (X_te, y_te) = load_subject_dependent(
        args.data_root,
        sub,
        ea=args.ea,
        standardize=False,
        t1_sec=args.t1_sec,
        t2_sec=args.t2_sec,
        ref_mode=ref_mode_test,
        keep_channels=args.keep_channels,
        ref_channel=args.ref_channel,
        laplacian=args.laplacian,
    )
    return (X_tr, y_tr), (X_te, y_te)


def _load_test_only(args, sub: int, *, ref_mode_test: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the test block only (no standardization)."""
    if args.loso:
        (_, _), (X_tgt, y_tgt) = load_LOSO_pool(
            args.data_root,
            sub,
            n_sub=args.n_sub,
            ea=args.ea,
            standardize=False,
            per_block_standardize=args.per_block_standardize,
            t1_sec=args.t1_sec,
            t2_sec=args.t2_sec,
            ref_mode=ref_mode_test,
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=args.laplacian,
        )
        return X_tgt, y_tgt

    (_, _), (X_te, y_te) = load_subject_dependent(
        args.data_root,
        sub,
        ea=args.ea,
        standardize=False,
        t1_sec=args.t1_sec,
        t2_sec=args.t2_sec,
        ref_mode=ref_mode_test,
        keep_channels=args.keep_channels,
        ref_channel=args.ref_channel,
        laplacian=args.laplacian,
    )
    return X_te, y_te


def _make_split_indices(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=int(seed))
    tr_idx, va_idx = next(sss.split(np.zeros_like(y), y))
    return tr_idx.astype(int), va_idx.astype(int)


def _current_channel_names(keep_channels: str | None) -> List[str]:
    keep_idx = parse_keep_channels(keep_channels, all_names=BCI2A_CH_NAMES)
    if keep_idx is None:
        return list(BCI2A_CH_NAMES)
    return [BCI2A_CH_NAMES[i] for i in keep_idx]


def _ref_params_for_modes(args, modes: List[str]):
    cur = _current_channel_names(args.keep_channels)

    need_ref = any((m or "").lower() in ("ref", "cz_ref", "channel_ref") for m in modes)
    need_lap = any((m or "").lower() in ("laplacian", "lap", "local") for m in modes)

    ref_idx = None
    if need_ref:
        m = name_to_index(cur)
        if args.ref_channel not in m:
            raise ValueError(f"ref_channel '{args.ref_channel}' not in channels: {cur}")
        ref_idx = m[args.ref_channel]

    lap_neighbors = None
    if need_lap:
        # neighbors projected onto current channel ordering
        lap_neighbors = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=cur)

    return ref_idx, lap_neighbors


class ValRefMeanAccFromInputs(tf.keras.callbacks.Callback):
    """Mean accuracy across modes using the same validation trials."""

    def __init__(self, val_inputs_by_mode: Dict[str, np.ndarray], y_val_onehot: np.ndarray, from_logits: bool, metric_name: str = "val_refmean_acc"):
        super().__init__()
        self.val_inputs_by_mode = val_inputs_by_mode
        self.y_true = y_val_onehot.argmax(-1).astype(int)
        self.from_logits = bool(from_logits)
        self.metric_name = metric_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accs = []
        for Xm in self.val_inputs_by_mode.values():
            y_pred = self.model.predict(Xm, verbose=0)
            if self.from_logits:
                y_hat = tf.nn.softmax(y_pred).numpy().argmax(-1)
            else:
                y_hat = y_pred.argmax(-1)
            accs.append((y_hat.astype(int) == self.y_true).mean())
        logs[self.metric_name] = float(np.mean(accs)) if accs else 0.0


class ValRefMeanAccFromSubsets(tf.keras.callbacks.Callback):
    """Equal-weight mean accuracy across subsets inside a concatenated val set."""

    def __init__(self, X_val_concat: np.ndarray, y_val_onehot: np.ndarray, ref_ids_val: np.ndarray, n_modes: int, from_logits: bool, metric_name: str = "val_refmean_acc"):
        super().__init__()
        self.Xv = X_val_concat
        self.y_true = y_val_onehot.argmax(-1).astype(int)
        self.ref_ids = np.asarray(ref_ids_val).astype(int)
        self.n_modes = int(n_modes)
        self.from_logits = bool(from_logits)
        self.metric_name = metric_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accs = []
        for mid in range(self.n_modes):
            idx = np.where(self.ref_ids == mid)[0]
            if idx.size == 0:
                continue
            y_pred = self.model.predict(self.Xv[idx], verbose=0)
            if self.from_logits:
                y_hat = tf.nn.softmax(y_pred).numpy().argmax(-1)
            else:
                y_hat = y_pred.argmax(-1)
            accs.append((y_hat.astype(int) == self.y_true[idx]).mean())
        logs[self.metric_name] = float(np.mean(accs)) if accs else 0.0


def train_one(
    args,
    X_tr,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    out_dir: str,
    subject: int,
    *,
    monitor: str,
    extra_cbs=None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.weights.h5")

    model = build_model(args)
    ssl_loaded = _maybe_load_ssl(model, args.ssl_weights, subject)
    if ssl_loaded:
        print(f"   ↳ loaded SSL weights: {ssl_loaded}")
    model.compile(
        loss=CategoricalCrossentropy(from_logits=args.from_logits),
        optimizer=Adam(learning_rate=args.lr),
        metrics=["accuracy"],
    )

    mode = "min" if monitor.endswith("loss") else "max"

    cbs = []
    if extra_cbs:
        cbs.extend(list(extra_cbs))

    # Make sure our custom metric is populated before checkpointing.
    cbs.extend([
        ModelCheckpoint(
            ckpt_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            mode=mode,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.90,
            patience=args.plateau_patience,
            verbose=0,
            min_lr=args.min_lr,
        ),
    ])

    if args.early_stop:
        cbs.append(
            EarlyStopping(
                monitor=monitor,
                patience=args.patience,
                mode=mode,
                restore_best_weights=False,
                verbose=0,
            )
        )

    if isinstance(X_tr, KSequence):
        # Keras 3 / TF 2.16+: `workers`, `use_multiprocessing`, `max_queue_size` are not
        # accepted by `fit()`. Sequence iteration is single-process by default.
        model.fit(
            X_tr,
            validation_data=(_reshape_for_model(X_va), y_va),
            epochs=args.epochs,
            verbose=0,
            callbacks=cbs,
        )
    else:
        model.fit(
            _reshape_for_model(X_tr),
            y_tr,
            validation_data=(_reshape_for_model(X_va), y_va),
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=False,
            verbose=0,
            callbacks=cbs,
        )

    return ckpt_path


def eval_one(args, weights_path: str, X_te: np.ndarray, y_te: np.ndarray) -> Tuple[float, float]:
    model = build_model(args)
    model.load_weights(weights_path)
    y_pred = model.predict(_reshape_for_model(X_te), verbose=0)
    y_hat = y_pred.argmax(-1) if not args.from_logits else tf.nn.softmax(y_pred).numpy().argmax(-1)
    acc = accuracy_score(y_te.astype(int), y_hat.astype(int))
    kappa = cohen_kappa_score(y_te.astype(int), y_hat.astype(int))
    return float(acc), float(kappa)


def run(args):
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    test_modes = [m.strip() for m in args.test_ref_modes.split(",") if m.strip()]
    if not test_modes:
        raise ValueError("--test_ref_modes must be a comma-separated list (e.g., 'native,car,ref,laplacian').")

    train_modes = [m.strip() for m in args.train_ref_modes.split(",") if m.strip()]
    if not train_modes:
        raise ValueError("--train_ref_modes must be non-empty.")

    if args.mix_train_refs and args.jitter_train_refs:
        raise ValueError("Pick only one of --mix_train_refs or --jitter_train_refs.")

    train_conds = ["mix"] if args.mix_train_refs else (["jitter"] if args.jitter_train_refs else train_modes)

    # parse jitter modes (if empty, reuse train_modes)
    jitter_modes = [m.strip() for m in args.jitter_ref_modes.split(",") if m.strip()] if args.jitter_ref_modes else train_modes

    # output containers
    results: Dict[str, Dict[str, List[float]]] = {c: {tm: [] for tm in test_modes} for c in train_conds}

    # ref params for in-place apply_reference used by jitter scaler/val
    ref_idx, lap_neighbors = _ref_params_for_modes(args, list({*test_modes, *jitter_modes}))

    for sub in range(1, args.n_sub + 1):
        # One split per subject (reused across all conditions)
        (X_base, y_base), _ = _load_train_and_test(args, sub, ref_mode_train="native", ref_mode_test="native")

        args.n_channels = int(X_base.shape[1])
        args.in_samples = int(X_base.shape[2])

        tr_idx, va_idx = _make_split_indices(y_base, args.val_ratio, args.seed)

        sub_dir = os.path.join(args.results_dir, f"sub_{sub:02d}")
        os.makedirs(sub_dir, exist_ok=True)
        with open(os.path.join(sub_dir, "splits.json"), "w") as f:
            json.dump({"train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist()}, f, indent=2)

        y_tr_oh = to_categorical(y_base[tr_idx], num_classes=args.n_classes)
        y_va_oh = to_categorical(y_base[va_idx], num_classes=args.n_classes)

        for cond in train_conds:
            mu_sd = None
            extra_cbs = None
            monitor = "val_accuracy"

            if cond == "jitter":
                # Train/val are slices of native base
                X_tr_raw = X_base[tr_idx]
                X_va_raw = X_base[va_idx]

                # Fit scaler on mixed-ref pool from TRAIN split only (no leakage)
                if args.standardize:
                    X_stats = np.concatenate(
                        [apply_reference(X_tr_raw, mode=m, ref_idx=ref_idx, lap_neighbors=lap_neighbors) for m in jitter_modes],
                        axis=0,
                    )
                    mu_sd = fit_standardizer(X_stats)

                mu, sd = mu_sd if mu_sd is not None else (None, None)

                train_in = RefJitterSequence(
                    X_tr_raw,
                    y_tr_oh,
                    batch_size=args.batch_size,
                    ref_modes=jitter_modes,
                    ref_channel=args.ref_channel,
                    laplacian=args.laplacian,
                    keep_channels=args.keep_channels,
                    mu=mu,
                    sd=sd,
                    shuffle=True,
                    seed=args.seed,
                )

                # Validation data for Keras + extra metric across modes
                if mu_sd is not None:
                    X_va_fit = apply_standardizer(X_va_raw, *mu_sd)
                else:
                    X_va_fit = X_va_raw.astype(np.float32, copy=False)

                val_inputs_by_mode: Dict[str, np.ndarray] = {}
                for m in test_modes:
                    Xv_m = apply_reference(X_va_raw, mode=m, ref_idx=ref_idx, lap_neighbors=lap_neighbors)
                    if mu_sd is not None:
                        Xv_m = apply_standardizer(Xv_m, *mu_sd)
                    val_inputs_by_mode[m] = _reshape_for_model(Xv_m)

                monitor = "val_refmean_acc"
                extra_cbs = [ValRefMeanAccFromInputs(val_inputs_by_mode, y_va_oh, args.from_logits)]

                weights_path = train_one(
                    args,
                    train_in,
                    y_tr_oh,
                    X_va_fit,
                    y_va_oh,
                    os.path.join(sub_dir, f"train_{cond}"),
                    subject=sub,
                    monitor=monitor,
                    extra_cbs=extra_cbs,
                )

            elif cond == "mix":
                X_tr_list, X_va_list = [], []
                ref_ids_va = []

                for mid, m in enumerate(train_modes):
                    (Xm_all, ym_all), _ = _load_train_and_test(args, sub, ref_mode_train=m, ref_mode_test=m)
                    if len(ym_all) != len(y_base):
                        raise RuntimeError(f"Sample count mismatch for mode '{m}' (got {len(ym_all)} vs base {len(y_base)})")
                    if not np.array_equal(ym_all, y_base):
                        raise RuntimeError(f"Label/order mismatch for mode '{m}' vs base. This should not happen.")

                    X_tr_list.append(Xm_all[tr_idx])
                    X_va_list.append(Xm_all[va_idx])
                    ref_ids_va.append(np.full((len(va_idx),), mid, dtype=np.int32))

                X_tr_raw = np.concatenate(X_tr_list, axis=0)
                X_va_raw = np.concatenate(X_va_list, axis=0)

                y_tr_mix = np.concatenate([y_tr_oh for _ in train_modes], axis=0)
                y_va_mix = np.concatenate([y_va_oh for _ in train_modes], axis=0)
                ref_ids_va = np.concatenate(ref_ids_va, axis=0)

                if args.standardize:
                    mu_sd = fit_standardizer(X_tr_raw)
                    X_tr = apply_standardizer(X_tr_raw, *mu_sd)
                    X_va = apply_standardizer(X_va_raw, *mu_sd)
                else:
                    X_tr = X_tr_raw.astype(np.float32, copy=False)
                    X_va = X_va_raw.astype(np.float32, copy=False)

                monitor = "val_refmean_acc"
                # Use reshaped X_va inside callback to avoid reshaping each epoch.
                X_va_4d = _reshape_for_model(X_va)
                extra_cbs = [ValRefMeanAccFromSubsets(X_va_4d, y_va_mix, ref_ids_va, n_modes=len(train_modes), from_logits=args.from_logits)]

                weights_path = train_one(
                    args,
                    X_tr,
                    y_tr_mix,
                    X_va,
                    y_va_mix,
                    os.path.join(sub_dir, f"train_{cond}"),
                    subject=sub,
                    monitor=monitor,
                    extra_cbs=extra_cbs,
                )

            else:
                (Xc_all, yc_all), _ = _load_train_and_test(args, sub, ref_mode_train=cond, ref_mode_test=cond)
                if len(yc_all) != len(y_base):
                    raise RuntimeError(f"Sample count mismatch for mode '{cond}' (got {len(yc_all)} vs base {len(y_base)})")
                if not np.array_equal(yc_all, y_base):
                    raise RuntimeError(f"Label/order mismatch for mode '{cond}' vs base. This should not happen.")

                X_tr_raw = Xc_all[tr_idx]
                X_va_raw = Xc_all[va_idx]

                if args.standardize:
                    mu_sd = fit_standardizer(X_tr_raw)
                    X_tr = apply_standardizer(X_tr_raw, *mu_sd)
                    X_va = apply_standardizer(X_va_raw, *mu_sd)
                else:
                    X_tr = X_tr_raw.astype(np.float32, copy=False)
                    X_va = X_va_raw.astype(np.float32, copy=False)

                weights_path = train_one(
                    args,
                    X_tr,
                    y_tr_oh,
                    X_va,
                    y_va_oh,
                    os.path.join(sub_dir, f"train_{cond}"),
                    subject=sub,
                    monitor="val_accuracy",
                    extra_cbs=None,
                )

            # Evaluate across test reference modes, using the SAME mu_sd (if any)
            y_te_base = None
            for te_mode in test_modes:
                X_te_raw, y_te = _load_test_only(args, sub, ref_mode_test=te_mode)
                if y_te_base is None:
                    y_te_base = y_te
                elif not np.array_equal(y_te, y_te_base):
                    raise RuntimeError(f"Test label/order mismatch for mode '{te_mode}' vs first test mode")

                if mu_sd is not None:
                    X_te = apply_standardizer(X_te_raw, *mu_sd)
                else:
                    X_te = X_te_raw.astype(np.float32, copy=False)

                acc, _ = eval_one(args, weights_path, X_te, y_te)
                results[cond][te_mode].append(acc)
                print(f"sub {sub:02d}  train={cond:<9}  test={te_mode:<9}  acc={acc*100:5.2f}")

        tf.keras.backend.clear_session()

    # summarize
    summary = {c: {te: float(np.mean(results[c][te])) for te in test_modes} for c in train_conds}

    out_json = os.path.join(args.results_dir, "gate_reference_summary.json")
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "per_subject": results}, f, indent=2)

    print("\nAveraged accuracy (%):")
    header = "train\\test".ljust(12) + "".join([m.ljust(12) for m in test_modes])
    print(header)
    print("-" * len(header))
    for cond in train_conds:
        row = cond.ljust(12)
        for te_mode in test_modes:
            row += f"{summary[cond][te_mode]*100:6.2f}".ljust(12)
        print(row)
    print(f"\nSaved: {out_json}")


def parse_args():
    p = argparse.ArgumentParser("Reference-invariance gate test")

    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results_gate")
    p.add_argument("--n_sub", type=int, default=9)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_channels", type=int, default=22)  # overridden dynamically
    p.add_argument("--in_samples", type=int, default=1000)
    p.add_argument("--t1_sec", type=float, default=2.0)
    p.add_argument("--t2_sec", type=float, default=6.0)

    # reference / channel control
    p.add_argument("--train_ref_modes", type=str, default="native", help="Comma-separated train mode(s).")
    p.add_argument("--test_ref_modes", type=str, default="native,car,ref,laplacian", help="Comma-separated test modes.")
    p.add_argument("--mix_train_refs", action="store_true", help="Train on a mixture (concatenation) of all train_ref_modes.")
    p.add_argument("--jitter_train_refs", action="store_true", help="Train with per-sample reference jitter (no concatenation).")
    p.add_argument("--jitter_ref_modes", type=str, default="", help="Comma-separated modes used for jitter. If empty, uses train_ref_modes.")
    p.add_argument("--keep_channels", type=str, default="", help="Comma-separated channel names to keep (intersection baseline).")
    p.add_argument("--ref_channel", type=str, default="Cz", help="Channel for ref-mode re-referencing.")
    p.add_argument("--laplacian", action="store_true", help="Enable Laplacian neighbor graph.")

    # preprocessing
    p.add_argument("--ea", action="store_true")
    p.add_argument("--no-ea", dest="ea", action="store_false")
    p.set_defaults(ea=True)

    p.add_argument("--standardize", action="store_true")
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    p.set_defaults(standardize=True)

    p.add_argument("--per_block_standardize", action="store_true")
    p.add_argument("--no-per_block_standardize", dest="per_block_standardize", action="store_false")
    p.set_defaults(per_block_standardize=True)

    p.add_argument("--loso", action="store_true")
    p.add_argument("--no-loso", dest="loso", action="store_false")
    p.set_defaults(loso=True)

    # model
    p.add_argument("--n_windows", type=int, default=5)
    p.add_argument("--attention", type=str, default="mha", choices=["mha", "mhla", "none", ""])
    p.add_argument("--eegn_F1", type=int, default=16)
    p.add_argument("--eegn_D", type=int, default=2)
    p.add_argument("--eegn_kernel", type=int, default=64)
    p.add_argument("--eegn_pool", type=int, default=7)
    p.add_argument("--eegn_dropout", type=float, default=0.3)
    p.add_argument("--tcn_depth", type=int, default=2)
    p.add_argument("--tcn_kernel", type=int, default=4)
    p.add_argument("--tcn_filters", type=int, default=32)
    p.add_argument("--tcn_dropout", type=float, default=0.3)
    p.add_argument("--tcn_activation", type=str, default="elu")
    p.add_argument("--fuse", type=str, default="average", choices=["average", "concat"])
    p.add_argument("--from_logits", action="store_true")

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--plateau_patience", type=int, default=15)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--seed", type=int, default=1)

    # SSL init (optional)
    p.add_argument(
        "--ssl_weights",
        type=str,
        default="",
        help="Path/template/dir for SSL encoder weights. Example: './results_ssl/LOSO_{sub:02d}' or './results_ssl/LOSO_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5'",
    )

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
