"""gate_reference.py

Paper-clean reference-invariance gate test.

Key properties vs the quick-and-dirty version:
  - One fixed train/val split per subject (same underlying trials) reused across
    *all* training conditions.
  - Test sets are also identical across reference modes: we load once in native
    and apply the reference transforms ourselves.
  - When --jitter_train_refs is used with --ea, jitter applies EA in a
    mode-consistent way (reference -> EA) using pre-fit EA matrices per mode.
  - Model selection is done by checkpointing:
      * single-mode training: monitor val_accuracy
      * mix/jitter training: monitor val_refmean_acc (mean val accuracy across
        the requested modes on the same val trials)

This script is deterministic given --seed. It also forces deterministic TF ops
and single-threaded execution.
"""

import os
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

os.environ["TF_DISABLE_LAYOUT_OPTIMIZER"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf

tf.keras.backend.set_image_data_format("channels_last")
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence as KSequence

from src.datamodules.bci2a import load_bci2a_session
from src.datamodules.transforms import (
    apply_reference,
    fit_standardizer,
    apply_standardizer,
    ea_fit,
    ea_apply,
    ea_align_trials,
)
from src.datamodules.channels import BCI2A_CH_NAMES, parse_keep_channels, neighbors_to_index_list, name_to_index
from src.models.model import build_atcnet
from src.datamodules.ref_jitter import RefJitterSequence


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


class ValRefMeanAccCallback(tf.keras.callbacks.Callback):
    """Compute mean validation accuracy across reference modes on the same val trials."""

    def __init__(self, val_by_mode: Dict[str, np.ndarray], y_val_oh: np.ndarray, from_logits: bool, name: str = "val_refmean_acc"):
        super().__init__()
        self.val_by_mode = val_by_mode
        self.y_true = y_val_oh.argmax(-1).astype(int)
        self.from_logits = bool(from_logits)
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accs = []
        for Xm in self.val_by_mode.values():
            y_pred = self.model.predict(Xm, verbose=0)
            if self.from_logits:
                y_hat = tf.nn.softmax(y_pred).numpy().argmax(-1)
            else:
                y_hat = y_pred.argmax(-1)
            accs.append((y_hat.astype(int) == self.y_true).mean())
        logs[self.name] = float(np.mean(accs)) if accs else 0.0


@dataclass
class RefParams:
    current_names: List[str]
    ref_idx: Optional[int]
    lap_neighbors: Optional[List[List[int]]]


def _make_ref_params(args) -> RefParams:
    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    current_names = [BCI2A_CH_NAMES[i] for i in keep_idx] if keep_idx is not None else list(BCI2A_CH_NAMES)

    ref_idx = None
    # compute ref_idx only if needed at runtime; still precompute safely
    if args.ref_channel:
        m = name_to_index(current_names)
        if args.ref_channel not in m:
            raise ValueError(f"ref_channel '{args.ref_channel}' not in current channel set")
        ref_idx = m[args.ref_channel]

    lap_neighbors = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=current_names)
    return RefParams(current_names=current_names, ref_idx=ref_idx, lap_neighbors=lap_neighbors)


def _apply_ref_mode(X: np.ndarray, mode: str, refp: RefParams) -> np.ndarray:
    m = (mode or "native").lower()
    if m in ("ref", "cz_ref", "channel_ref"):
        return apply_reference(X, mode=m, ref_idx=refp.ref_idx, lap_neighbors=None)
    if m in ("laplacian", "lap", "local"):
        return apply_reference(X, mode=m, ref_idx=None, lap_neighbors=refp.lap_neighbors)
    return apply_reference(X, mode=m, ref_idx=None, lap_neighbors=None)


def _fit_split_indices(y: np.ndarray, *, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y), dtype=np.int64)
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=y.astype(int),
    )
    # stable ordering for deterministic batching
    return np.sort(tr_idx), np.sort(va_idx)


def _save_splits(path: str, train_idx: np.ndarray, val_idx: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, f, indent=2)


def _load_sd_base(args, sub: int, refp: RefParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load once in native (no EA, no standardization). We will apply ref/EA/standardize ourselves.
    X_tr, y_tr = load_bci2a_session(
        args.data_root, sub, True,
        t1_sec=args.t1_sec, t2_sec=args.t2_sec,
        ref_mode="native",
        keep_channels=args.keep_channels,
        ref_channel=args.ref_channel,
        laplacian=False,
    )
    X_te, y_te = load_bci2a_session(
        args.data_root, sub, False,
        t1_sec=args.t1_sec, t2_sec=args.t2_sec,
        ref_mode="native",
        keep_channels=args.keep_channels,
        ref_channel=args.ref_channel,
        laplacian=False,
    )
    return X_tr.astype(np.float32), y_tr.astype(np.int64), X_te.astype(np.float32), y_te.astype(np.int64)


def _load_loso_blocks_base(args, refp: RefParams) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    blocks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for s in range(1, args.n_sub + 1):
        X1, y1 = load_bci2a_session(
            args.data_root, s, True,
            t1_sec=args.t1_sec, t2_sec=args.t2_sec,
            ref_mode="native",
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=False,
        )
        X2, y2 = load_bci2a_session(
            args.data_root, s, False,
            t1_sec=args.t1_sec, t2_sec=args.t2_sec,
            ref_mode="native",
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=False,
        )
        X = np.concatenate([X1, X2], axis=0).astype(np.float32)
        y = np.concatenate([y1, y2], axis=0).astype(np.int64)
        blocks[s] = (X, y)
    return blocks


def _build_loso_src_pool(blocks_base: Dict[int, Tuple[np.ndarray, np.ndarray]], target_sub: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    src_keys = [k for k in sorted(blocks_base.keys()) if k != target_sub]
    X_src = np.concatenate([blocks_base[k][0] for k in src_keys], axis=0)
    y_src = np.concatenate([blocks_base[k][1] for k in src_keys], axis=0)
    return X_src, y_src, src_keys


def _apply_ref_ea_block(X_base: np.ndarray, mode: str, refp: RefParams, ea: bool) -> np.ndarray:
    X = _apply_ref_mode(X_base, mode, refp)
    if ea:
        X = ea_align_trials(X)
    return X


def _train_one(
    args,
    X_tr,
    y_tr_oh: np.ndarray,
    X_val_for_loss: np.ndarray,
    y_val_for_loss: np.ndarray,
    out_dir: str,
    *,
    monitor: str,
    extra_cbs: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.weights.h5")

    model = build_model(args)
    model.compile(
        loss=CategoricalCrossentropy(from_logits=args.from_logits),
        optimizer=Adam(learning_rate=args.lr),
        metrics=["accuracy"],
    )

    mode = "min" if monitor.endswith("loss") else "max"
    cbs: List[tf.keras.callbacks.Callback] = []
    if extra_cbs:
        cbs.extend(extra_cbs)
    cbs.extend([
        ModelCheckpoint(ckpt_path, monitor=monitor, save_best_only=True, save_weights_only=True, mode=mode, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=args.plateau_patience, verbose=0, min_lr=args.min_lr),
    ])
    if args.early_stop:
        cbs.append(EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", restore_best_weights=False, verbose=0))

    if isinstance(X_tr, KSequence):
        model.fit(
            X_tr,
            validation_data=(_reshape_for_model(X_val_for_loss), y_val_for_loss),
            epochs=args.epochs,
            verbose=0,
            callbacks=cbs,
            workers=0,
            use_multiprocessing=False,
            max_queue_size=1,
        )
    else:
        model.fit(
            _reshape_for_model(X_tr), y_tr_oh,
            validation_data=(_reshape_for_model(X_val_for_loss), y_val_for_loss),
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=False,
            verbose=0,
            callbacks=cbs,
        )
    return ckpt_path


def _eval_one(args, weights_path: str, X_te: np.ndarray, y_te: np.ndarray) -> Tuple[float, float]:
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

    refp = _make_ref_params(args)

    test_modes = [m.strip().lower() for m in args.test_ref_modes.split(",") if m.strip()]
    train_modes = [m.strip().lower() for m in args.train_ref_modes.split(",") if m.strip()]
    if not test_modes:
        raise ValueError("--test_ref_modes must be non-empty")
    if not train_modes:
        raise ValueError("--train_ref_modes must be non-empty")
    if args.mix_train_refs and args.jitter_train_refs:
        raise ValueError("Pick only one of --mix_train_refs or --jitter_train_refs")

    train_conds = ["mix"] if args.mix_train_refs else (["jitter"] if args.jitter_train_refs else train_modes)
    jitter_modes = [m.strip().lower() for m in args.jitter_ref_modes.split(",") if m.strip()] if args.jitter_ref_modes else train_modes

    # containers
    results: Dict[str, Dict[str, List[float]]] = {c: {tm: [] for tm in test_modes} for c in train_conds}

    # cache LOSO base blocks if needed
    loso_blocks_base = _load_loso_blocks_base(args, refp) if args.loso else None

    for sub in range(1, args.n_sub + 1):
        if args.loso:
            assert loso_blocks_base is not None
            # Build pooled *native* sources once for splitting, plus group ids.
            src_keys = [k for k in sorted(loso_blocks_base.keys()) if k != sub]
            X_src_base = np.concatenate([loso_blocks_base[k][0] for k in src_keys], axis=0)
            y_src_base = np.concatenate([loso_blocks_base[k][1] for k in src_keys], axis=0)
            group_ids = np.concatenate([np.full(len(loso_blocks_base[k][1]), k, dtype=np.int64) for k in src_keys], axis=0)

            X_tgt_base, y_tgt_base = loso_blocks_base[sub]

            args.n_channels = int(X_src_base.shape[1])
            args.in_samples = int(X_src_base.shape[2])

            tr_idx, va_idx = _fit_split_indices(y_src_base, val_ratio=args.val_ratio, seed=args.seed)
            split_path = os.path.join(args.results_dir, f"sub_{sub:02d}", "splits.json")
            _save_splits(split_path, tr_idx, va_idx)

            # For LOSO, EA must be applied per subject block (not on the pooled sources).
            def build_src_mode(mode: str, *, need_W: bool = False):
                X_blocks = []
                W_by_group = {} if need_W else None
                for s in src_keys:
                    Xs_ref = _apply_ref_mode(loso_blocks_base[s][0], mode, refp)
                    if args.ea:
                        W = ea_fit(Xs_ref)
                        Xs = ea_apply(Xs_ref, W)
                        if need_W:
                            W_by_group[int(s)] = W
                    else:
                        Xs = Xs_ref
                    X_blocks.append(Xs)
                return np.concatenate(X_blocks, axis=0), W_by_group

            def build_tgt_mode(mode: str):
                Xt_ref = _apply_ref_mode(X_tgt_base, mode, refp)
                if args.ea:
                    Xt_ref = ea_align_trials(Xt_ref)
                return Xt_ref

            base_for_train = X_src_base
            y_full = y_src_base
            base_for_test = X_tgt_base
            y_test = y_tgt_base
        else:
            X_tr_base, y_tr_base, X_te_base, y_te_base = _load_sd_base(args, sub, refp)
            args.n_channels = int(X_tr_base.shape[1])
            args.in_samples = int(X_tr_base.shape[2])

            tr_idx, va_idx = _fit_split_indices(y_tr_base, val_ratio=args.val_ratio, seed=args.seed)
            split_path = os.path.join(args.results_dir, f"sub_{sub:02d}", "splits.json")
            _save_splits(split_path, tr_idx, va_idx)

            base_for_train = X_tr_base
            y_full = y_tr_base
            base_for_test = X_te_base
            y_test = y_te_base

        y_full_oh = to_categorical(y_full, num_classes=args.n_classes)

        for cond in train_conds:
            # -----------------------------
            # Build train/val tensors
            # -----------------------------
            extra_cbs = None
            monitor = "val_accuracy"

            if cond in train_modes:
                # single reference mode training
                if args.loso:
                    X_full_mode, _ = build_src_mode(cond, need_W=False)
                else:
                    X_full_mode = _apply_ref_ea_block(base_for_train, cond, refp, ea=args.ea)
                X_tr_m = X_full_mode[tr_idx]
                X_va_m = X_full_mode[va_idx]
                y_tr = y_full_oh[tr_idx]
                y_va = y_full_oh[va_idx]

                mu_sd = None
                if args.standardize:
                    mu_sd = fit_standardizer(X_tr_m)
                    X_tr_m = apply_standardizer(X_tr_m, *mu_sd)
                    X_va_m = apply_standardizer(X_va_m, *mu_sd)

                train_in = X_tr_m
                X_val_for_loss = X_va_m
                y_val_for_loss = y_va

            elif cond == "mix":
                # concatenation baseline over train_modes
                X_tr_list, X_va_list = [], []
                for m in train_modes:
                    if args.loso:
                        X_full_m, _ = build_src_mode(m, need_W=False)
                    else:
                        X_full_m = _apply_ref_ea_block(base_for_train, m, refp, ea=args.ea)
                    X_tr_list.append(X_full_m[tr_idx])
                    X_va_list.append(X_full_m[va_idx])

                y_tr_base = y_full_oh[tr_idx]
                y_va_base = y_full_oh[va_idx]

                X_tr_mix = np.concatenate(X_tr_list, axis=0)
                y_tr_mix = np.concatenate([y_tr_base] * len(train_modes), axis=0)
                X_va_mix = np.concatenate(X_va_list, axis=0)
                y_va_mix = np.concatenate([y_va_base] * len(train_modes), axis=0)

                mu_sd = None
                if args.standardize:
                    mu_sd = fit_standardizer(X_tr_mix)
                    X_tr_mix = apply_standardizer(X_tr_mix, *mu_sd)
                    X_va_mix = apply_standardizer(X_va_mix, *mu_sd)

                # val_refmean_acc is computed on the SAME val trials, one copy per mode
                val_by_mode = {m: _reshape_for_model(apply_standardizer(X_va_list[i], *mu_sd) if mu_sd is not None else X_va_list[i]) for i, m in enumerate(train_modes)}
                extra_cbs = [ValRefMeanAccCallback(val_by_mode, y_va_base, args.from_logits)]
                monitor = "val_refmean_acc"

                train_in = X_tr_mix
                X_val_for_loss = X_va_mix
                y_val_for_loss = y_va_mix

            elif cond == "jitter":
                # reference jitter baseline
                # For EA: pre-fit W per jitter mode on the FULL training block after reference.
                ea_ws = None
                if args.ea:
                    ea_ws = {}
                    if args.loso:
                        for m in jitter_modes:
                            _, W_by_group = build_src_mode(m, need_W=True)
                            ea_ws[m] = W_by_group
                    else:
                        for m in jitter_modes:
                            X_full_m = _apply_ref_mode(base_for_train, m, refp)
                            ea_ws[m] = ea_fit(X_full_m)

                # Build standardized mixed pool for scaling (train subset only)
                mu_sd = None
                if args.standardize:
                    X_stats = []
                    for m in jitter_modes:
                        if args.loso:
                            X_full_m, _ = build_src_mode(m, need_W=False)
                            X_m = X_full_m[tr_idx]
                        else:
                            X_m = _apply_ref_mode(base_for_train[tr_idx], m, refp)
                            if ea_ws is not None and m in ea_ws:
                                X_m = ea_apply(X_m, ea_ws[m])
                            elif args.ea:
                                X_m = ea_align_trials(X_m)
                        X_stats.append(X_m)
                    X_stats = np.concatenate(X_stats, axis=0)
                    mu_sd = fit_standardizer(X_stats)

                mu, sd = (mu_sd if mu_sd is not None else (None, None))
                train_in = RefJitterSequence(
                    base_for_train[tr_idx],
                    y_full_oh[tr_idx],
                    batch_size=args.batch_size,
                    ref_modes=jitter_modes,
                    ref_channel=args.ref_channel,
                    laplacian=True,  # neighbor list is computed internally from keep_channels
                    keep_channels=args.keep_channels,
                    ea_ws=ea_ws,
                    group_ids=(group_ids[tr_idx] if args.loso else None),
                    mu=mu,
                    sd=sd,
                    shuffle=True,
                    seed=args.seed,
                )

                # validation for loss: use concatenated val across jitter modes
                X_va_list = []
                for m in jitter_modes:
                    if args.loso:
                        X_full_m, _ = build_src_mode(m, need_W=False)
                        X_m = X_full_m[va_idx]
                    else:
                        X_m = _apply_ref_mode(base_for_train[va_idx], m, refp)
                        if ea_ws is not None and m in ea_ws:
                            X_m = ea_apply(X_m, ea_ws[m])
                        elif args.ea:
                            X_m = ea_align_trials(X_m)
                    if mu_sd is not None:
                        X_m = apply_standardizer(X_m, *mu_sd)
                    X_va_list.append(X_m)

                y_va_base = y_full_oh[va_idx]
                X_val_for_loss = np.concatenate(X_va_list, axis=0)
                y_val_for_loss = np.concatenate([y_va_base] * len(jitter_modes), axis=0)

                val_by_mode = {m: _reshape_for_model(X_va_list[i]) for i, m in enumerate(jitter_modes)}
                extra_cbs = [ValRefMeanAccCallback(val_by_mode, y_va_base, args.from_logits)]
                monitor = "val_refmean_acc"

            else:
                raise ValueError(f"Unknown training condition: {cond}")

            exp_dir = os.path.join(args.results_dir, f"sub_{sub:02d}", f"train_{cond}")
            weights_path = _train_one(
                args,
                train_in,
                y_full_oh[tr_idx] if not isinstance(train_in, KSequence) else y_full_oh[tr_idx],
                X_val_for_loss,
                y_val_for_loss,
                exp_dir,
                monitor=monitor,
                extra_cbs=extra_cbs,
            )

            # -----------------------------
            # Evaluate across test modes
            # -----------------------------
            for te_mode in test_modes:
                if args.loso:
                    X_te = build_tgt_mode(te_mode)
                else:
                    X_te = _apply_ref_ea_block(base_for_test, te_mode, refp, ea=args.ea)
                if args.standardize:
                    # Use the same mu/sd as training condition.
                    # For jitter/mix, mu_sd is already defined in that branch.
                    # For single-mode, mu_sd is defined there.
                    X_te = apply_standardizer(X_te, *mu_sd) if mu_sd is not None else X_te
                acc, _ = _eval_one(args, weights_path, X_te, y_test)
                results[cond][te_mode].append(acc)
                print(f"sub {sub:02d}  train={cond:<9}  test={te_mode:<9}  acc={acc*100:5.2f}")

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
    p = argparse.ArgumentParser("Reference-invariance gate test (paper-clean)")
    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results_gate")
    p.add_argument("--n_sub", type=int, default=9)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_channels", type=int, default=22)  # overridden dynamically
    p.add_argument("--in_samples", type=int, default=1000)  # overridden dynamically
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

    # preprocessing
    p.add_argument("--ea", action="store_true")
    p.add_argument("--no-ea", dest="ea", action="store_false")
    p.set_defaults(ea=True)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    p.set_defaults(standardize=True)
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
    return p.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
