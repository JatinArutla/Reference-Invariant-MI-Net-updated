"""gate_reference.py

Reference-invariance *gate test* for MI-Net.

What it does
  - Trains a supervised ATCNet model under a chosen *training* reference scheme.
  - Evaluates the same trained weights under multiple *test* reference schemes.
  - Optionally uses a stronger baseline: *mixed-reference training* by
    concatenating training data re-referenced with several modes.

This is the experiment that tells you whether “common re-referencing” and/or
“reference jitter” already kills the mismatch. If they do, the SSL paper is weak.

Notes
  - Standardization is fit on the training set (after EA, if enabled) and then
    applied to every test mode. This is important: each test mode should not
    get its own fitted scaler.
  - EA (Euclidean Alignment) is applied per block (it does not use train stats).
"""

import os
import argparse
import json
from typing import Dict, List, Tuple

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

from src.datamodules.bci2a import load_LOSO_pool, load_subject_dependent
from src.datamodules.transforms import fit_standardizer, apply_standardizer
from src.models.model import build_atcnet

from tensorflow.keras.utils import Sequence as KSequence
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


def _load_train_and_test(
    args,
    sub: int,
    *,
    ref_mode_train: str,
    ref_mode_test: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return (X_train,y_train), (X_test,y_test) with:

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

    # subject-dependent
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


def _maybe_mixed_ref_training(args, sub: int, ref_modes_train: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate train data across multiple training reference modes."""
    Xs, ys = [], []
    for m in ref_modes_train:
        (X_tr, y_tr), _ = _load_train_and_test(args, sub, ref_mode_train=m, ref_mode_test=m)
        Xs.append(X_tr)
        ys.append(y_tr)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _fit_and_apply_standardization(args, X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not args.standardize:
        return X_tr.astype(np.float32), X_te.astype(np.float32)
    mu, sd = fit_standardizer(X_tr)
    return apply_standardizer(X_tr, mu, sd), apply_standardizer(X_te, mu, sd)


def _fit_standardizer_if_needed(args, X_tr: np.ndarray):
    """Return (mu, sd) or (None, None)."""
    if not args.standardize:
        return None, None
    return fit_standardizer(X_tr)


def train_one(args, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.weights.h5")

    model = build_model(args)
    model.compile(
        loss=CategoricalCrossentropy(from_logits=args.from_logits),
        optimizer=Adam(learning_rate=args.lr),
        metrics=["accuracy"],
    )

    cbs = [
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=args.plateau_patience, verbose=0, min_lr=args.min_lr),
    ]
    if args.early_stop:
        cbs.append(EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", restore_best_weights=False, verbose=0))

    if isinstance(X_tr, KSequence):
        model.fit(
            X_tr,
            validation_data=(_reshape_for_model(X_va), y_va),
            epochs=args.epochs,
            verbose=0,
            callbacks=cbs,
        )
    else:
        model.fit(
            _reshape_for_model(X_tr), y_tr,
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

    # training conditions: either fixed mode(s), or one mixed-reference baseline, or jitter baseline.
    train_modes = [m.strip() for m in args.train_ref_modes.split(",") if m.strip()]
    if not train_modes:
        raise ValueError("--train_ref_modes must be non-empty.")

    if args.mix_train_refs and args.jitter_train_refs:
        raise ValueError("Pick only one of --mix_train_refs or --jitter_train_refs.")

    train_conds = ["mix"] if args.mix_train_refs else (["jitter"] if args.jitter_train_refs else train_modes)

    # output containers
    results: Dict[str, Dict[str, List[float]]] = {c: {tm: [] for tm in test_modes} for c in train_conds}

    # parse jitter modes (if empty, reuse train_modes)
    jitter_modes = [m.strip() for m in args.jitter_ref_modes.split(",") if m.strip()] if args.jitter_ref_modes else train_modes

    for sub in range(1, args.n_sub + 1):
        for cond in train_conds:

            if cond == "mix":
                X_tr_raw, y_tr_raw = _maybe_mixed_ref_training(args, sub, ref_modes_train=train_modes)

            elif cond == "jitter":
                # base data you jitter FROM, pick one fixed mode (native is simplest)
                (X_tr_raw, y_tr_raw), _ = _load_train_and_test(args, sub, ref_mode_train="native", ref_mode_test="native")

            else:
                (X_tr_raw, y_tr_raw), _ = _load_train_and_test(args, sub, ref_mode_train=cond, ref_mode_test=cond)

            args.n_channels = int(X_tr_raw.shape[1])
            args.in_samples = int(X_tr_raw.shape[2])

            y_oh = to_categorical(y_tr_raw, num_classes=args.n_classes)
            X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
                X_tr_raw,
                y_oh,
                test_size=args.val_ratio,
                random_state=args.seed,
                stratify=y_oh.argmax(-1),
            )

            # standardizer stats
            mu_sd = None
            if args.standardize:
                if cond == "jitter":
                    # fit on a mixed pool so the scaler matches what jitter will produce
                    X_stats, _ = _maybe_mixed_ref_training(args, sub, ref_modes_train=jitter_modes)
                    mu_sd = fit_standardizer(X_stats)
                else:
                    mu_sd = fit_standardizer(X_tr_raw)

            # build training input
            if cond == "jitter":
                mu, sd = mu_sd if mu_sd is not None else (None, None)
                train_in = RefJitterSequence(
                    X_tr_raw, y_tr,
                    batch_size=args.batch_size,
                    ref_modes=jitter_modes,
                    ref_channel=args.ref_channel,
                    laplacian=args.laplacian,
                    keep_channels=args.keep_channels,
                    mu=mu, sd=sd,
                    shuffle=True,
                    seed=args.seed,
                )
                # validation must be a normal tensor
                if mu_sd is not None:
                    mu, sd = mu_sd
                    X_va = apply_standardizer(X_va_raw, mu, sd)
                else:
                    X_va = X_va_raw.astype(np.float32)
            else:
                # normal training
                X_tr = apply_standardizer(X_tr_raw, *mu_sd) if mu_sd is not None else X_tr_raw.astype(np.float32)
                X_va = apply_standardizer(X_va_raw, *mu_sd) if mu_sd is not None else X_va_raw.astype(np.float32)
                train_in = X_tr

            exp_dir = os.path.join(args.results_dir, f"sub_{sub:02d}", f"train_{cond}")
            weights_path = train_one(args, train_in, y_tr, X_va, y_va, exp_dir)

            # evaluate across test reference modes, using the SAME mu_sd
            for te_mode in test_modes:
                X_te_raw, y_te = _load_test_only(args, sub, ref_mode_test=te_mode)
                if mu_sd is not None:
                    mu, sd = mu_sd
                    X_te = apply_standardizer(X_te_raw, mu, sd)
                else:
                    X_te = X_te_raw.astype(np.float32)

                acc, _ = eval_one(args, weights_path, X_te, y_te)
                results[cond][te_mode].append(acc)
                print(f"sub {sub:02d}  train={cond:<9}  test={te_mode:<9}  acc={acc*100:5.2f}")

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
    p.add_argument("--n_channels", type=int, default=22)  # will be overridden dynamically
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
    p.add_argument("--ea", action="store_true"); p.add_argument("--no-ea", dest="ea", action="store_false"); p.set_defaults(ea=True)
    p.add_argument("--standardize", action="store_true"); p.add_argument("--no-standardize", dest="standardize", action="store_false"); p.set_defaults(standardize=True)
    p.add_argument("--per_block_standardize", action="store_true"); p.add_argument("--no-per_block_standardize", dest="per_block_standardize", action="store_false"); p.set_defaults(per_block_standardize=True)
    p.add_argument("--loso", action="store_true"); p.add_argument("--no-loso", dest="loso", action="store_false"); p.set_defaults(loso=True)

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
