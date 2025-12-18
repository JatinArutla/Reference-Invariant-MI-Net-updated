# train_ssl.py
# Self-supervised pretraining for ATCNet on BCI IV-2a
# - Supports LOSO and subject-dependent (single subject or loop all)
# - Optional linear / k-NN probes during training

import os, argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.datamodules.bci2a import load_LOSO_pool, load_subject_dependent
from src.datamodules.channels import BCI2A_CH_NAMES, parse_keep_channels, neighbors_to_index_list, name_to_index
from src.models.model import build_atcnet
from src.models.wrappers import build_ssl_projector
from src.selfsupervised.views import make_ssl_dataset
from src.selfsupervised.losses import nt_xent_loss, barlow_twins_loss, vicreg_loss


# ----------------- Utils -----------------

def set_seed(seed: int = 1):
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def build_encoder(args):
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
        from_logits=False,
        return_ssl_feat=True,  # exposes averaged per-window feature as second output
    )


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _prep_ref_view_params(args):
    """Compute ref_idx and lap_neighbors for SSL view generation."""
    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx] if keep_idx is not None else None
    current_names = keep_names if keep_names is not None else BCI2A_CH_NAMES

    ref_map = name_to_index(current_names)
    ref_idx = None
    if args.ref_channel:
        if args.ref_channel not in ref_map:
            raise ValueError(f"ref_channel '{args.ref_channel}' not in channels: {current_names}")
        ref_idx = ref_map[args.ref_channel]

    ref_modes = _split_csv(getattr(args, "view_ref_modes", None)) or ["car", "ref"]
    view_mode = (args.view_mode or "aug").lower()
    need_lap = args.laplacian or any(m.lower() in ("laplacian", "lap", "local") for m in ref_modes)
    lap_neighbors = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=keep_names) if need_lap else None

    return {
        "ref_modes": ref_modes,
        "ref_idx": ref_idx,
        "lap_neighbors": lap_neighbors,
    }


def _ssl_loss_fn(args):
    loss_name = (args.ssl_loss or "nt_xent").lower()
    if loss_name in ("nt_xent", "ntxent", "simclr"):
        temperature = tf.constant(args.temperature, tf.float32)
        def _loss(z1, z2):
            return nt_xent_loss(z1, z2, temperature)
        return _loss, True
    if loss_name in ("barlow", "barlow_twins"):
        def _loss(z1, z2):
            return barlow_twins_loss(z1, z2, lambd=float(args.barlow_lambda))
        return _loss, False
    if loss_name in ("vicreg",):
        def _loss(z1, z2):
            return vicreg_loss(
                z1, z2,
                sim_coeff=float(args.vicreg_sim),
                std_coeff=float(args.vicreg_std),
                cov_coeff=float(args.vicreg_cov),
            )
        return _loss, False
    raise ValueError(f"Unknown ssl_loss: {args.ssl_loss}")

def _pack_X(X):  # [N,C,T] -> [N,1,C,T]
    X = X.astype(np.float32, copy=False)
    return X if X.ndim == 4 else X[:, None, ...]

def _to_b1ct(x):
    # If dataset yields [B,C,T] → make [B,1,C,T]; if already [B,1,C,T] → pass-through
    return x if x.shape.rank == 4 else tf.expand_dims(x, 1)

def _probe_now(encoder, X, y, split: float, k: int):
    feat_model = tf.keras.Model(encoder.input, encoder.outputs[1], name="feature_tap")
    Z = feat_model(_pack_X(X), training=False).numpy()
    Ztr, Zva, ytr, yva = train_test_split(Z, y, test_size=split, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
    acc_lr = accuracy_score(yva, lr.predict(Zva))
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1).fit(Ztr, ytr)
    acc_knn = accuracy_score(yva, knn.predict(Zva))
    return acc_lr, acc_knn


# ----------------- Runners -----------------

def run_loso(args):
    ref_params = _prep_ref_view_params(args)
    loss_fn, need_l2norm = _ssl_loss_fn(args)
    for tgt in range(1, args.n_sub + 1):
        fold_dir = os.path.join(args.results_dir, f"LOSO_{tgt:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        (X_src, y_src), (X_tgt, y_tgt) = load_LOSO_pool(
            args.data_root, tgt,
            n_sub=args.n_sub, ea=args.ea, standardize=args.standardize,
            per_block_standardize=args.per_block_standardize,
            t1_sec=args.t1_sec, t2_sec=args.t2_sec,
            ref_mode=args.data_ref_mode,
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=args.laplacian,
        )

        # If channel subsetting is active, n_channels changes.
        args.n_channels = int(X_src.shape[1])

        ds = make_ssl_dataset(
            X_src, n_channels=args.n_channels, in_samples=args.in_samples,
            batch_size=args.batch_size, shuffle=True,
            seed=args.seed,
            deterministic=True,
            view_mode=args.view_mode,
            ref_modes=ref_params["ref_modes"],
            ref_idx=ref_params["ref_idx"],
            lap_neighbors=ref_params["lap_neighbors"],
        )

        encoder = build_encoder(args)
        ssl_model = build_ssl_projector(encoder, proj_dim=args.proj_dim, out_dim=args.out_dim, l2norm=need_l2norm)
        opt = tf.keras.optimizers.Adam(args.lr)

        @tf.function(reduce_retracing=True)  # comment out to run eagerly
        def train_step(v1, v2):
            with tf.GradientTape() as tape:
                z1 = ssl_model(v1, training=True)
                z2 = ssl_model(v2, training=True)
                loss = loss_fn(z1, z2)
            grads = tape.gradient(loss, ssl_model.trainable_variables)
            grads_vars = [(g, v) for g, v in zip(grads, ssl_model.trainable_variables) if g is not None]
            opt.apply_gradients(grads_vars)
            return loss

        # warm-up call to build variables once
        warm = next(iter(ds))
        _ = train_step(_to_b1ct(warm[0]), _to_b1ct(warm[1]))

        for ep in range(1, args.epochs + 1):
            losses = []
            for v1, v2 in ds:
                losses.append(train_step(_to_b1ct(v1), _to_b1ct(v2)))
            if ep % args.log_every == 0:
                print(f"[LOSO {tgt:02d}] epoch {ep:03d}/{args.epochs}  ssl_loss={float(tf.reduce_mean(losses)): .4f}")

            if args.probe_every > 0 and ep % args.probe_every == 0:
                if args.probe_on == "target":
                    acc_lr, acc_knn = _probe_now(encoder, X_tgt, y_tgt, args.probe_split, args.probe_k)
                else:
                    acc_lr, acc_knn = _probe_now(encoder, X_src, y_src, args.probe_split, args.probe_k)
                print(f"   ↳ probe@{ep}: linear={acc_lr:.3f}  knn@{args.probe_k}={acc_knn:.3f}")

            if ep % args.save_every == 0 or ep == args.epochs:
                wpath = os.path.join(fold_dir, f"ssl_encoder_sub{tgt}_epoch{ep}.weights.h5")
                encoder.save_weights(wpath)


def run_subject_dependent_one(args, sub: int):
    ref_params = _prep_ref_view_params(args)
    loss_fn, need_l2norm = _ssl_loss_fn(args)
    (X_tr, y_tr), (X_te, y_te) = load_subject_dependent(
        args.data_root, sub,
        ea=args.ea, standardize=args.standardize,
        t1_sec=args.t1_sec, t2_sec=args.t2_sec,
        ref_mode=args.data_ref_mode,
        keep_channels=args.keep_channels,
        ref_channel=args.ref_channel,
        laplacian=args.laplacian,
    )

    # If channel subsetting is active, n_channels changes.
    args.n_channels = int(X_tr.shape[1])

    ds = make_ssl_dataset(
        X_tr, n_channels=args.n_channels, in_samples=args.in_samples,
        batch_size=args.batch_size, shuffle=True,
        view_mode=args.view_mode,
        ref_modes=ref_params["ref_modes"],
        ref_idx=ref_params["ref_idx"],
        lap_neighbors=ref_params["lap_neighbors"],
    )

    sub_dir = os.path.join(args.results_dir, f"SUBJ_{sub:02d}")
    os.makedirs(sub_dir, exist_ok=True)

    encoder = build_encoder(args)
    ssl_model = build_ssl_projector(encoder, proj_dim=args.proj_dim, out_dim=args.out_dim, l2norm=need_l2norm)
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function(reduce_retracing=True)
    def train_step(v1, v2):
        with tf.GradientTape() as tape:
            z1 = ssl_model(v1, training=True)
            z2 = ssl_model(v2, training=True)
            loss = loss_fn(z1, z2)
        grads = tape.gradient(loss, ssl_model.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, ssl_model.trainable_variables) if g is not None]
        opt.apply_gradients(grads_vars)
        return loss

    warm = next(iter(ds))
    _ = train_step(_to_b1ct(warm[0]), _to_b1ct(warm[1]))

    for ep in range(1, args.epochs + 1):
        losses = []
        for v1, v2 in ds:
            losses.append(train_step(_to_b1ct(v1), _to_b1ct(v2)))
        if ep % args.log_every == 0:
            print(f"[SUBJ {sub:02d}] epoch {ep:03d}/{args.epochs}  ssl_loss={float(tf.reduce_mean(losses)): .4f}")

        if args.probe_every > 0 and ep % args.probe_every == 0:
            Xp, yp = (X_te, y_te) if args.probe_on == "target" else (X_tr, y_tr)
            acc_lr, acc_knn = _probe_now(encoder, Xp, yp, args.probe_split, args.probe_k)
            print(f"   ↳ probe@{ep}: linear={acc_lr:.3f}  knn@{args.probe_k}={acc_knn:.3f}")

        if ep % args.save_every == 0 or ep == args.epochs:
            wpath = os.path.join(sub_dir, f"ssl_encoder_sub{sub}_epoch{ep}.weights.h5")
            encoder.save_weights(wpath)


def run_subject_dependent_all(args):
    for sub in range(1, args.n_sub + 1):
        run_subject_dependent_one(args, sub)


# ----------------- CLI -----------------

def parse_args():
    p = argparse.ArgumentParser("ATCNet SSL pretraining (+ optional linear/kNN probe)")
    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results_ssl")
    p.add_argument("--n_sub", type=int, default=9)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_channels", type=int, default=22)
    p.add_argument("--in_samples", type=int, default=1000)
    p.add_argument("--t1_sec", type=float, default=2.0)
    p.add_argument("--t2_sec", type=float, default=6.0)
    p.add_argument("--ea", action="store_true"); p.add_argument("--no-ea", dest="ea", action="store_false"); p.set_defaults(ea=True)
    p.add_argument("--standardize", action="store_true"); p.add_argument("--no-standardize", dest="standardize", action="store_false"); p.set_defaults(standardize=True)
    p.add_argument("--per_block_standardize", action="store_true"); p.add_argument("--no-per_block_standardize", dest="per_block_standardize", action="store_false"); p.set_defaults(per_block_standardize=True)

    # data reference controls (applied at load time, before EA/standardization)
    p.add_argument("--data_ref_mode", type=str, default="native", choices=["native","car","ref","laplacian"],
                   help="Fixed reference transform to apply when loading data.")
    p.add_argument("--keep_channels", type=str, default=None,
                   help="Comma-separated channel names to keep (e.g., 'C3,Cz,C4').")
    p.add_argument("--ref_channel", type=str, default="Cz", help="Channel name for mode='ref' (must exist after keep_channels).")
    p.add_argument("--laplacian", action="store_true", help="Enable Laplacian neighbors (needed for view_mode ref/laplacian).")

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

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.5)

    # SSL view definition
    p.add_argument("--view_mode", type=str, default="aug", choices=["aug","ref","ref+aug"],
                   help="How to generate SSL positive pairs.")
    p.add_argument("--view_ref_modes", type=str, default="car,ref", help="Comma-separated ref modes for view_mode='ref'.")

    # SSL loss
    p.add_argument("--ssl_loss", type=str, default="nt_xent", choices=["nt_xent","barlow","vicreg"])
    p.add_argument("--barlow_lambda", type=float, default=5e-3)
    p.add_argument("--vicreg_sim", type=float, default=25.0)
    p.add_argument("--vicreg_std", type=float, default=25.0)
    p.add_argument("--vicreg_cov", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)

    # mode
    p.add_argument("--loso", action="store_true")
    p.add_argument("--subject", type=int, default=None,
                   help="Subject ID for subject-dependent mode; omit to loop 1..n_sub")

    # projector
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=64)

    # probes
    p.add_argument("--probe_every", type=int, default=0, help="run linear/kNN probe every N epochs (0=off)")
    p.add_argument("--probe_split", type=float, default=0.2, help="validation split for probe")
    p.add_argument("--probe_k", type=int, default=5, help="k for k-NN probe")
    p.add_argument("--probe_on", type=str, default="target", choices=["target","source"], help="which labeled set to probe on")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)

    if args.loso:
        run_loso(args)
    else:
        if args.subject is None:
            run_subject_dependent_all(args)
        else:
            run_subject_dependent_one(args, args.subject)


if __name__ == "__main__":
    main()