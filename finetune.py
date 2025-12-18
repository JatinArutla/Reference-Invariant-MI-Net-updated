import os, argparse, time, glob
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

tf.keras.backend.set_image_data_format("channels_last")
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from src.datamodules.bci2a import load_LOSO_pool, load_subject_dependent
from src.models.model import build_atcnet

def set_seed(seed: int = 1):
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def _reshape_for_model(X: np.ndarray) -> np.ndarray:
    N, C, T = X.shape
    return X.reshape(N, 1, C, T).astype(np.float32, copy=False)

def _maybe_load_ssl(model: tf.keras.Model, ssl_template: str, subject: int) -> None:
    if not ssl_template:
        return
    path = ssl_template.format(sub=subject)
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, f"*sub{subject}_epoch*.weights.h5")))
        if candidates:
            path = candidates[-1]
    if os.path.exists(path):
        model.load_weights(path, by_name=True, skip_mismatch=True)

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

def _load_train_val_for_subject(args, sub: int):
    if args.loso:
        (X_src, y_src), _ = load_LOSO_pool(
            args.data_root, sub,
            n_sub=args.n_sub, ea=args.ea, standardize=args.standardize,
            per_block_standardize=args.per_block_standardize,
            t1_sec=args.t1_sec, t2_sec=args.t2_sec,
            ref_mode=args.data_ref_mode,
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=args.laplacian,
        )
        args.n_channels = int(X_src.shape[1])
        y_oh = to_categorical(y_src, num_classes=args.n_classes)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_src, y_oh, test_size=args.val_ratio, random_state=args.seed,
            stratify=y_oh.argmax(-1)
        )
    else:
        (X_tr_raw, y_tr), (X_te_raw, _) = load_subject_dependent(
            args.data_root, sub,
            ea=args.ea, standardize=args.standardize,
            t1_sec=args.t1_sec, t2_sec=args.t2_sec,
            ref_mode=args.data_ref_mode,
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=args.laplacian,
        )
        args.n_channels = int(X_tr_raw.shape[1])
        y_oh = to_categorical(y_tr, num_classes=args.n_classes)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr_raw, y_oh, test_size=args.val_ratio, random_state=args.seed,
            stratify=y_oh.argmax(-1)
        )
    return _reshape_for_model(X_tr), _reshape_for_model(X_va), y_tr, y_va

def run(args):
    set_seed(args.seed)
    if os.path.exists(args.results_dir) and args.clean:
        import shutil; shutil.rmtree(args.results_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    log_path = os.path.join(args.results_dir, "log.txt")
    best_list_path = os.path.join(args.results_dir, "best_models.txt")
    log = open(log_path, "w"); best_list = open(best_list_path, "w")

    acc = np.zeros((args.n_sub, args.n_runs))
    kappa = np.zeros((args.n_sub, args.n_runs))

    t0_all = time.time()
    for sub in range(1, args.n_sub + 1):
        print(f"\nTraining on subject {sub}  ({'LOSO' if args.loso else 'subject-dependent'})")
        log.write(f"\nTraining on subject {sub}\n")

        X_tr, X_va, y_tr, y_va = _load_train_val_for_subject(args, sub)

        best_acc, best_hist, best_run_path = 0.0, None, None
        for run_idx in range(args.n_runs):
            tf.random.set_seed(args.seed + run_idx)
            np.random.seed(args.seed + run_idx)

            model = build_model(args)
            _maybe_load_ssl(model, args.ssl_weights, sub)

            run_dir = os.path.join(args.results_dir, "saved_models", f"run-{run_idx+1}")
            os.makedirs(run_dir, exist_ok=True)
            ckpt_path = os.path.join(run_dir, f"subject-{sub}.weights.h5")

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

            t0 = time.time()
            hist = model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=args.epochs,
                batch_size=args.batch_size,
                shuffle=False,
                verbose=0,
                callbacks=cbs,
            )
            model.load_weights(ckpt_path)
            y_pred = model.predict(X_va, verbose=0)
            y_hat = y_pred.argmax(axis=-1) if not args.from_logits else tf.nn.softmax(y_pred).numpy().argmax(axis=-1)
            y_true = y_va.argmax(axis=-1)

            acc[sub-1, run_idx] = accuracy_score(y_true, y_hat)
            kappa[sub-1, run_idx] = cohen_kappa_score(y_true, y_hat)

            msg = (
                f"Subject: {sub}  run {run_idx+1}  time: {(time.time()-t0)/60:.1f} m  "
                f"val_acc: {acc[sub-1, run_idx]:.4f}  val_loss: {min(hist.history['val_loss']):.3f}"
            )
            print(msg); log.write(msg + "\n")

            if acc[sub-1, run_idx] > best_acc:
                best_acc = acc[sub-1, run_idx]; best_hist = hist; best_run_path = ckpt_path

        if best_run_path:
            rel = os.path.relpath(best_run_path, args.results_dir)
            best_list.write(rel + "\n")

        if args.plot_curves and best_hist is not None:
            import matplotlib.pyplot as plt
            plt.plot(best_hist.history["accuracy"]); plt.plot(best_hist.history["val_accuracy"])
            plt.title(f"Accuracy - subject {sub}"); plt.legend(["Train","Val"]); plt.show()
            plt.plot(best_hist.history["loss"]); plt.plot(best_hist.history["val_loss"])
            plt.title(f"Loss - subject {sub}"); plt.legend(["Train","Val"]); plt.show(); plt.close()

    hdr1 = "         " + "".join([f"sub_{i:02d}   " for i in range(1, args.n_sub+1)]) + "  average"
    hdr2 = "         " + "".join(["------   " for _ in range(args.n_sub)]) + "  -------"
    info = "\n---------------------------------\nValidation performance (acc %):\n"
    info += "---------------------------------\n" + hdr1 + "\n" + hdr2
    for r in range(args.n_runs):
        line = f"\nRun {r+1}: "
        for s in range(args.n_sub):
            line += f"{acc[s, r]*100:6.2f}   "
        line += f"  {np.mean(acc[:, r])*100:6.2f}   "
        info += line
    info += f"\n---------------------------------\nAverage acc - all runs: {np.mean(acc)*100:.2f} %"
    info += f"\nTrain Time (all): {(time.time()-t0_all)/60:.1f} min\n---------------------------------\n"
    print(info); log.write(info + "\n")

    best_list.close(); log.close()

def parse_args():
    p = argparse.ArgumentParser("ATCNet supervised finetuning")
    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--n_sub", type=int, default=9)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_channels", type=int, default=22)
    p.add_argument("--in_samples", type=int, default=1000)
    p.add_argument("--t1_sec", type=float, default=2.0)
    p.add_argument("--t2_sec", type=float, default=6.0)

    # reference / channel control
    p.add_argument(
        "--data_ref_mode",
        type=str,
        default="native",
        choices=["native", "car", "ref", "laplacian"],
        help="Reference mode applied to loaded data before EA/standardization.",
    )
    p.add_argument(
        "--keep_channels",
        type=str,
        default="",
        help="Comma-separated channel names to keep (intersection baseline). If set, data is subset/reordered before reference transform.",
    )
    p.add_argument(
        "--ref_channel",
        type=str,
        default="Cz",
        help="Recorded channel name used when data_ref_mode='ref' (default Cz).",
    )
    p.add_argument(
        "--laplacian",
        action="store_true",
        help="Also build Laplacian neighbors (needed when data_ref_mode='laplacian').",
    )

    p.add_argument("--ea", action="store_true"); p.add_argument("--no-ea", dest="ea", action="store_false"); p.set_defaults(ea=True)
    p.add_argument("--standardize", action="store_true"); p.add_argument("--no-standardize", dest="standardize", action="store_false"); p.set_defaults(standardize=True)
    p.add_argument("--per_block_standardize", action="store_true"); p.add_argument("--no-per_block_standardize", dest="per_block_standardize", action="store_false"); p.set_defaults(per_block_standardize=True)
    p.add_argument("--val_ratio", type=float, default=0.2)
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
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--plateau_patience", type=int, default=20)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--plot_curves", action="store_true")

    # SSL init
    p.add_argument("--ssl_weights", type=str, default="", help="Path/template to SSL weights, e.g., './results_ssl/LOSO_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5' or './results_ssl_subj/SUBJ_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5'")
    return p.parse_args()

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()