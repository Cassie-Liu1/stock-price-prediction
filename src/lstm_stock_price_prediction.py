"""Multivariate Keras LSTM: 90 past trading observations to the 60th future close."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["Open", "High", "Low", "Close", "Adj Close"]


def target_indices(n, train_end, val_end, window, horizon):
    """Embargo early targets so each later forecast origin follows model selection.

    A target at t uses observations [t-horizon-window+1, t-horizon].
    Later splits start at origin train_end-1 or val_end-1. This avoids training
    or checkpoint selection using labels unavailable at the forecast origin.
    """
    if not 0 < window or not 0 < horizon or not window + horizon - 1 < train_end < val_end < n:
        raise ValueError("Invalid window, horizon, or chronological boundaries")
    splits = {
        "train": np.arange(window + horizon - 1, train_end),
        "validation": np.arange(train_end + horizon - 1, val_end),
        "test": np.arange(val_end + horizon - 1, n),
    }
    if any(len(indices) == 0 for indices in splits.values()):
        raise ValueError("Validation/test periods must be at least horizon observations long")
    return splits


def make_windows(features, target, indices, window, horizon):
    x = np.stack([features[t - horizon - window + 1:t - horizon + 1] for t in indices])
    return x.astype(np.float32), target[indices].astype(np.float32)


def build_model(window, n_features):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/GOOG.csv")
    parser.add_argument("--output-dir", default="reports/lstm")
    parser.add_argument("--window", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.window, args.horizon, args.epochs, args.batch_size) < 1:
        parser.error("window, horizon, epochs, and batch-size must be positive")

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    frame = pd.read_csv(args.input)
    if not {"Date", *FEATURES}.issubset(frame.columns):
        raise ValueError("CSV must contain Date and all five price features")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.sort_values("Date").reset_index(drop=True)
    if frame.Date.isna().any() or frame.Date.duplicated().any() or not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Dates must be unique/nonmissing and features finite")
    raw = frame[FEATURES].to_numpy(dtype=np.float64)
    close = frame.Close.to_numpy(dtype=np.float64).reshape(-1, 1)
    train_end, val_end = int(len(frame) * 0.70), int(len(frame) * 0.85)
    indices = target_indices(len(frame), train_end, val_end, args.window, args.horizon)
    feature_scaler = StandardScaler().fit(raw[:train_end])
    target_scaler = StandardScaler().fit(close[:train_end])
    features = feature_scaler.transform(raw)
    target = target_scaler.transform(close)
    windows = {split: make_windows(features, target, ids, args.window, args.horizon) for split, ids in indices.items()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.window, len(FEATURES))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(str(output / "best_model.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(output / "training_history.csv")),
    ]
    def dataset(x, y, shuffle=False):
        data = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            data = data.shuffle(len(x), seed=args.seed, reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return data.batch(args.batch_size).with_options(options).prefetch(1)

    train = dataset(*windows["train"], shuffle=True)
    validation = dataset(*windows["validation"])
    history = model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks, shuffle=False, verbose=2)
    model = tf.keras.models.load_model(output / "best_model.keras")
    metrics = []
    for split in ["validation", "test"]:
        ids = indices[split]
        x, y = windows[split]
        # Explicit batches avoid uncontrolled prediction thread pools.
        predicted_scaled = np.concatenate([model(x[start:start + args.batch_size], training=False).numpy() for start in range(0, len(x), args.batch_size)])
        prediction = target_scaler.inverse_transform(predicted_scaled).ravel()
        actual = close[ids, 0]
        origin = ids - args.horizon
        baseline = close[origin, 0]
        for name, forecast in [("LSTM", prediction), ("Last observed close", baseline)]:
            error = actual - forecast
            scaled_error = error / target_scaler.scale_[0]
            metrics.append({"split": split, "model": name, "n": len(ids),
                            "MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.mean(error ** 2))),
                            "MSE_standardized": float(np.mean(scaled_error ** 2))})
        pd.DataFrame({"forecast_origin_date": frame.Date.iloc[origin].to_numpy(),
                      "target_date": frame.Date.iloc[ids].to_numpy(), "actual_close": actual,
                      "predicted_close": prediction, "last_observed_close": baseline}).to_csv(output / f"{split}_predictions.csv", index=False)
        if split == "test":
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(frame.Date.iloc[ids], actual, label="Actual")
            ax.plot(frame.Date.iloc[ids], prediction, label="LSTM")
            ax.plot(frame.Date.iloc[ids], baseline, label=f"Close observed {args.horizon} trading steps earlier", alpha=0.7)
            ax.set_title(f"LSTM: {args.window}-step lookback, {args.horizon}-step horizon")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / "test_forecast.png", dpi=160)
            plt.close(fig)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    np.savez(output / "scalers.npz", feature_mean=feature_scaler.mean_, feature_scale=feature_scaler.scale_,
             target_mean=target_scaler.mean_, target_scale=target_scaler.scale_)
    metadata = vars(args) | {
        "features": FEATURES, "target": "Close at exactly horizon trading observations after final input",
        "train_end_row_exclusive": train_end, "validation_end_row_exclusive": val_end,
        "split_counts": {key: len(ids) for key, ids in indices.items()},
        "split_target_dates": {key: [str(frame.Date.iloc[ids[0]].date()), str(frame.Date.iloc[ids[-1]].date())] for key, ids in indices.items()},
        "boundary_embargo": args.horizon - 1, "epochs_completed": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "best_validation_mse_standardized": float(min(history.history["val_loss"])),
        "tensorflow_version": tf.__version__, "dataset_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "comparison_note": "Not directly comparable to the next-day univariate Transformer",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
