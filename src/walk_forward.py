"""Expanding-window evaluation with fixed settings and non-overlapping tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_folds(n, initial_train, validation_size, test_size):
    if min(initial_train, validation_size, test_size) < 1:
        raise ValueError("Split sizes must be positive")
    folds = []
    train_end = initial_train
    while train_end + validation_size < n:
        val_end = train_end + validation_size
        test_end = min(val_end + test_size, n)
        folds.append((train_end, val_end, test_end))
        train_end += test_size
    if not folds:
        raise ValueError("No test observations available")
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/GOOG.csv")
    parser.add_argument("--output-dir", default="reports/walk_forward")
    parser.add_argument("--initial-train", type=int, default=2400)
    parser.add_argument("--validation-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()
    if args.initial_train <= args.window or min(args.epochs, args.window) < 1:
        parser.error("initial-train must exceed a positive window; epochs must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    frame = pd.read_csv(args.input)
    folds = make_folds(len(frame), args.initial_train, args.validation_size, args.test_size)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"folds": folds, "protocol": "expanding train; fixed validation; disjoint rolling one-step tests", "settings_changed_during_evaluation": False, "architecture_informed_by_prior_holdout": True}
    (output / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, predictions = [], []
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    script = Path(__file__).with_name("transformer_stock_price_prediction_original.py")
    for fold_id, (train_end, val_end, test_end) in enumerate(folds, 1):
        for seed in args.seeds:
            run_dir = output / f"fold_{fold_id}_seed_{seed}"
            print(f"Fold {fold_id}/{len(folds)}, seed {seed}: train={train_end}, validation={val_end-train_end}, test={test_end-val_end}", flush=True)
            command = [sys.executable, "-u", str(script), "--input", args.input,
                       "--output-dir", str(run_dir), "--seed", str(seed),
                       "--epochs", str(args.epochs), "--window", str(args.window),
                       "--train-end", str(train_end), "--val-end", str(val_end),
                       "--test-end", str(test_end), "--mode", "residual"]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            (output / f"fold_{fold_id}_seed_{seed}.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            metrics = pd.read_csv(run_dir / "metrics.csv")
            metadata = json.loads((run_dir / "run_metadata.json").read_text())
            model, baseline = metrics.iloc[0], metrics.iloc[1]
            rows.append({"fold": fold_id, "seed": seed, "test_start": metadata["test_start_date"],
                         "test_end": metadata["test_end_date"], "n_test": metadata["test_targets"],
                         "best_epoch": metadata["best_epoch"], "MAE": model.MAE, "RMSE": model.RMSE,
                         "baseline_MAE": baseline.MAE, "baseline_RMSE": baseline.RMSE,
                         "MAE_improvement_pct": 100 * (1 - model.MAE / baseline.MAE),
                         "RMSE_improvement_pct": 100 * (1 - model.RMSE / baseline.RMSE)})
            pred = pd.read_csv(run_dir / "test_predictions.csv")
            pred["fold"], pred["seed"] = fold_id, seed
            predictions.append(pred)
            pd.DataFrame(rows).to_csv(output / "fold_metrics.csv", index=False)
            print(f"  MAE={model.MAE:.4f}; baseline={baseline.MAE:.4f}; best epoch={metadata['best_epoch']}", flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_csv(output / "all_predictions.csv", index=False)
    summaries = []
    for seed, pred in combined.groupby("seed"):
        error = pred.actual_close - pred.predicted_close
        baseline_error = pred.actual_close - pred.naive_previous_close
        summaries.append({"seed": seed, "n_test": len(pred), "MAE": error.abs().mean(),
                          "RMSE": np.sqrt(np.mean(error ** 2)), "baseline_MAE": baseline_error.abs().mean(),
                          "baseline_RMSE": np.sqrt(np.mean(baseline_error ** 2))})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "aggregate_metrics.csv", index=False)
    metrics = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for seed, group in metrics.groupby("seed"):
        axes[0].plot(group.fold, group.MAE_improvement_pct, marker="o", label=f"Seed {seed}")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Temporal fold", ylabel="MAE improvement over previous close (%)", title="Positive = better than baseline", xticks=range(1, len(folds)+1))
    axes[0].legend()
    baseline = metrics.drop_duplicates("fold")
    axes[1].plot(baseline.fold, baseline.baseline_MAE, marker="o", label="Previous close")
    axes[1].plot(metrics.groupby("fold").MAE.mean(), marker="o", label="Transformer (mean across seeds)")
    axes[1].set(xlabel="Temporal fold", ylabel="MAE (price units)", title="Error by test period", xticks=range(1, len(folds)+1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "walk_forward_comparison.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
