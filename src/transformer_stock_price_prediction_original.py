"""Univariate, chronological next-day Close forecasting with a small Transformer."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, window: int):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.values) - self.window

    def __getitem__(self, index):
        return self.values[index:index + self.window], self.values[index + self.window]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class TransformerPredictor(nn.Module):
    def __init__(self, window: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2, mode: str = "residual"):
        super().__init__()
        self.mode = mode
        self.input_projection = nn.Linear(1, d_model)
        self.position = PositionalEncoding(d_model, window)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)
        if mode == "residual":
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x):
        last_close = x[:, -1, :]
        if self.mode == "residual":
            x = x - last_close.unsqueeze(1)
        x = self.position(self.input_projection(x))
        output = self.output(self.encoder(x)[:, -1, :])
        return last_close + output if self.mode == "residual" else output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/GOOG.csv")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["absolute", "residual"], default="residual")
    parser.add_argument("--train-end", type=int, help="Exclusive training row boundary")
    parser.add_argument("--val-end", type=int, help="Exclusive validation row boundary")
    parser.add_argument("--test-end", type=int, help="Exclusive test row boundary")
    args = parser.parse_args()
    if min(args.window, args.epochs, args.batch_size) < 1:
        parser.error("window, epochs, and batch-size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame = pd.read_csv(args.input, parse_dates=["Date"]).sort_values("Date")
    if not {"Date", "Close"}.issubset(frame.columns):
        raise ValueError("Input CSV must contain Date and Close columns")
    if frame["Date"].isna().any() or frame["Date"].duplicated().any() or not np.isfinite(frame["Close"].to_numpy(dtype=float)).all():
        raise ValueError("Date must be unique and Close must not contain missing values")
    prices = frame["Close"].to_numpy(dtype=np.float32).reshape(-1, 1)
    n = len(prices)
    boundaries = (args.train_end, args.val_end, args.test_end)
    if any(value is not None for value in boundaries):
        if any(value is None for value in boundaries):
            parser.error("train-end, val-end, and test-end must be supplied together")
        train_end, val_end, test_end = boundaries
        if not args.window < train_end < val_end < test_end <= n:
            parser.error("Require window < train-end < val-end < test-end <= rows")
        frame = frame.iloc[:test_end].copy()
        prices = prices[:test_end]
        n = test_end
    else:
        train_end, val_end = int(n * 0.70), int(n * 0.85)
    if train_end <= args.window or val_end <= train_end or n <= val_end:
        raise ValueError("Not enough observations for requested chronological splits")

    scaler = MinMaxScaler().fit(prices[:train_end])
    scaled = scaler.transform(prices)
    # Include prior-period context in each later split, but score only targets
    # belonging to that split. No future target is used as an input.
    train_data = WindowDataset(scaled[:train_end], args.window)
    val_data = WindowDataset(scaled[train_end - args.window:val_end], args.window)
    test_data = WindowDataset(scaled[val_end - args.window:], args.window)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(args.window, mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # The zero-initialized residual head starts at the previous-close baseline.
    # Include it in validation selection so training must improve validation MSE.
    best_val = float(np.mean(np.diff(scaled[train_end - 1:val_end, 0]) ** 2)) if args.mode == "residual" else float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_squared_error = 0.0
        val_count = 0
        with torch.inference_mode():
            for x, y in val_loader:
                error = model(x.to(device)) - y.to(device)
                val_squared_error += error.square().sum().item()
                val_count += y.numel()
        val_loss = val_squared_error / val_count
        history.append({"epoch": epoch + 1, "validation_mse_scaled": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch + 1}/{args.epochs}: validation MSE={val_loss:.6f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    scaled_predictions = []
    with torch.inference_mode():
        for x, _ in test_loader:
            scaled_predictions.extend(model(x.to(device)).cpu().numpy().ravel())
    prediction = scaler.inverse_transform(np.asarray(scaled_predictions).reshape(-1, 1)).ravel()
    actual = prices[val_end:].ravel()
    naive = prices[val_end - 1:-1].ravel()
    mae = float(np.mean(np.abs(actual - prediction)))
    rmse = float(np.sqrt(np.mean((actual - prediction) ** 2)))
    naive_mae = float(np.mean(np.abs(actual - naive)))
    naive_rmse = float(np.sqrt(np.mean((actual - naive) ** 2)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    metadata = {
        "mode": args.mode, "seed": args.seed, "window": args.window,
        "epochs": args.epochs, "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_val,
        "validation_baseline_mse_scaled": float(np.mean(np.diff(scaled[train_end - 1:val_end, 0]) ** 2)),
        "train_targets": len(train_data), "validation_targets": len(val_data), "test_targets": len(test_data),
        "train_end_date": str(frame["Date"].iloc[train_end - 1].date()),
        "validation_end_date": str(frame["Date"].iloc[val_end - 1].date()),
        "test_start_date": str(frame["Date"].iloc[val_end].date()),
        "test_end_date": str(frame["Date"].iloc[-1].date()),
        "torch_version": torch.__version__, "device": str(device),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame({
        "Date": frame["Date"].iloc[val_end:].to_numpy(),
        "actual_close": actual,
        "predicted_close": prediction,
        "naive_previous_close": naive,
    }).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([{"model": "Transformer", "MAE": mae, "RMSE": rmse},
                  {"model": "Previous close baseline", "MAE": naive_mae, "RMSE": naive_rmse}]
                 ).to_csv(output_dir / "metrics.csv", index=False)
    plt.figure(figsize=(12, 5))
    plt.plot(frame["Date"].iloc[val_end:], actual, label="Actual")
    plt.plot(frame["Date"].iloc[val_end:], prediction, label="Transformer")
    plt.plot(frame["Date"].iloc[val_end:], naive, label="Previous close baseline", alpha=0.7)
    plt.title(f"Chronological test: MAE={mae:.2f}, RMSE={rmse:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_forecast.png", dpi=160)
    plt.close()
    print(f"Test MAE={mae:.4f} RMSE={rmse:.4f}; naive MAE={naive_mae:.4f} RMSE={naive_rmse:.4f}")


if __name__ == "__main__":
    main()
