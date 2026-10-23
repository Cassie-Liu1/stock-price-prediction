# Model Analysis and Evaluation

[中文版](model-review.zh-CN.md)

## Evidence inspected

The legacy LSTM notebook and model weights were moved out during project cleanup. The new LSTM script reimplements the forecasting task without loading those weights or guaranteeing reproduction of the historical results.

- `GOOG.csv`: 4,006 daily records (2004-08-19 to 2020-07-17), seven columns, no obvious missing rows in the supplied file.
- Legacy `LSTM_model_stocks.ipynb`: multivariate StandardScaler preprocessing, a 90-day historical window, 60-day future forecasting, stacked LSTM, early stopping, learning-rate reduction, and checkpointing.
- Legacy `Transformer_stock_price_prediction.py`: a Close-only prototype using a 150-step window and PyTorch `nn.Transformer`.

## Reproducibility and Evaluation Risks

### Current Transformer Script

- The repaired script uses chronological 70/15/15 splits and fits the scaler on the training period only.
- Validation selects the best checkpoint; the test period is scored once after training.
- Test output includes MAE (mean absolute error), RMSE (root mean squared error), and a previous-close baseline.
- The script predicts one next-step Close value from a fixed lookback window.

### Legacy LSTM Notebook

- The legacy experiment targets older TensorFlow/Keras interfaces.
- Its records do not directly establish generalization to future periods; the experiment needs reevaluation using correct chronological splits.
- Future dates were generated as calendar days even though the input is a trading-day series. Production use would require the relevant market's trading calendar.

## Direct-Price Prediction: Diagnosis and Correction

The first repaired direct-price run had test MAE 313.1386 and RMSE 329.7129, versus 15.6358 and 23.3316 for previous close. Training closes ranged from 49.82 to 672.93, whereas test closes ranged from 976.22 to 1541.74. Predictions ranged from 821.16 to 948.36 and underestimated all test prices. This supports price-level extrapolation failure as the main observed symptom; it is not proof that this is the only contributing cause.

The revised model subtracts the last close from each input window and predicts a change relative to that price:

```text
Predicted close = Last observed close + Predicted change
```

The output head is initialized to zero, so the initial model matches the previous-close baseline; epoch 0 is eligible for validation selection. Validation error is aggregated with equal weight for each observation, preventing the final smaller batch from receiving excessive weight.

Seed 42, 30 epochs, and the unchanged chronological boundaries selected epoch 17. Test MAE is 15.5973 and RMSE is 23.3166. The small advantage over persistence should be described as near-baseline performance. Three automated checks cover later-split input/target alignment, initial persistence behavior, and invariance of the learned change to constant price shifts.

The old artifacts are retained in `reports/absolute_original/`; the revised run records its configuration and validation history. The prior run used batch-averaged validation loss, so it is an archival comparison rather than a perfectly controlled architecture ablation. Test feedback informed this revision, so these results cannot be treated as untouched final-test evidence.

## Completed Expanding-Window Evaluation

`src/walk_forward.py` runs four non-overlapping test folds, with expanding training prefixes of 2,400, 2,800, 3,200, and 3,600 observations and 252-observation validation blocks. Test sizes are 400, 400, 400, and 154. Seeds 42, 7, and 123 each use 30 epochs with the same residual architecture and 60-observation lookback. Each run fits its own train-only scaler and selects its own validation checkpoint; settings are not changed in response to the fold results. This models repeated one-step predictions with observed prior closes, rather than a multi-step forecast issued at the fold start.

The evaluated dates span 2015-03-04 to 2020-07-17, with 1,354 distinct test dates per seed.

| Overall metric | Transformer: range across three seeds | Previous-close baseline |
| --- | ---: | ---: |
| MAE | 10.9011–10.9077 | 10.9425 |
| RMSE | 17.4827–17.4858 | 17.5010 |

Overall MAE improves by approximately 0.32%–0.38%. MAE is lower in 11 of 12 runs and RMSE in 10 of 12, but the first period is almost tied with the baseline. Absolute error increases markedly in the final period for both methods.

The four automated checks pass. Saved outputs contain exactly 12 runs and 4,062 prediction rows, with no duplicate dates within a seed and finite forecast values. Aggregated error is calculated from individual predictions per seed, accounting for the shorter final fold. The summary never treats repeated predictions from different seeds as independent test dates.

Artifacts are saved under `reports/walk_forward/`:

- `protocol.json`: evaluation protocol.
- `fold_metrics.csv`: metrics for each temporal fold and seed.
- `aggregate_metrics.csv`: overall metrics grouped by seed.
- `all_predictions.csv`: all prediction records.
- `walk_forward_comparison.png`: evaluation comparison chart.
- Per-run logs, configurations, training histories, predictions, and charts.

These results provide reproducible historical evidence of near-baseline performance. They do not establish statistical significance, generalization to untouched new data, or economic usefulness after transaction costs.

## Further Evaluation

1. Compare LSTM and Transformer under matched features, horizons, dates, and preprocessing before making empirical architecture claims.
2. Estimate uncertainty with a time-dependence-aware method and check an untouched future period if new data becomes available.
3. Preserve the raw data and record a dataset hash and a full environment lockfile for stronger reproducibility.

## Multivariate LSTM Implementation and Results

`src/lstm_stock_price_prediction.py` uses five price features: `Open`, `High`, `Low`, `Close`, and `Adj Close`. StandardScaler is fitted only on the training period. Each input contains 90 historical trading observations; the output is a single close at the 60th trading observation after the final input, not the entire 60-day price path.

The network consists of LSTM(64), Dropout, LSTM(10), Dropout, and Dense(1), with early stopping, learning-rate reduction, and best-model checkpointing.

The first 59 targets in both validation and test periods are excluded so that training and model-selection labels are already available when later forecasts are issued. This leaves 2,655 training, 542 validation, and 542 test samples.

The run stopped after 11 epochs. The best checkpoint was epoch 1, with validation MSE of 4.672148 in standardized space.

| Test metric | LSTM | Horizon-matched last-observed-close baseline |
| --- | ---: | ---: |
| MAE | 646.3543 | 124.1356 |
| RMSE | 659.0420 | 144.6645 |

The LSTM substantially underpredicts later prices and does not beat the baseline. This baseline uses the last close available when the forecast is issued: the price 60 trading observations before the target.

Artifacts are saved in `reports/lstm/`, including the model, scaler parameters, configuration, training history, predictions, and chart. Three additional tests check the forecasting horizon and temporal boundaries.

The current LSTM and Transformer use different input features and forecasting horizons, so their errors cannot be compared directly to determine which architecture is better.
