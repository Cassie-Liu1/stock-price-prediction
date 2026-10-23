# GOOG 股票价格预测：多特征 LSTM 与残差 Transformer

[English](README.md) | [中文技术分析](docs/model-review.zh-CN.md)

本项目使用 4,006 条 GOOG 历史交易记录，展示时间序列模型的构建、问题诊断和评估流程，包含两个可运行的预测任务：

- **多特征 LSTM**：使用过去 90 个交易记录的五个价格特征，预测最后一个输入之后第 60 个交易记录的收盘价。
- **残差 Transformer**：使用过去 60 个交易记录的收盘价，预测下一个交易记录的收盘价。

项目的主要成果是可复现的模型评估。直接价格预测曾明显低估后期价格；改成残差 Transformer 后，这一问题大幅改善，但多个时间段的验证表明，它整体仍接近简单价格基线。新增 LSTM 虽已完整运行，但明显弱于相同预测跨度的基线。

技术栈：Python、TensorFlow/Keras、PyTorch、pandas、NumPy、scikit-learn、Matplotlib、unittest。

## 主要结果

Transformer 扩展窗口滚动验证包含四个不重叠的测试时间段、三个随机种子，共 12 次实验，覆盖 1,354 个不同交易日期。

| 总体指标 | 残差 Transformer：三个种子的范围 | 前一天收盘价基线 |
| --- | ---: | ---: |
| MAE：平均绝对误差 | 10.9011–10.9077 | 10.9425 |
| RMSE：均方根误差 | 17.4827–17.4858 | 17.5010 |

MAE 在 12 次实验中有 11 次更低，但总体改善只有约 0.32%–0.38%。这些实验共享历史数据，不是相互独立的统计试验。模型修改曾参考旧测试结果，因此这些结果属于历史开发和稳定性评估，不能视为从未接触过的最终测试。

![滚动验证：模型与基线对比](reports/walk_forward/walk_forward_comparison.png)

可查看 [各次实验指标](reports/walk_forward/fold_metrics.csv)、[总体指标](reports/walk_forward/aggregate_metrics.csv) 和 [中文技术分析](docs/model-review.zh-CN.md)。

## 快速运行 Transformer

以下命令在项目根目录执行。已完成实验使用 Python 3.12、macOS ARM64 和 CPU。

```bash
cd stock-price-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/transformer_stock_price_prediction_original.py --input data/GOOG.csv --output-dir reports/reproduced
python src/walk_forward.py --input data/GOOG.csv --output-dir reports/walk_forward_reproduced
```

两个复现目录与已保存结果分开，方便对照。滚动验证需要训练 12 个模型，耗时会比单次划分更长。`requirements.txt` 记录主要依赖的运行版本，不是完整的间接依赖锁定文件；不同平台可能产生数值差异。

Transformer 默认采用残差预测：

```text
预测收盘价 = 前一天已知收盘价 + 模型预测的变化量
```

脚本文件名虽然保留 `original`，但代码已修正，默认运行的是残差模型。

## 运行多特征 LSTM

LSTM 使用独立虚拟环境 `.venv-lstm`。现有 Anaconda 环境曾出现原生扩展冲突，因此建议按以下方式运行。

```bash
cd stock-price-prediction
python -m venv .venv-lstm
source .venv-lstm/bin/activate
python -m pip install -r requirements-lstm.txt
python -m unittest discover -s tests -p 'test_lstm_windows.py' -v
python src/lstm_stock_price_prediction.py --input data/GOOG.csv --output-dir reports/lstm_reproduced
```

输入特征为 `Open`、`High`、`Low`、`Close`、`Adj Close`，使用仅在训练期拟合的 StandardScaler。预测目标 Close 有单独的训练期标准化参数，用于将预测还原到实际价格。

网络结构为：

```text
90 × 5 输入 → LSTM(64，返回完整序列) → Dropout(0.2)
            → LSTM(10) → Dropout(0.2) → Dense(1)
```

输出是第 60 个未来交易记录的单个收盘价，不是全部 60 个未来价格。窗口长度和隐藏单元数量属于固定设计选择，尚未证明是最优设置。

训练使用 Adam、MSE 和三个回调：

- `EarlyStopping`：验证误差连续 10 轮不改善时停止。
- `ReduceLROnPlateau`：连续 5 轮不改善时，学习率减半。
- `ModelCheckpoint`：保存验证集表现最好的模型。

最多训练 100 轮，实际停止轮次和结果以运行记录为准。

数据按时间设置 70%/15%/15% 边界。验证和测试区间各排除前 59 个目标，确保后续预测发出时，训练及模型选择使用的标签都已可知。最终有 2,655 个训练样本、542 个验证样本和 542 个测试样本。重叠窗口不会创造新的独立观测。

### 已完成的 LSTM 实验

随机种子 42 的运行在第 11 轮停止，最佳模型为第 1 轮；标准化空间下的最佳验证 MSE 为 4.672148。

| 测试指标 | LSTM | 相同预测跨度的最近已知价格基线 |
| --- | ---: | ---: |
| MAE | 646.3543 | 124.1356 |
| RMSE | 659.0420 | 144.6645 |

LSTM 严重低估后期价格，未超过基线。这里的基线使用预测发出时最后一个已知价格，即目标日期之前 60 个交易记录的收盘价。

`best_model.keras` 仅在本地保存；`reports/lstm/` 提交标准化参数 `scalers.npz`、训练历史、配置、验证/测试预测、指标和图表。旧实验中的 MSE 0.14366 未被复现，不能作为当前结果。

LSTM 和 Transformer 的输入特征、预测跨度不同，不能直接用上述误差比较两种架构的优劣。旧 notebook 和权重已移出当前项目，新 LSTM 不加载这些旧权重。

## 数据集

`data/GOOG.csv` 包含 4,006 条日线记录，时间为 2004-08-19 至 2020-07-17；包括表头共 4,007 行。

| 字段 | 含义 |
| --- | --- |
| Date | 交易日期 |
| Open | 开盘价 |
| High | 当日最高价 |
| Low | 当日最低价 |
| Close | 收盘价 |
| Adj Close | 调整后的收盘价 |
| Volume | 成交量 |

原始文件由项目所有者提供；尚未确认原始供应方和再分发条件，CSV 仅在本地保存，不随新提交分发。复现前需自行准备相同字段及时间范围的数据。当前 Transformer 仅使用 Close，LSTM 使用上述五个价格特征；两者都使用 Date 排序。

## 文件结构

```text
.
├── data/GOOG.csv                                   # 原始数据
├── src/
│   ├── transformer_stock_price_prediction_original.py  # 单次 Transformer 实验
│   ├── walk_forward.py                            # Transformer 滚动验证
│   └── lstm_stock_price_prediction.py              # 多特征 LSTM
├── tests/
│   ├── test_forecasting.py                        # Transformer 与滚动边界检查
│   └── test_lstm_windows.py                       # LSTM 窗口和预测跨度检查
├── reports/
│   ├── absolute_original/                        # 修改前的直接价格预测结果
│   ├── walk_forward/                             # 12 次滚动验证及汇总
│   ├── lstm/                                     # LSTM 模型和结果
│   └── ...                                       # 单次残差 Transformer 结果
├── docs/
│   ├── model-review.md                           # 英文技术分析
│   └── model-review.zh-CN.md                      # 中文技术分析
├── requirements.txt                              # Transformer 依赖
├── requirements-lstm.txt                         # 独立 LSTM 环境依赖
├── .gitignore                                    # 忽略缓存和虚拟环境
├── README.md                                     # 英文项目说明
└── README.zh-CN.md                                # 中文项目说明
```

## Transformer 单次划分结果

按时间顺序划分 70%/15%/15% 的训练、验证和测试数据，MinMaxScaler 只在训练期拟合。随机种子为 42，历史窗口为 60 条，训练 30 轮，验证集选中第 17 轮。测试期包含 601 条记录，时间为 2018-02-28 至 2020-07-17。

| 方法 | 测试 MAE | 测试 RMSE |
| --- | ---: | ---: |
| 直接价格 Transformer：旧运行 | 313.1386 | 329.7129 |
| 残差 Transformer | 15.5973 | 23.3166 |
| 前一天价格基线 | 15.6358 | 23.3316 |

残差模型相对基线的 MAE 改善约 0.25%，RMSE 改善约 0.06%。这是单一种子、单一历史区间的结果，没有显著性检验。旧直接价格运行采用不同的验证误差汇总方式，因此改进前后不是完全控制变量的消融比较。

旧结果保存在 `reports/absolute_original/`；新版记录 `training_history.csv` 和 `run_metadata.json`。验证 MSE 按观测数等权汇总，模型选择包含第 0 轮的基线状态。

复现直接价格版本可运行：

```bash
python src/transformer_stock_price_prediction_original.py --mode absolute --output-dir reports/absolute_rerun
```

## Transformer 扩展窗口验证

初始训练数据为 2,400 条，随后 252 条用于验证，再测试 400 条；每轮训练区间扩大 400 条，最后测试剩余 154 条。四个测试区间不重叠。每段使用种子 42、7、123，分别从头训练 30 轮，独立拟合预处理参数和选择最佳模型。

| 测试时间段 | 记录数 | Transformer MAE：三个种子均值 | 前一天价格基线 MAE |
| --- | ---: | ---: | ---: |
| 2015-03-04 至 2016-09-30 | 400 | 7.0131 | 7.0139 |
| 2016-10-03 至 2018-05-04 | 400 | 8.4156 | 8.4529 |
| 2018-05-07 至 2019-12-05 | 400 | 12.8078 | 12.8267 |
| 2019-12-06 至 2020-07-17 | 154 | 22.5363 | 22.7197 |

每个测试区间内不重新训练，但已经观察到的测试价格可以进入下一个预测窗口。较早测试期数据在后续轮次成为历史后，也可以进入训练或验证区间。

`fold_metrics.csv` 记录每次实验；`aggregate_metrics.csv` 按每个种子的全部 1,354 个测试日期计算 MAE/RMSE，不直接平均各段 RMSE，也不把三个种子对同一天的预测视为独立样本。各次运行的日志、配置、历史、预测和图表均保留。

总体 MAE 改善约 0.32%–0.38%，RMSE 改善约 0.087%–0.105%。第一个时间段基本与基线持平，未根据测试结果挑选种子。当前结果说明模型接近基线，尚未建立具有实际意义的预测优势。

## 结果解读与局限

本项目定位为时间序列模型诊断和可复现评估案例。历史股票价格具有明显的分布变化，现有实验不能证明可部署的交易信号、盈利能力或统计显著的预测优势。更强的泛化结论需要新的未参与开发的数据，以及考虑时间依赖的不确定性分析。


Dataset provenance: the owner-supplied GOOG.csv is local only. The original provider and redistribution terms have not been established. Prepare compatible data independently; no dataset license is granted by this repository.
