# NeuralAlpha-Ashare

面向 A 股的本地、严格时点（PIT）、横截面机器学习选股与可成交回测平台。

项目默认针对一台 6 核 CPU、RTX 3060 6GB、16GB 内存的 Windows 电脑设计：主模型使用
CPU LightGBM，Ridge 作为低方差基线；数据和特征使用 Polars/PyArrow 的列式、分区与增量计算，
避免逐行 Python 热路径。GPU 挑战模型是可选项，不影响默认闭环运行。

> 本项目用于研究与工程验证，不构成投资建议，也不会自动向券商下单。

## 核心约束

- 信号在 `t` 日收盘后产生，最早在 `t+1` 开盘成交。
- 5/20/60 个交易日标签分别训练，先在当日横截面内排名，再按 0.2/0.5/0.3 合成。
- Purged Walk-Forward，验证集负责调参，测试集只负责最终评估。
- 长仓；默认排除 ST、北交所和一字涨停，不在涨停板追单。
- 支持停牌、板块/日期涨跌停规则、T+1、100 股整手、冲击成本与成交容量。
- 股票单边佣金 `0.00008499999`，ETF/LOF/基金单边佣金 `0.00005000001`，最低佣金 `0`；股票卖出印花税默认 `0.0005`。
- 默认实盘研究资金 20 万元，另提供 100 万元容量对照。

## 快速开始（Windows / Python 3.11）

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,fast]"
alpha-ashare doctor
alpha-ashare update
alpha-ashare build
alpha-ashare train
alpha-ashare walk-forward
alpha-ashare backtest
alpha-ashare daily
```

也可以双击 `run_gui.bat`。GUI 只调用同一套 CLI，不复制任何研究或回测逻辑。

## 数据

默认使用 `TickFlow.free()` 获取公开行情与证券目录，不需要 API Key。原始行情按年写入
`data/raw/year=YYYY/`，派生数据按年写入 `data/derived/year=YYYY/`。首次全量下载会比较慢，
以后仅拉取最近一小段重叠窗口并去重，便于修订和断点恢复。

严格历史回测需要当时可见的证券名称、状态和上市信息快照。若数据源只能返回“当前目录”，
系统会在研究清单中明确标记 `DEGRADED`，而不是假装没有幸存者偏差。

## 常用命令

```text
alpha-ashare doctor                       检查环境和数据状态
alpha-ashare update [--start ... --end]   增量更新 TickFlow 行情
alpha-ashare build [--year 2025]          构建 PIT 特征与标签
alpha-ashare train [--model lightgbm]      训练候选模型并登记
alpha-ashare walk-forward                  生成样本外预测与评估
alpha-ashare backtest [--capital 200000]   运行可成交组合回测
alpha-ashare daily                         更新、推理并输出日报
alpha-ashare weekly                        输出周报
alpha-ashare gui                           启动桌面界面
```

所有默认值集中在 `config/default.yaml`。路径相对于项目根目录解析，也可以通过 `--config`
传入另一份配置。

## 目录

```text
src/neural_alpha_ashare/  唯一业务实现
config/                   可审计配置
data/                     分区行情、特征和清单（不提交 Git）
models/                   模型与 champion/challenger 注册表
predictions/              样本外和每日预测
backtests/                净值、持仓和成交记录
docs/                     可直接由 GitHub Pages 托管的静态报告
tests/                    泄漏、规则、费用、切分和模型测试
```

## 设计说明

LightGBM 在这类中小规模表格横截面数据上通常比直接上深度网络更稳、更省内存，也更容易诊断。
RTX 3060 保留给可选 CatBoost GPU 挑战实验；Windows 下默认不依赖 LightGBM CUDA。
模型替换必须通过注册表与样本外指标，不能静默覆盖 champion。

要运行 3060 挑战模型：`pip install -e ".[gpu]"`，然后执行
`alpha-ashare walk-forward --model catboost-gpu`。6GB 显存下会自动把单折训练限制在 120 万行。

标签公式、切分隔离、幸存者偏差状态与 16GB 内存预算见
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)。
