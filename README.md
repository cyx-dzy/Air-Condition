# 光储荷微电网 Python 复现项目

这个项目把 `项目与代码.docx` 里的 MATLAB 演示代码改写成 Python 工程。你可以把它理解成一个体育馆微电网的“小脑袋”：

1. 先看历史数据，学习负荷和光伏的规律。
2. 预测明天 24 小时的用电和发电。
3. 生成多种可能发生的明天。
4. 在这些不确定场景下安排电池充放电。
5. 输出图表和 CSV，方便展示和继续分析。

## 快速运行

使用模拟数据跑完整流程：

```bash
python -m microgrid_pipeline demo
```

使用真实 CSV：

```bash
python -m microgrid_pipeline run --data data/real/sample.csv
```

运行后会生成 `outputs/`：

- `predictions.csv`：未来 24 小时负荷和光伏点预测。
- `scenarios_load.csv`：负荷多场景。
- `scenarios_pv.csv`：光伏多场景。
- `schedule.csv`：储能充放电、SOC、成本等调度结果。
- `figures/`：两张展示图。

## 真实数据格式

最低需要这些列：

```csv
timestamp,temperature_c,load_kw,pv_kw,event_flag
2026-07-01 00:00,25.1,118.5,0,0
2026-07-01 19:00,29.3,520.4,0,1
```

字段解释：

- `timestamp`：时间。
- `temperature_c`：气温。
- `load_kw`：体育馆用电负荷。
- `pv_kw`：光伏发电功率。
- `event_flag`：是否有大型活动，`0` 普通日，`1` 活动日。

## 目前实现范围

已经实现：

- 模拟数据生成。
- 真实 CSV 入口。
- PyTorch 的 LSTM + 多头注意力预测模型。
- 基于训练残差的多场景生成。
- SciPy 开源 MILP 调度。
- CVaR 风险项。
- CSV 和 PNG 输出。

暂未实现：

- TCN-DAE 数据清洗。
- DEC 深度聚类。
- 严格数学意义上的 Wasserstein DRO。

这些可以作为后续论文增强版继续补。

