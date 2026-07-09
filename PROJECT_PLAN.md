# Python 工程化复现计划：光储荷微电网预测与风险调度

## 1. 项目目标

把 `项目与代码.docx` 中的 MATLAB 演示版改写成一个可运行、可解释、可继续扩展的 Python 项目。第一版优先完成完整闭环：

```text
历史数据 -> 负荷/光伏预测 -> 多场景生成 -> CVaR 风险调度 -> 图表与 CSV 结果
```

本项目先复现文档中已经由代码实际打通的主链路。文档里提到但原 MATLAB 代码没有真正实现的 TCN-DAE、DEC、Wasserstein DRO，暂时作为后续增强方向。

## 2. 输入数据

第一版支持两种数据来源：

1. 模拟数据：程序自动生成体育馆逐小时温度、负荷、光伏和活动日特征。
2. 真实 CSV：用户提供逐小时数据，最低字段如下：

```csv
timestamp,temperature_c,load_kw,pv_kw,event_flag
2026-07-01 00:00,25.1,118.5,0,0
2026-07-01 19:00,29.3,520.4,0,1
```

字段说明：

- `timestamp`：时间戳。
- `temperature_c`：气温，影响空调负荷。
- `load_kw`：体育馆总用电负荷。
- `pv_kw`：光伏出力。
- `event_flag`：是否有大型赛事/演艺活动，`0` 表示普通日，`1` 表示活动日。

## 3. 技术路线

- 数据层：生成或读取逐小时数据，整理成按天划分的 24 小时训练样本。
- 预测层：使用 PyTorch 实现 `LSTM + MultiHeadAttention + LSTM + Linear` 网络，分别预测未来 24 小时负荷和光伏。
- 场景层：根据训练残差的逐小时标准差生成多场景，默认生成 10 个场景。
- 调度层：使用 `scipy.optimize.milp` 建立开源混合整数线性规划模型，目标函数综合期望成本和 CVaR 风险成本。
- 输出层：导出预测、场景、调度 CSV，并生成可视化 PNG 图。

## 4. 命令行设计

```bash
python -m microgrid_pipeline demo
python -m microgrid_pipeline run --data data/real/sample.csv
```

输出目录默认为 `outputs/`，包含：

- `predictions.csv`
- `scenarios_load.csv`
- `scenarios_pv.csv`
- `schedule.csv`
- `figures/prediction_scenarios.png`
- `figures/dispatch_schedule.png`

## 5. 验收标准

- `demo` 命令能在没有真实数据的情况下跑完整流程。
- 预测输出必须是 24 小时。
- 场景输出默认是 `10 x 24`。
- 调度结果满足 SOC 不越界、充放电不同时发生、末端 SOC 回到初始容量。
- 真实 CSV 缺少必要字段时，给出清晰错误。
- README 能让初学者理解怎么运行、数据长什么样、每个模块在做什么。

