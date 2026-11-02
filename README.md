# GEFCom2014-L 电力负荷概率预测（大数据课程作业）

> **选题来源**：Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, Rob J. Hyndman (2016), *"Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond"*, **International Journal of Forecasting**, 32(3), 896–913（第 4 章：Probabilistic electric load forecasting, `GEFCom2014-L` & `GEFCom2014-E`）。
>
> **核心理念**：**拒绝模型堆砌，坚持从场景物理与业务矛盾出发进行启发式建模**。围绕“静态预测接下来一个月（744 小时）每小时负荷分位数与期望值”的任务设定，用透明可解释的**分时惩罚样条广义加性模型（Hourly Penalized Spline GAM）**与**两阶段正交不确定性分解**，通过三组递进式有限实验完整回答论文第 4.1 节提出的三大赛道核心挑战。

---

## 目录

1. [核心研究问题与三大启发式洞察](#1-核心研究问题与三大启发式洞察)
2. [仓库目录结构](#2-仓库目录结构)
3. [环境配置与一键复现](#3-环境配置与一键复现)
4. [核心实验结论与图表速览](#4-核心实验结论与图表速览)
5. [团队协作与分工指南](#5-团队协作与分工指南)

---

## 1. 核心研究问题与三大启发式洞察

在静态提前一个月（744 小时）预测每小时负荷的 99 个分位数（$q = 0.01, 0.02, \dots, 0.99$）时，盲目堆叠 LSTM/Transformer/GBDT 等黑盒模型往往治标不治本。我们从数据与物理场景本身挖掘出三个关键切入点：

1. **为什么主办方提供了 2001–2004 年“没有负荷标签”的气温数据？（回应挑战 3：Scenario Generation）**
   - 负荷记录始于 2005 年（7 年），而 25 个气象站的气温记录始于 2001 年（11 年）。在预测未来一个月时，**未来真实气温完全未知**。
   - 我们将月度负荷不确定性正交分解为**气候轨迹认知不确定性（Stage 1）**与**给定气温下的人类行为条件残差不确定性（Stage 2）**。通过引入 2001–2004 年无标签历史气象库（其中 2001 年 12 月 `49.5°F` 恰为 2011 年 12 月 `48.5°F` 暖冬的最佳物理原型）并结合 $\pm 6$ 天天气窗口平移构造 70 条物理连续温度情景，**极端尾部 Pinball Loss 下降 18.4%，总体 Pinball Loss 较官方 Benchmark 下降 73.8%（`34.07 → 8.91`），90% 预测区间覆盖率达 `89.2%`**。
   - 对照“已知未来真实气温（Oracle Upper Bound，`Pinball = 2.14, MAPE = 3.70%`）”严谨证明：**超过 76% 的月度负荷预测误差源于未来天气轨迹未知，而非负荷回归函数拟合不足**。

2. **点预测最优的气象站 = 概率预测最优的气象站吗？（回应挑战 1：Weather Station Selection）**
   - 对全部 25 个匿名气象站（`w1`–`w25`）的单站扫描表明：单站在已知真实气温下的点预测 RMSE 与未知气温下的月度概率预测 Pinball Loss 呈**负相关（Spearman $\rho = -0.39$）**。
   - 我们在无先验干预下，通过前向贪心 GCV 与两阶段指数加权（EWA）算法，**独立复现出论文 Table 6 中冠军 `Tololo`（`w10, w6, w13, w22`）与亚军 `Adada`（`w6, w13, w10`）选出的核心气象站组合**，将总体 Pinball Loss 进一步推进至 **`8.86`**（Oracle 点预测 RMSE 降至 **`6.22 MW`**）。

3. **月首短期天气惯性能维持多久？联邦节假日如何改变负荷流形？（回应挑战 2：Multi-Horizon & Table 5）**
   - **月首 48 小时平滑截断（`Tololo` Cutoff）**：利用月末已知的最后 24 小时昼夜温差异常与负荷残差，在预测月前 48 小时内做平滑锥度衰减融合，使 **1–48h 短期窗口的 Pinball Loss 下降 17.4%（`5.25 → 4.34`）、RMSE 下降 12.3%（`17.90 → 15.69 MW`）**，且避免污染 49 小时后的天气回暖转型期，从大气物理机制上验证了冠军 `Tololo` 以 48 小时作为短/中期分界线的合理性。
   - **Table 5 联邦节假日“工作日→周末相移”**：将落在工作日的重大联邦节假日及桥接日（如感恩节、黑色星期五、圣诞节周一补休）相移映射至周末日型，使感恩节窗口的负荷预测 RMSE 从 `14.05 MW` 骤降至 **`10.38 MW`（降幅 26.1%）**。

---

## 2. 仓库目录结构

```text
load-prediction/
├── README.md                         # 项目总览、复现指南与团队协作文档
├── Report.md                         # 完整方法论证与三组实验分析报告（含图表）
├── docs/
│   └── Paper_Section4_Reference.md   # 论文第 4 章原文与 Table 5/6 整理（含原图链接）
├── requirements.txt                  # Python 依赖包列表
├── Data/
│   ├── figs/                         # 论文第 4 章原始配图 (Fig2.jpg ~ Fig7.jpg)
│   └── Load/                         # GEFCom2014-L 原始数据集 (Task 1..15 及 Task 15 真值)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # 数据整合、时间戳对齐、Table 5 联邦节假日与桥接日特征工程
│   ├── metrics.py                    # 99分位数 Pinball Loss（总体/尾部/中心）、RMSE/MAE/MAPE/PICP
│   ├── models.py                     # 分时三次样条 GAM、建筑热惯性、两阶段情景卷积与48h短期截断
│   ├── station_selection.py          # 25个气象站全扫描与 Table 6 五种选站/EWA策略对比
│   └── run_experiments.py            # 实验主入口：一键运行全部三组实验并输出表格与高清图表
└── results/
    ├── figures/                      # 自动生成的 4 张核心实验可视化图 (fig1 ~ fig4)
    └── tables/                       # 自动生成的 5 张实验定量结果 CSV 表格
```

---

## 3. 环境配置与一键复现

推荐使用 `uv` 或标准 `venv` + `pip` 创建虚拟环境（Python 3.10+ 均可）：

```bash
# 方式一：使用 uv（极速推荐）
uv venv .venv
uv pip install -r requirements.txt
MPLCONFIGDIR=/tmp .venv/bin/python -m src.run_experiments

# 方式二：使用标准 python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_experiments
```

运行 `python -m src.run_experiments` 约需 **15–20 秒**，将在终端打印完整三组实验对比表，并自动更新 `results/tables/*.csv` 与 `results/figures/*.png`。

---

## 4. 核心实验结论与图表速览

详细文字分析与公式推导见 [**Report.md**](./Report.md)。

### 图 1：场景物理特征勘测与论文 Fig. 2–4 复刻深化
![场景物理特征探索与论文图表复刻](./results/figures/fig1_eda_physics.png)

### 图 2：实验一 —— 静态单月（744h）概率预测扇形图与 99 分位数 Pinball Loss 曲线
![实验一：静态单月概率预测扇形图与Pinball曲线](./results/figures/fig2_exp1_scenarios.png)

### 图 3：实验二 —— 25 气象站点预测 vs. 概率预测异质性及 Table 6 选站策略对比
![实验二：25气象站异质性与选站策略对比](./results/figures/fig3_exp2_stations.png)

### 图 4：实验三 —— 月首 48 小时短期持久性截断与 Table 5 联邦节假日相移修正
![实验三：月首48h截断与联邦节假日相移修正](./results/figures/fig4_exp3_horizon_holiday.png)

---

## 5. 团队协作与分工指南

为了方便团队成员并行推进后续课程报告润色、补充实验与答辩展示，建议按以下模块分工：

### 模块 A：代码扩展与更多静态月份横向对比（面向偏工程/算法成员）
- **接口位置**：[`src/run_experiments.py`](./src/run_experiments.py) 中已支持任意月份切分（目前默认跑了 `2011-12` 冬季、`2011-11` 秋季感恩节、`2011-07` 夏季高温峰值）。
- **可扩展点**：
  1. 在 `results/tables/exp1_scenario_generation.csv` 基础上，将春季过渡月（如 `2011-04` 舒适区拐点月）纳入静态对比，展示温度处于 `58°F–65°F` 拐点区时的分位数形态变化。
  2. 如果课程加分项涉及分布式大数据框架（如 Spark），可将 [`src/station_selection.py`](./src/station_selection.py) 中 25 个气象站 × 70 条情景的并行评估封装为 `PySpark` / `joblib` 多进程并行算子。

### 模块 B：实验图表中文排版与课程报告撰写（面向偏论文/文档成员）
- **文档基座**：当前 [**Report.md**](./Report.md) 已包含完整的四节结构（痛点剖析、方法数学表述、三组实验定量表格、启发式结论）。
- **可扩展点**：
  1. 若课程报告要求全中文图表标注，可在 [`src/run_experiments.py`](./src/run_experiments.py) 中配好中文字体（如 `SimHei` / `Noto Sans CJK SC`）并将图题与图例切换为中文。
  2. 结合 `Data/figs/Fig2.jpg`–`Fig7.jpg` 补充对 2015 年课堂扩展赛（`GEFCom2014-E`）背景的文献综述段落。

### 模块 C：答辩 PPT ストーリー（Storyline）设计（面向汇报成员）
建议答辩 PPT 严格遵循**“立靶子 → 拆机理 → 三连实验求证”**的叙事节奏（凸显启发式思考，契合老师偏好）：
1. **第一幕（破题）**：展示 `fig1_eda_physics.png`，抛出三个反直觉问题——*为什么多给 4 年无负荷气温？为什么 2010 年负荷暴涨而 2011 年回落？为什么点预测准的气象站做概率预测反而失准？*
2. **第二幕（方法）**：展示“两阶段正交不确定性分解 + 分时样条热力学 GAM”，强调**零黑盒堆砌、全物理可解释**。
3. **第三幕（实证）**：依次展示 `fig2`（证明 >76% 误差来自未来天气未知，2001–2004 气象库降低尾部风险 18.4%）、`fig3`（无监督复现冠亚军 `Tololo`/`Adada` 核心气象站 `w6, w10, w13, w22`）、`fig4`（解释 `Tololo` 48h 截断的天气学成因及感恩节误差下降 26.1%）。

### Git 协作工作流约定
```bash
# 1. 克隆仓库并安装环境
git clone https://github.com/Ofeecc/load-prediction.git
cd load-prediction
uv venv .venv && uv pip install -r requirements.txt

# 2. 创建个人特性分支进行修改
git checkout -b feat/your-name-task

# 3. 提交前确保实验脚本可正常跑通
MPLCONFIGDIR=/tmp .venv/bin/python -m src.run_experiments
git add .
git commit -m "feat: 简明描述你的改动"
git push origin feat/your-name-task
```
