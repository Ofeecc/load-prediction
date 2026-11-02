# 月度负荷概率预测：近期信息应该影响多少？

在每个月开始前，一次性预测下一整个月每小时负荷的99个分位数。研究重点是：历史天气提供多种未来可能，近期天气提供局部参考，如何决定参考程度，并避免调整预测中心时无意压缩天气情景之间的差异。

**当前结论**：门控能减轻无条件近期修正造成的损失，但相对不修正的收益跨年份并不稳定。额外相似性加权不是普遍有效的改进，保持温度情景差异也不保证负荷预测损失更低。完整证据和失败月份见[Report.md](Report.md)。

## 数据与任务

- GEFCom2014-L，一个匿名负荷区域、25个匿名气象站；[数据发布者说明](https://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html)。原始CSV已在`Data/Load/`，引用资料在`docs/`。保留原始数据来源及使用条件。
- 2001—2011年天气、2005—2011年负荷，共96,408小时、61,344个负荷标签。
- 各月有672—744小时，按业务小时1—24对齐。月中不更新观测；未来真实天气仅用于明确标识的Oracle诊断。

## 本轮验证与合并范围

1. 先保留队友0506c2b的默认行为，仅加入可选对照开关；90项原有指标一致。冻结原规则，在2008—2009年24个月和已探索竞赛15个月上验证。
2. 比较“是否门控”与“逐条锚定/整体平移”四组温度专用对照，关闭额外残差修正和收缩，隔离数据处理机制。
3. 再选择性合并时间校验、统一节假日、真实时间衰减权重、评分检查、统一加权分位数口径与可追溯回测。同样39个月重新运行，完整保存预测并复核。
4. 保留队友的默认策略与继承趋势基函数，不整包合并旧分支，不因当前成绩改默认方案；较早年份迁移仍是回顾性验证。

详细取舍见[docs/merge_decisions.md](docs/merge_decisions.md)。上一版报告可在[0506c2b](https://github.com/Ofeecc/load-prediction/blob/0506c2b/Report.md)查看。

## 结果文件

| 路径 | 含义 |
| --- | --- |
| `results/boundary_before_merge/` | 队友原规则的39个月冻结验证，默认行为与原表一致 |
| `results/boundary_merged/` | 选择性合并后的正式结果、配置、图和核验记录 |
| `monthly_metrics.csv` | 每月、方法、预测时距的全部指标 |
| `summary.csv` | 按小时合并指标，另列月平均损失与获益月份数 |
| `paired_effects.csv` | 固定其他条件的单因素配对 |
| `scenario_audit.csv` | 历史片段起止、首末小时权重及有效情景数 |
| `protocol.json` | 执行前冻结配置、Git状态、代码/数据哈希、包版本 |
| `verification.json` | 从完整预测重算后的核验结果 |

原`results/tables/`、`results/figures/`、`results/predictions/`保留为旧探索结果。当前结果应从`boundary_merged`读取。完整双精度预测保存在本机`boundary_merged/predictions/`并被Git排除，可由同一命令重新生成。

## 运行

推荐Python 3.12。在仓库根目录安装`requirements.txt`后：

```powershell
python -m unittest discover -s tests -v
python -m src.boundary_validation --output results/my_validation --save-predictions
python scripts/summarize_boundary.py --input results/my_validation --report results/my_validation/Report.md
```

输出目录必须不存在，防止覆盖已有实验。可用`--months 2011-03`做单月运行检查；单月结果不能代替完整39个月验证。报告脚本的跨时期对照表针对默认完整协议。

`python -m src.run_experiments`仍可运行队友的完整探索实验，输出到`results/exploratory_rerun/`，避免覆盖历史归档。

## 核心代码

- `src/models.py`：分时样条Ridge、门控、可选温度修正方式与统一加权分位数。
- `src/boundary_validation.py`：冻结规则的跨时期回测；预测前隐藏目标真值，预测保存后评分。
- `src/data_loader.py`、`src/metrics.py`：原始行校验、基线对齐和严格评分输入检查。
- `tests/test_forecast_invariants.py`：11项真实数据和分布性质检查。
- `scripts/summarize_boundary.py`：复核完整预测、来源和汇总，再生成报告。

下一步应研究门控规则的时间迁移和样本外残差校准，保留不修正基线与所有失败月份，不继续按测试月份调阈值。
