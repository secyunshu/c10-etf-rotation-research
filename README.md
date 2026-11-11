# ETF动量轮动：当前默认V2.2

**当前默认：前两只ETF＋每年年底重建基金池。** 用户于2026-11-11明确采用年度换池政策。252交易日总回报动量、月度调仓、风险分组、SHY防守和原仓位/费用规则保持不变。

[当前政策指针](standard/CURRENT.json) · [机器配置](standard/current/STRATEGY.json) · [完整当前规则](docs/CURRENT_STANDARD.md) · [回测与证据](experiments/v2-annual-repool/FINAL_REPORT.md) · [机制及来源反证](experiments/annual-pool-falsification/FINAL_REPORT.md)

## 默认方案如何运行

- **每年建立池**：用截至年底可获得的规模资料，在原95候选/36类别内筛选AUM≥50亿美元、每类规模最大的代表；保留21经济风险组。下一年首次交易时启用新池。
- **每月选持仓**：检查253点历史、价格≥5美元、20日日均成交额≥1000万美元且20天均有成交，然后按252日总回报排名，同组最多一只，取前两槽。
- **防守与比例**：每个通过“动量严格超过SHY”的风险槽目标42.75%；两槽通过时SHY9.5%、现金约5%。失败槽转SHY，不补位。年度换池不重置账户本金或应收分红。
- **时点**：价格取月末最后交易日收盘，下一交易日开盘执行。年度资料截止日、价格日和执行日分别记录，不能使用执行后才可得的资料。

## 2022–2025回测

单一50,000美元连续账户、原费用及单边5bp滑点、整股、税0，2025最后交易日开盘清仓。每格为年度净收益／年内最大回撤。

|年份|年度换池：当前默认|固定池：历史对照|SPY持有|QQQ持有|
|---|---:|---:|---:|---:|
|2022|24.32%／20.23%|24.32%／20.23%|−17.39%／23.04%|−30.88%／32.79%|
|2023|12.78%／13.62%|3.21%／16.83%|23.85%／9.18%|49.77%／10.05%|
|2024|25.20%／13.35%|14.36%／12.62%|22.80%／7.79%|23.84%／12.78%|
|2025|50.53%／10.77%|50.92%／10.76%|17.01%／17.35%|20.41%／21.48%|

|方案|累计收益|年化收益|完整最大回撤|夏普 rf0|
|---|---:|---:|---:|---:|
|**V2.2年度池前两只**|**164.24%**|**27.65%**|**20.23%**|**1.32**|
|V2.1固定池前两只|121.46%|22.11%|21.36%|1.12|
|QQQ持有|54.36%|11.52%|32.79%|0.61|
|SPY持有|47.02%|10.17%|23.04%|0.67|

20bp滑点下年度方案年化26.53%、完整回撤20.22%。SPY/QQQ沿95%初始投入、分红留现金同费账户，非网站100%分红复投曲线。年度回撤与完整跨年回撤不同。

**年度池的2026表现尚未验证。** 历史固定池2026结果仍在[旧V2报告](standard/results/RESULTS.md)，不能称为当前年度池的2026收益。当前包只包含已验收的2022–2025年度名单，不会自动沿用2025池替代缺失年份。

## 当前默认入口

复用成熟Backtrader执行和独立财务审计，不需要API账号即可离线验证。未传策略参数时就是年度池前两只；默认只读参考核验与报表重算，显式`--execute`才启动原生复现。

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
export PYTHONPATH=src:.
python -m scripts.current_standard
# 实际复现：两固定池桥接＋两年度池成本账户
python -m scripts.current_standard --execute --output runs/current-standard
# 默认政策及相关回归测试
python -m pytest -q standard/current/tests experiments/v2-annual-repool/tests --typeguard-packages=scripts
```

已完成目录只复核，不重跑；失败留证，发布复现独立预算4计划／8上限。报告生成于输出目录的`report/`。[详细复现与历史入口说明](docs/REPRODUCE_V2.md)。

政策版本及Git标签为**v2.2.0**；原兼容运行包的`pyproject.toml`版本2.1.0保持冻结，用于旧研究源码锁，不代表当前默认政策。以`standard/CURRENT.json`为准。旧`python -m c10standard.run`和`standard/STRATEGY.json`明确是固定池历史复现入口；当前默认入口是`python -m scripts.current_standard`。

## 证据边界

这是用户在看过历史结果后作出的政策选择，不是新的样本外通过或实盘资格。旧归档规模日期不同步，部分2023/24页面早于年底220/225天；AUM专属经济日及历史退市覆盖不足，仍为条件性历史重建。

机制审查显示增益高度集中于2023年的SOXX→SMH；2024/25继续更新没有改变最终目标。另找到INDA当年更新的发行商资料，与所用陈旧快照的门槛状态不同。以上[完整反证](experiments/annual-pool-falsification/FINAL_REPORT.md)保留，不因采用默认而删除；历史名单及收益没有悄悄修正。未来年度输入必须另行核验来源、截止和资料陈旧度。

此入口用于已冻结历史复现，没有券商下单接口，也不是实时选基程序。真实开盘成交、购买力、冲击、税费与未来独立样本仍未证明。代码GPL-3.0-or-later；第三方市场数据权利见[NOTICE](NOTICE.md)。

## 历史与研究索引

- [V2.1固定池规则与对照](docs/STANDARD_V2.md) · [原README快照](standard/current/evidence/prior_v2_1/README.md)
- [年度池数据/源码/完整参考账本](experiments/v2-annual-repool/README.md)
- [年度机制证伪与来源审计](experiments/annual-pool-falsification/README.md)
- [2023新起点重建池](experiments/v2-2023-start/README.md) · [条件QQQ底仓](experiments/qqq-overlay/README.md)
- [三个价格路径指标](experiments/wechat-momentum-paths/README.md) · [13612W](experiments/calendar13612w/README.md) · [六动量变种](experiments/top2-momentum-variants/README.md)
- [V1复现](docs/V1_OVERVIEW.md) · [原研究测试归档与范围](experiments/source_archive/README.md)
