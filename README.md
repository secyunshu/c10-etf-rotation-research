# 美股 ETF 动量轮动 · V2.3

当前标准采用**年度 ETF 池、固定 252 个交易日总回报、月度 Top 2 和条件底仓**。每只入选 ETF 的目标权重为 45%。前两只包含 QQQ 或 SPY 时，剩余 10% 配 SHY，否则配 QQQ。取消 SHY 动量门槛。

[当前政策指针](standard/CURRENT.json) · [机器配置](standard/v2_3/STRATEGY.json) · [完整规则](docs/CURRENT_STANDARD.md)

- 每年用年底可获得的规模资料，在原 95 个候选和 36 个类别中按至少 50 亿美元规模、每类规模最大选代表，下一交易年启用。
- 每月检查行情历史、价格和流动性，按 252 个交易日总回报排序。同一经济风险组最多选一只。
- 月末最后一个交易日收盘出信号，下一交易日开盘调仓。名单相同时也按目标权重再平衡。整股、费用和成交价格会使实际权重偏离目标。

## 已复现的回测

以 50,000 美元开仓，同一费用及单边 5 bp 滑点，股息计入账户。2022–2025 连续运行至 2025 年末收盘。2026 为年初独立新账户，估值到美东 11 月 11 日收盘。

| 账户 | 2022–2025 累计净收益 | 净年化 | 最大回撤 | 夏普 | 2026 净收益 | 2026 最大回撤 | 2026 夏普 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2.3 轮动 | +177.59% | 29.24% | 21.95% | 1.27 | +12.21% | 26.90% | 0.53 |
| QQQ 长持 | +56.34% | 11.88% | 34.72% | 0.60 | +11.56% | 14.69% | 0.69 |
| SPY 长持 | +48.39% | 10.42% | 24.20% | 0.65 | +11.42% | 8.77% | 0.98 |

[历史逐年报告](experiments/annual-qqq-spy-sleeve/REPORT.md) · [2026 年初开仓报告](experiments/conditional-sleeve-2026/REPORT.md)

## 离线运行

从仓库根目录使用锁定的 Python 依赖。

```bash
export PYTHONPATH=src:.
python -m scripts.current_standard
python -m scripts.current_standard --execute --output runs/current-standard-v2-3
python -m pytest -q standard/v2_3/tests
```

默认命令核验来源和存档账本，再重算报告。`--execute` 复现六个 Backtrader 账户并逐账比较。重复运行同一输出目录会复核已有账户。两条命令均使用封存数据，覆盖截至 2026 年 11 月 11 日的研究样本。手动模拟需要用已核验的当年名单和最新完整月末数据生成信号。

历史归档的规模数据日期不同步，候选覆盖及退市覆盖有局限。采用标准发生在查看历史结果之后，未来独立表现和实际执行仍待积累。政策标签为 `v2.3.0`。较早的年度版见 [V2.2 标签](https://github.com/secyunshu/c10-etf-rotation-research/tree/v2.2.0)，固定池版本见 [V2.1 标签](https://github.com/secyunshu/c10-etf-rotation-research/tree/v2.1.0)。第三方数据权利见 [NOTICE](NOTICE.md)。
