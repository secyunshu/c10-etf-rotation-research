# 原V2每年年底重建ETF池

一个连续50k账户，2022-01-03至2025-12-31；2022/23/24/25分别使用截至2021/22/23/24年底的历史规模资料选池。其余保持原V2：252交易日总回报、月末调仓、前两只/同组最多一只、SHY门槛、失败槽不补位、42.75%每风险槽、SHY至少9.5%、约5%现金。

年度换池不是每年重新开户。退出池的旧持仓下一年首次open正常卖出，分红应收继续核算；choices=()，不调用旧框架年度强制清仓。只有2025最后交易日open终止清仓。未来年份的基金可以留在行情并集用于完整账目，但不提前参与排名。

资料采用截至各年底可核验的最新有效归档reported AUM，原95候选/36类别、5B门槛/每类最大。**不等同全部基金同一12月31日精确规模快照**：归档异步、经济日期不明、候选后见设计及退市覆盖局限保留。新2023/24部分页面可早至约220/225天前，不据收益调整取样。完整[计划](PLAN.md)、[数据验收](DATA_ACCEPTANCE.json)、[每年池变化](universe/ANNUAL_CHANGES.json)、[最终报告](FINAL_REPORT.md)。

## 复现

主仓锁定环境，仓库根目录运行。不需要行情API密钥，规范数据截至2025，已有分红pay日及拆股。旧源码/标准及已有结果不改。

```bash
# 从公开账本重算表格与金融审计，native0
PYTHONPATH=src:. python -m scripts.annual_report
PYTHONPATH=src:. python -m pytest -q experiments/v2-annual-repool/tests --typeguard-packages=scripts

# 原固定池两桥先行，再年度池两档成本；共4native，硬cap8含失败
PYTHONPATH=src:. python -m scripts.annual_run --stage bridge --output runs/v2-annual-repool
PYTHONPATH=src:. python -m scripts.annual_run --stage science --output runs/v2-annual-repool
PYTHONPATH=src:. python -m scripts.annual_run --stage science --output runs/v2-annual-repool --verify-only
PYTHONPATH=src:. python -m scripts.annual_report --native-output runs/v2-annual-repool
```

最后报告命令与已发布表逐单元比较，不覆盖差异。成功账户仅复核，同key禁止换路径重跑；失败保留。本地attempts.jsonl为同次独立复现任务共享日志，不得清空来绕预算。本次研究原日志另归档RESEARCH_ATTEMPTS.jsonl。

timings.json分别记录price_asof、selection_asof、source_asof、execute_on和poolhash。年末价格bar在周五时，周末公开的资料只有到selection_asof才用于确定新年度目标；不能把targets的价格日期标签误当全部资料当日已可得。所有目标先于成交、不会使用次日收盘。

新模块scripts/annual_*.py只扩展日期/名单契约及信号装配；Backtrader和原HistoricalRotation/Phase2Rotation继续拥有订单/现金/持仓/费用/公司行动。统计和金融核验复用已验证函数。五账本、timings、原生日收益和指标公开；大ARTIFACT原生对象本地保留，可重建，receipt中的hash不冒已分发文件。

SPY/QQQ沿原95%初始投入、分红留现金同费账户，非网站100%复投曲线。Sharpe rf0/252年化；年度DD重置期初高点，完整DD跨年保留。历史已经见过，不称新OOS；本轮没有2026数据或新行情调用，也不自动把年度池改成默认策略。
