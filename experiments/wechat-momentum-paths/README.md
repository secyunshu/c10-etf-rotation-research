# 文章启发的三个路径动量实验

当前标准仍是V2前2只（C11）。本目录为独立研究，不改变默认配置。

- 原文：[【国信金工】动量类因子全解析](https://mp.weixin.qq.com/s/zTU2-FcGSXkeYOxLZlmUQw)。浏览器核读后，只公布来源元数据、公式观察及本研究，不转载全文/图片。
- [文章评估](ARTICLE_REVIEW.md)、[冻结计划](PLAN.md)、[计划挑战审查](PLAN_REVIEW.json)、[源码与输入锁](SOURCE_LOCK.json)。
- [最终结论](FINAL_REPORT.md)、[逐年数据](results/ANNUAL.csv)、[总收益及回撤](results/SUMMARY.csv)、[局部统计检验](results/LOCAL_BOOTSTRAP.csv)。

三项固定为P01日排名持续性Rank120_20、P02字面Smooth240、P03距离前250日高点（总回报适配）。均为历史34风险ETF＋SHY、21组cap1、固定前2槽、R252相对SHY防守、月末信号次交易日开盘；5/20bp成本及三账户时间口径。2026止于2026-11-06，已见样本，不是新OOS。

## 复现

在仓库根目录，按主仓[环境说明](../../README.md)安装锁定依赖并激活Python环境。数据已随`standard/data`发布，不需要API密钥或网络行情。源码新增仅`src/c10paths`，复用既有Backtrader引擎及独立财务审计。

先离线重算已存18实验＋24既存对照的报表（0次native）：

```bash
PYTHONPATH=src python -m c10paths.report --output runs/wechat-report
PYTHONPATH=src python -m c10paths.verify --output runs/wechat-causality.json
python -m pytest -q tests/test_paths_signals.py tests/test_paths_run.py tests/test_paths_report.py --typeguard-packages=c10paths
```

从原始规范输入实际执行Backtrader，先2桥再18科学：

```bash
PYTHONPATH=src python -m c10paths.run --stage bridge
PYTHONPATH=src python -m c10paths.run --stage science
PYTHONPATH=src python -m c10paths.run
PYTHONPATH=src python -m c10paths.report --native-output runs/wechat-paths
```

最后一条报告命令会将重跑账户生成的六张表与本仓库已发布表逐单元比较，包括目标表；不会覆盖不同结果。未传`--stage`只验证已生成账户，不启动新回测。成功账户在同目录复核，禁止换目录重跑同trial；失败保留且阻断自动重跑。每个独立复现任务有自己的本地`evidence/attempts.jsonl`，硬上限28；本研究原始计数单独归档为`evidence/RESEARCH_ATTEMPTS.jsonl`，不可将原研究失败/计数删除重置。本轮协调工作始终共用同一计数。

`reference/<variant>/<C11合约名>/`保留五账本、实际目标、原生总收益、年度/YTD指标和外层变种身份。共同C11仅是执行合约壳，实际信号由BINDING.json明确绑定，不能靠目录重命名改变身份。完整大ARTIFACT原生序列化留本地；它可通过以上命令重建，不把未分发大对象称已上传。

所有源码/输入改变都会使SOURCE_LOCK校验失败；新假说应另登记。已有公开仓原MANIFEST和旧证据未覆盖本新增实验，本目录另有自己的MANIFEST.sha256。

## 读表口径

`SUMMARY.csv`中历史行是2022–2025全期；两个2026行分别重置50k和连续账户的2026YTD。`ANNUAL.csv`保留连续账户的各年，不与历史清仓账户拼接。CAGR仅历史列有值；DD以该报告期间开端的净值作为起点。费用/换手来自原生成交，不是无摩擦信号收益。

`LOCAL_BOOTSTRAP.csv`完整保留三候选、两成本、三block的18行。均值和点态区间为每日收益基点，检验的是相对C11的均值差；不代表CAGR或降回撤显著性，不覆盖整个历史研究搜索。条件历史池、样本已见、真实执行未验证等限制与原标准一致。
