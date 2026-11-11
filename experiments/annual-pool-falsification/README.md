# 年度换池优势的有限证伪

本轮固定A原池、B仅从2023起SOXX换SMH、C年度更新但半导体固定SOXX、D完整年度更新。E是2023更新一次后固定的连续账户一致性对照，不作为额外选优候选。B/C是反事实机制测试，不声称满足完整年度AUM准入政策。

[预登记计划](PLAN.md) · [来源复核](SOURCE_REVIEW.md) · [最终报告](FINAL_REPORT.md) · [年度表](results/ANNUAL.csv) · [全期表](results/SUMMARY.csv) · [财富配对与交互](results/WEALTH_CONTRASTS.csv) · [路径一致性](results/PATH_EQUALITY.csv)

源数据/执行与原V2完全复用；所有账户是2022-01-03新50k至2025-12-31末open清仓，月末决定次open执行、K2/252/组cap/SHY门槛/原费用不变。原始数据和旧默认不修改。本轮不读取2026市场数据，不增加权重/窗口搜索。

## 离线复现

在主仓锁定Python环境中，从仓库根目录执行。使用已有annual研究的2025止数据，不需API。

```bash
# 公开参考账本重算，native0
PYTHONPATH=src:. python -m scripts.falsify_report
PYTHONPATH=src:. python -m pytest -q experiments/annual-pool-falsification/tests --typeguard-packages=scripts

# 两桥A5/D20与6科学B/C/E各两成本
PYTHONPATH=src:. python -m scripts.falsify_run --stage bridge --output runs/pool-falsify
PYTHONPATH=src:. python -m scripts.falsify_run --stage science --output runs/pool-falsify
PYTHONPATH=src:. python -m scripts.falsify_run --stage science --output runs/pool-falsify --verify-only
PYTHONPATH=src:. python -m scripts.falsify_report --native-output runs/pool-falsify
```

每次独立复现任务固定共享journal，硬cap12含失败；本次研究原日志另归档RESEARCH_ATTEMPTS.jsonl。成功只复核、同账户不能换路径重跑，失败证据保留。最后命令核对已发布表格，不覆盖不同数值。

CASES.json锁定五政策四期实际名单、分组及逐资产来源截止。CASE_BINDING记录真实政策，旧annual Contract仅执行容器；不能把目录改名或相同inner key当同策略。真实timings和raworacle都使用实际政策池。复用annual_native和annual_run的验证，不另造金融引擎。

REFERENCE中五账本、目标、timing、原生日收益、指标、外层身份均公开；完整大ARTIFACT本地留存、可重建。A/D旧结果及SPY/QQQ直接引用同窗参考，桥接不重复作为独立科学样本。

PATH_EQUALITY的五账比对只去掉行情并集带来的零持仓行；订单、现金、分红、非零股数、成交和收益都必须一致。最终财富差及D−B−C＋A是同一初始资本下的可加差额；CAGR/Sharpe/DD只是各政策统计，不解释为可加贡献。

历史规模证据严格区分当年公开档案、经济日期相同但后发的官方资料与未找到的同日PIT证明。SOURCE_REVIEW不覆盖旧归档值，B/C只诊断数据口径/代表依赖，不择优修池。历史已见，不声称新OOS或普遍有效/无效。
