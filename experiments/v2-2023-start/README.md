# 原V2前两只：按2022年底可得规模重建池，2023开始

本实验保持原252交易日总回报动量、月末调仓、风险分组、SHY门槛、两槽42.75%及原费用。2022-12-31休市，首信号2022-12-30close，首次开仓2023-01-03open；新50k现金账户，2024/25连续运行，2025-12-31最后open清仓。

**基金池是截至2022年底可核验的最新公开归档条件重建，不是各基金同一天12月30日精确AUM排名。** 95候选及类别继承前期设计，PIT/退市覆盖偏差未完全消除。来源公开时点、页面日及经济日分开记录，SOXX的规模争议、近排名差及异步日期局限完整保留，不能把这份结果称严格无偏历史全市场研究。

[冻结计划](PLAN.md) · [数据验收](DATA_ACCEPTANCE.json) · [完整95候选](universe/CANDIDATES.json) · [36类别赢家](universe/CATEGORY_WINNERS.csv) · [最终报告](FINAL_REPORT.md)

原V2池34风险＋SHY；重建池33风险＋SHY。SMH/IEFA/VWO/BND/VCSH分别替代SOXX/VEA/IEMG/AGG/VCIT，INDA无达到规模门槛候选而退出。SHY仍仅防守。池在起点固定，三年中不重新选池。新旧池同2023新账户对照，加SPY/QQQ相同费用、95%初始投入及分红留现金，避免用旧2022资本切片冒同起点对照。

## 复现

使用主仓锁定Python环境，在仓库根目录运行；数据物理截止2025，已包含必要规范OHLCV、真实turnover、现金支付日、拆股及TR，不需要网络行情或API密钥。

```bash
# 默认从公开参考账本重算及核对，不运行native
PYTHONPATH=src:. python -m scripts.v2_2023_report

# 新测试显式运行；不改变旧tests目录或配置
PYTHONPATH=src:. python -m pytest -q experiments/v2-2023-start/tests --typeguard-packages=scripts

# 两桥原2022账户先验证，再跑8个2023新账户
PYTHONPATH=src:. python -m scripts.v2_2023_run --stage bridge --output runs/v2-2023-start
PYTHONPATH=src:. python -m scripts.v2_2023_run --stage science --output runs/v2-2023-start
# 已完成账户只复核，不新增native
PYTHONPATH=src:. python -m scripts.v2_2023_run --stage science --output runs/v2-2023-start --verify-only
PYTHONPATH=src:. python -m scripts.v2_2023_report --native-output runs/v2-2023-start
```

最后报告命令逐单元比对已发布表，不覆盖不同结果。成功账户不能换目录再跑；失败保留并阻断自动重试。每次独立复现任务journal在本实验evidence/attempts.jsonl，硬cap16含失败。本研究实际计数另归档RESEARCH_ATTEMPTS.jsonl，不得删除旧计数重置同任务预算。

源码只新增`scripts/v2_2023_*.py`；自有日期/池契约不绕旧固定schema。仅迁移Cerebro配置/信号入口，撮合/订单/现金/费用/Position/TimeReturn仍原Backtrader，分红拆股沿已验证适配；财务审计及年度统计复用原函数。原源码、数据、标准默认与manifest不变，新实验有独立SOURCE_LOCK和MANIFEST。

`reference/<合约>/`提供五账本、目标、逐日原生收益及指标和明确池身份。完整大ARTIFACT保留本地可按命令重建；发布的receipt中其hash是本地产物证据，不冒已分发。原始网页/全部供应商响应保留本地；公开最小来源观察及hash。规范VCSH三项修正均有用户批准，见universe内批准记录。

年度最大回撤每年从期初资本起算，完整回撤跨年延续；Sharpe为每日账户净收益、rf=0、按252日年化，与之前报告一致。没有2026新经济运行；实施阶段误扩旧回归测试曾内部读取已存2026夹具，未用于新池/新信号/收益，记录于IMPLEMENTATION_REVIEW。三年样本已经见过，不称新OOS。
