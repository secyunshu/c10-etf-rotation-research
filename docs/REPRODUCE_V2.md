# 当前V2.2与历史V2复现入口

当前政策以[CURRENT.json](../standard/CURRENT.json)及[当前规则](CURRENT_STANDARD.md)为准。仓库根目录、锁定依赖环境：

```bash
export PYTHONPATH=src:.
# 默认年度池K2，校验来源/配置并重算公开账本，native0
python -m scripts.current_standard
# 真正重现2固定桥＋2年度成本账户，共4native
python -m scripts.current_standard --execute --output runs/current-standard
# 重复同路径只复验，不再native
python -m scripts.current_standard --execute --output runs/current-standard
```

先两桥再年度账户是技术复现，不是重新选择策略。每次独立复现任务共享`standard/current/evidence/attempts.jsonl`，hard8包含失败；本次发布原日志归档另存，不能清空已有日志绕预算。完整账簿、目标、timings、原生日收益及指标须与已验年度参考一致。默认只支持2022–2025，无2026年度池，不自动延用最后一年名单。

政策release v2.2.0与冻结兼容运行包2.1.0是不同版本轴。下方命令明确用于**历史固定池V2.1**，不会被偷偷重定向；旧账簿、代码及源锁完整保留。

---

# V2离线复现

推荐Python3.14.3；依赖沿用仓库锁文件。安装后不需要富途账号、私有仓库、API密钥或联网行情。

```bash
git clone https://github.com/secyunshu/c10-etf-rotation-research.git
cd c10-etf-rotation-research
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
export PYTHONPATH=src

# 从参考账本独立生成V2年度/YTD表与图，不运行回测
python -m c10standard.report

# 以下默认单账户与全36账户二选一，不必重复运行默认账户。
# 默认标准：历史池＋分组，前2只C11，5bp，2022–2025
python -m c10standard.run --output runs/standard_default --journal runs/standard_attempts.jsonl

# 全36固定账户（3策略＋3基准，2档成本，3种时间口径）
python -m c10standard.run --all36 --output runs/standard_all --journal runs/standard_attempts.jsonl

# 已运行目录只核验，不新开账户
python -m c10standard.run --all36 --verify-only --output runs/standard_all
```

单个对照可加`--rule C10`或`C00`、`--cost-bps 20`、`--cohort reset2026`或`continuous2022_2026`。当前默认K2由标准JSON读取；没有参数扫描入口。

每个正式账户从公开行情产生真实Backtrader订单、持仓、分红、NAV及TimeReturn，并核对固定参考账本、目标、指标与真实原生收益。6个旧历史基准仅有原生总收益标量，未伪造其旧逐日原生向量；新运行的逐日向量仍独立核对NAV。所有2026参考已含真实逐日TimeReturn及期末Trade。

已有成功目录会复验；失败保留FAILURE.txt/原始ARTIFACT。没有完整原生完成标记时拒绝静默重跑，恢复不能覆盖失败证据。独立运行标准CLI的原日志上限96包含失败。旧2.0发布实际验证为36正式＋36隔离复制共72次，原收据保留。本次2.1发布使用新的独立release身份与共享72上限，固定3个默认C11账户＋54个既有实验复现＝57次；这不是新增策略搜索。请保持同一次验证的journal路径一致。

2025与2026行情物理分目录；历史账户只解析2025目录。文件manifest、固定策略契约与预先保存的参考版本共同检测漂移；hash不能证明供应商事实正确。原始第三方整页资料没有发布，规范输入、最小日期观察及来源哈希足够复现本包计算，但不替代独立市场数据认证。

Windows请使用WSL2，原生尝试日志用POSIX文件锁。本机绝对路径不会成为数据或模块依赖。旧版命令`python -m c10share.run`仍专门复现V1，其旧36池结果不属于V2。


## 旧版证据与新默认

旧`v2.0.0`标签和`standard/evidence/REPRODUCTION_LOCK.json`代表旧默认K1发布。变更涉及的旧源码、配置、测试及报告保存在`standard/evidence/v2_0_preserved/`，`V2_0_PATH_MAP.json`将旧相对hash路径映射至快照；其余原始输入／参考账本字节不变。当前版本的验收和源码锁单独命名，不能用旧72次通过记录证明本次默认或新实验适配已通过。

新增54实验的运行和全部测试见[实验指南](EXPERIMENTS.md)。它们是已完成研究的复现，不会自动变成标准默认。
