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
# 默认标准：历史池＋分组，前1只，5bp，2022–2025
python -m c10standard.run --output runs/standard_default --journal runs/standard_attempts.jsonl

# 全36固定账户（3策略＋3基准，2档成本，3种时间口径）
python -m c10standard.run --all36 --output runs/standard_all --journal runs/standard_attempts.jsonl

# 已运行目录只核验，不新开账户
python -m c10standard.run --all36 --verify-only --output runs/standard_all
```

单个变体可加`--rule C11`或`C00`、`--cost-bps 20`、`--cohort reset2026`或`continuous2022_2026`。默认K1未变，没有参数扫描入口。

每个正式账户从公开行情产生真实Backtrader订单、持仓、分红、NAV及TimeReturn，并核对固定参考账本、目标、指标与真实原生收益。6个旧历史基准仅有原生总收益标量，未伪造其旧逐日原生向量；新运行的逐日向量仍独立核对NAV。所有2026参考已含真实逐日TimeReturn及期末Trade。

已有成功目录会复验；失败保留FAILURE.txt/原始ARTIFACT。没有完整原生完成标记时拒绝静默重跑，恢复不能覆盖失败证据。共享journal每次开始均计数，上限96包含失败；公开发布验证计划为36正式＋36隔离复制共72次，不属于新的科学试验。请保持同一次验证的journal路径一致。

2025与2026行情物理分目录；历史账户只解析2025目录。文件manifest、固定策略契约与预先保存的参考版本共同检测漂移；hash不能证明供应商事实正确。原始第三方整页资料没有发布，规范输入、最小日期观察及来源哈希足够复现本包计算，但不替代独立市场数据认证。

Windows请使用WSL2，原生尝试日志用POSIX文件锁。本机绝对路径不会成为数据或模块依赖。旧版命令`python -m c10share.run`仍专门复现V1，其旧36池结果不属于V2。
