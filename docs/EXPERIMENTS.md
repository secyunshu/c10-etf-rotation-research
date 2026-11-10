# 新增实验与测试的完整公开

当前默认仍是V2原252日评分的C11前2只。13612W和六个独立变种作为研究对照，不会因执行复现而切换默认。原始报告中的结果、阴性证据、数据和账本均保留。

## 已公开的内容

|内容|位置|可执行范围|
|---|---|---|
|当前标准配置、回测、财务核验|`standard/`、`src/c10standard/`|默认C11，显式C10/C00或`--all36`复现旧对照|
|13612W18个固定账户|`experiments/calendar13612w/`|公开行情→原生Backtrader→已有五账本逐项比对|
|六个前2变种36个固定账户|`experiments/top2-momentum-variants/`|同上，固定清单、不提供任意参数搜索|
|便携实验执行与原始信号独立核验|`src/c10experiments/`|运行时只依赖本仓与锁定依赖|
|新增活跃测试|`tests/test_experiments*.py`、`test_default_policy.py`|默认pytest可收集；不运行原生账户|
|原研究9脚本＋8测试模块、76测试函数|`experiments/source_archive/`|原文归档供审计；依赖原研究机的集成测试不冒充可直接运行|

原文清单、SHA、函数名和依赖边界见[source archive INDEX](../experiments/source_archive/INDEX.json)。便携测试是按公开输入重新建立的可执行覆盖，不宣称与原76个测试函数逐个同名或原研究供应商证据链已重跑。

## 离线运行

先按[V2复现指南](REPRODUCE_V2.md)建立环境。建议Python3.14.3；安装依赖后不需要网络、Futu账号、API密钥或私有研究目录。

```bash
export PYTHONPATH=src

# 只验证54固定实验的清单和参考文件，不运行原生账户
python -m c10experiments.run

# 从保存净值重算两实验14张CSV和六次局部bootstrap，不运行原生账户
python -m c10experiments.report --output runs/experiment_report

# 显式重新运行54个固定账户并与已发布参考比较
python -m c10experiments.run --all54 --output runs/experiments_all --journal runs/experiment_release_attempts.jsonl

# 已成功对象仅复验，不新开账户
python -m c10experiments.run --all54 --verify-only --output runs/experiments_all
```

`--case`接受ACCOUNTS中的一个完整固定实验key，例如`E01_K2_5bp_historical2022_2025`；与`--all54`二选一。不可通过改窗口、权重或K产生未登记策略。每条新账户都有外层实验身份和来源绑定，内部C10/C11/C00只是执行壳；不能把某个E01账户误认作原252标准。

失败目录与原生ARTIFACT保留。无完整完成标记则拒绝自动重跑；已经完整生成的账户可恢复核验，不能默默消费第二次账户。新发布日志具有独立release身份和72次启动上限，所有失败均计数。旧2.0的96次日志不能挪用。本次验收固定3个默认C11账户＋54个实验＝57次；下载后的再次核验仅检查源／账本／报表，未追加57次重复账户。

## 运行测试

```bash
python -m pip install -r requirements-dev.lock
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p typeguard --typeguard-packages=c10standard,c10experiments -q
python -m mypy --strict src
pyright --pythonpath .venv/bin/python
python -m ruff check src tests
```

活跃测试覆盖默认C11与显式旧C10、月末端点、波动率／回归／过滤公式、风险组和固定槽、未来价格隔离、篡改参考、恢复标量、日志失败计数，以及历史／2026分母、配对和局部统计范围。原始研究集成测试归档不在默认`tests/`路径，实际运行与未运行的数量在发布验收中分别报告。

## 数据与结果不能混淆

- 标准行情为35资产，2025与2026物理分目录；历史最后日开盘清仓，2026重置／连续账户在11月6日收盘估值。
- 13612W采用真实1/3/6/12日历月末收益权重12:4:2:1；六变种严格按其原预登记。SHA和独立raw信号验证约束公式漂移。
- 复算旧表时保留2026累计收益，不能将其冒年化。局部bootstrap仅六变种／两个成本／5、20、60日块／10000次，不校正过去全部研究，也不把点区间误称CAGR区间。
- 来源、程序、报表可复现不等于策略可靠。历史池仍为条件重建，2026已见，真实冲击／购买力和未来表现未获证明。
