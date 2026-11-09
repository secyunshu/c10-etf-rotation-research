# 如何独立复现

本仓库有两种互补核验：从冻结净值独立复算指标；从冻结行情重新运行原生Backtrader，再对比参考账本。后者才检验公开策略入口是否产生相同历史路径。

## 安装

建议使用与原研究一致的Python3.14.3。项目语法要求Python≥3.12，但没有据此宣称所有Python／操作系统组合都已验证。

```bash
git clone https://github.com/secyunshu/c10-etf-rotation-research.git
cd c10-etf-rotation-research
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
export PYTHONPATH=src
```

也可使用uv：`uv venv --python 3.14.3 .venv`，然后`uv pip install --python .venv/bin/python -r requirements.lock`。所有命令从仓库根目录运行。当前支持macOS／Linux；Windows请使用WSL2 Linux环境。尝试日志依赖POSIX文件锁，不能把PowerShell原生运行说成已支持。

安装联网下载通用依赖，之后不需要Futu SDK、OpenD、API账号或联网取行情。`requirements.lock`记录运行与画图依赖；开发检查另装`requirements-dev.lock`。请先用锁定版本复现，再考虑升级依赖。

## 第一步：检查输入和参考

```bash
python -m c10share.verify
```

检查规范输入及参考文件的SHA256、允许的身份／日期，并复核保存账户的财务与指标。如果输入被改动、文件缺失或截止不符，程序应明确失败，不自动下载替代数据，也不补齐假数据。

## 第二步：从行情重跑

```bash
python -m c10share.run --output runs/reproduction
```

默认依次运行12个固定账户：C10、C11、C00、SPY、QQQ、SPY60／IEF40，各含5bp和20bp。没有参数优化，没有随机择优。新输出写入`runs/`，保存参考放在`data/reference/`；不要让新输出覆盖参考。

核对包括每日资产／现金、每笔成交、持仓、分红、交易、目标信号及收益指标。判断以解析后的日期、行数、文字与数值为准，gzip文件的压缩时间戳不同不应误报账户差异。实际比较方式和最大数值差会记录在执行输出中；出现失败时保留证据，不能因为最终收益接近就忽略订单或中间净值差异。

每次原生账户尝试记录开始、结束或失败。该公开版本为复现固定结果而设，并不是无限参数搜索入口。发布验证采用独立48次有限预算，不重置旧研究预算；第三方重跑得到的相同历史不构成新的独立统计样本。

## 第三步：独立复算和画图

```bash
python scripts/build_report.py
```

从保存的每日资产独立计算年度／累计收益、回撤、年化波动、Sharpe、月度收益及最大回撤区间，输出到`results/`与`figures/`，与参考指标核对。它不会重新交易；只跑这一命令不能代替第二步。

CSV中的`total_return`、`cagr`、`max_dd`、`annual_volatility`均为小数比例，例如`0.5328`代表53.28%，不是0.5328%。`opening_equity`、`end_equity`、`profit`、`fees`、`dividends`为美元；回撤为正的损失幅度，图中回撤线可画成负数。`orders`是成交记录条数，并不是月度调仓次数或完整往返交易数。

年度收益不使用年度CAGR；每年首日以前的资产也必须纳入收益和回撤。全期年化使用252／交易日数，完整期1003日。详见[指标定义](STRATEGY.md#指标)。

## 开发检查

```bash
python -m pip install -r requirements-dev.lock
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m mypy --strict src scripts
python -m ruff check src scripts tests
# pyright需另安装其CLI，发布使用1.1.408
pyright --pythonpath .venv/bin/python
```

本仓库只包含当前公开入口的针对性测试，原核心源保留原样。可选的GitHub Actions配置见[示例](github-actions-example.yml)，本次发布没有启用远端CI，验收记录中的检查均注明实际运行环境。显式类型检查、被篡改数据拒绝、日期边界和指标校验，不等于已测试真实券商所有异常。

## 如何定位不一致

1. 核对Python／依赖版本与输入哈希。更新过供应商历史或手改CSV都会改变数据身份。
2. 核对是在仓库根目录运行、`PYTHONPATH=src`，没有导入电脑上另一个`rotbt`包。
3. 看第一个失败账户、日期与字段，保留失败输出。先找最早差异，再看累计结果。
4. 不要使用复权价成交，不要改基准分红再投规则，不要把C10的执行壳top字段直接改为1，也不要把终期清仓改成收盘后。
5. 报告问题时只附软件版本、公开输入hash和错误输出；不需要券商账号、密码、密钥或账户截图。

若只是重新画图，字体／图像元数据可能随系统变化；这不应影响表格中的经济结果。任何未验证环境应如实标注，不用“本机可跑”替代独立复现。
