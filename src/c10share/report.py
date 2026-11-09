"""Independently summarize saved native-account NAV; never execute a strategy."""

import csv
import gzip
import json
from datetime import date, timedelta
from math import fsum, isclose, isfinite, prod, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

from rotbt.models import (
    DailyRecord,
    Distribution,
    Metrics,
    Model,
    OrderRecord,
)

from .io import load_accounts, load_task


class NavPoint(Model):
    day: date
    equity: float = Field(gt=0, allow_inf_nan=False)


class Account(Model):
    name: str
    rule: str
    cost_bps: int
    initial_cash: float = Field(gt=0, allow_inf_nan=False)
    split_year: int
    daily: tuple[NavPoint, ...]
    orders: tuple[OrderRecord, ...] = ()
    dividends: tuple[Distribution, ...] = ()
    reference: tuple[Metrics, ...] = ()

    def validate_history(self) -> None:
        days = [row.day for row in self.daily]
        if not days or days != sorted(set(days)):
            raise ValueError("NAV dates must be nonempty and strictly increasing")
        if days[-1] > date(2025, 12, 31):
            raise ValueError("report must not read beyond 2025")
        if any(not isfinite(row.equity) or row.equity <= 0 for row in self.daily):
            raise ValueError("NAV equity must be finite and positive")


class PeriodStats(Model):
    account: str
    rule: str
    cost_bps: int
    period: str
    year: int | None
    start: date
    end: date
    days: int
    opening_equity: float
    end_equity: float
    profit: float
    total_return: float
    cagr: float
    max_dd: float
    sharpe: float
    annual_volatility: float
    orders: int
    fees: float
    dividends: float
    turnover: float
    positive_years: int


class MonthlyStats(Model):
    account: str
    rule: str
    cost_bps: int
    month: str
    start: date
    end: date
    opening_equity: float
    end_equity: float
    total_return: float


class Summary(PeriodStats):
    end_nav: float
    calmar: float | None
    worst_month: float
    best_month: float
    positive_months: int
    months: int


class DrawdownDetail(Model):
    account: str
    cost_bps: int
    max_dd: float
    peak_date: date | None
    peak_equity: float
    trough_date: date
    trough_equity: float
    recovery_date: date | None
    trading_days_to_trough: int
    trading_days_to_recovery: int | None


class Verification(Model):
    status: Literal["PASS"] = "PASS"
    accounts: int
    source_hashes_checked: int
    periods_checked: int
    metric_values_checked: int
    annual_rows: int
    monthly_rows: int
    tolerance: float = 1e-10
    source_manifest: str = "data/ACCOUNTS.json"
    notes: tuple[str, ...] = (
        "Daily NAV and account books are saved native Backtrader output.",
        "All metrics independently recalculated without calling the native analyzer.",
        "Year and month returns compound to each full-account total return.",
        "Calendar-year return includes the first day's movement from preceding capital.",
        "CAGR uses 252 trading days; calendar-year return is not annualized.",
        "Maximum drawdown includes period-opening capital; positive magnitude is loss.",
    )


def period_stats(account: Account, first: int, last: int, period: str) -> PeriodStats:
    account.validate_history()
    if not 0 <= first <= last < len(account.daily):
        raise ValueError("invalid period indices")
    rows = account.daily[first : last + 1]
    opening = account.daily[first - 1].equity if first else account.initial_cash
    previous = [opening] + [row.equity for row in rows[:-1]]
    changes = [row.equity / prior - 1 for row, prior in zip(rows, previous, strict=True)]
    volatility = stdev(changes) if len(changes) > 1 else 0.0
    peak, drawdown = opening, 0.0
    for row in rows:
        peak = max(peak, row.equity)
        drawdown = max(drawdown, 1 - row.equity / peak)
    orders = [row for row in account.orders if rows[0].day <= row.day <= rows[-1].day]
    dividends = [row for row in account.dividends if rows[0].day <= row.ex_date <= rows[-1].day]
    positive_years = 0
    for year in sorted({row.day.year for row in rows}):
        indices = [i for i, row in enumerate(rows) if row.day.year == year]
        positive_years += rows[indices[-1]].equity > previous[indices[0]]
    growth = rows[-1].equity / opening
    return PeriodStats(
        account=account.name,
        rule=account.rule,
        cost_bps=account.cost_bps,
        period=period,
        year=int(period) if period.isdigit() else None,
        start=rows[0].day,
        end=rows[-1].day,
        days=len(rows),
        opening_equity=opening,
        end_equity=rows[-1].equity,
        profit=rows[-1].equity - opening,
        total_return=growth - 1,
        cagr=growth ** (252 / len(rows)) - 1,
        max_dd=drawdown,
        sharpe=sqrt(252) * mean(changes) / volatility if volatility else 0.0,
        annual_volatility=volatility * sqrt(252),
        orders=len(orders),
        fees=fsum(row.fee for row in orders),
        dividends=fsum(row.net_amount for row in dividends),
        turnover=fsum(abs(row.size * row.price) for row in orders)
        / mean(row.equity for row in rows),
        positive_years=positive_years,
    )


def annual_stats(account: Account) -> list[PeriodStats]:
    output = []
    for year in sorted({row.day.year for row in account.daily}):
        indices = [i for i, row in enumerate(account.daily) if row.day.year == year]
        output.append(period_stats(account, indices[0], indices[-1], str(year)))
    return output


def monthly_stats(account: Account) -> list[MonthlyStats]:
    account.validate_history()
    output = []
    for month in sorted({row.day.strftime("%Y-%m") for row in account.daily}):
        indices = [i for i, row in enumerate(account.daily) if row.day.strftime("%Y-%m") == month]
        first, last = indices[0], indices[-1]
        opening = account.daily[first - 1].equity if first else account.initial_cash
        output.append(
            MonthlyStats(
                account=account.name,
                rule=account.rule,
                cost_bps=account.cost_bps,
                month=month,
                start=account.daily[first].day,
                end=account.daily[last].day,
                opening_equity=opening,
                end_equity=account.daily[last].equity,
                total_return=account.daily[last].equity / opening - 1,
            )
        )
    return output


def drawdown_detail(account: Account) -> DrawdownDetail:
    account.validate_history()
    peak, peak_index, maximum = account.initial_cash, -1, 0.0
    saved_peak, trough = -1, 0
    for i, row in enumerate(account.daily):
        if row.equity > peak:
            peak, peak_index = row.equity, i
        loss = 1 - row.equity / peak
        if loss > maximum + 1e-15:
            maximum, saved_peak, trough = loss, peak_index, i
    peak_equity = account.daily[saved_peak].equity if saved_peak >= 0 else account.initial_cash
    recovery = next(
        (
            i
            for i in range(trough + 1, len(account.daily))
            if account.daily[i].equity >= peak_equity
        ),
        None,
    )
    return DrawdownDetail(
        account=account.name,
        cost_bps=account.cost_bps,
        max_dd=maximum,
        peak_date=account.daily[saved_peak].day if saved_peak >= 0 else None,
        peak_equity=peak_equity,
        trough_date=account.daily[trough].day,
        trough_equity=account.daily[trough].equity,
        recovery_date=account.daily[recovery].day if recovery is not None else None,
        trading_days_to_trough=trough - saved_peak,
        trading_days_to_recovery=recovery - saved_peak if recovery is not None else None,
    )


def read_csv[T: BaseModel](path: Path, model: type[T]) -> tuple[T, ...]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return tuple(model.model_validate(row) for row in csv.DictReader(handle))


def read_accounts(root: Path) -> tuple[list[Account], int]:
    references = load_accounts(root)
    accounts, hashes = [], 0
    for ref in sorted(references, key=lambda row: (row.cost_bps, row.name)):
        folder = root / "data/reference" / ref.name
        task = load_task(ref, root)
        hashes += len(ref.reference_files)
        account = Account(
            name=ref.name,
            rule=ref.rule,
            cost_bps=ref.cost_bps,
            initial_cash=task.settings.initial_cash,
            split_year=task.settings.effective_research_period.split_year,
            daily=tuple(
                NavPoint(day=row.day, equity=row.equity)
                for row in read_csv(folder / "daily.csv.gz", DailyRecord)
            ),
            orders=read_csv(folder / "orders.csv.gz", OrderRecord),
            dividends=read_csv(folder / "dividends.csv.gz", Distribution),
            reference=TypeAdapter(tuple[Metrics, ...]).validate_json(
                (folder / "metrics.json").read_bytes()
            ),
        )
        account.validate_history()
        if account.daily[0].day != task.start or account.daily[-1].day != task.end:
            raise ValueError(f"task period mismatch: {ref.name}")
        accounts.append(account)
    if any(
        [row.day for row in account.daily] != [row.day for row in accounts[0].daily]
        for account in accounts
    ):
        raise ValueError("comparison accounts must use the same trading dates")
    return accounts, hashes


def verify_account(account: Account) -> int:
    expected_periods = {"early", "late", "full"} | {str(row.day.year) for row in account.daily}
    if {row.period for row in account.reference} != expected_periods:
        raise ValueError("missing archived metric periods")
    checked = 0
    for archived in account.reference:
        indices = [
            i
            for i, row in enumerate(account.daily)
            if archived.period == "full"
            or (archived.period == "early" and row.day.year <= account.split_year)
            or (archived.period == "late" and row.day.year > account.split_year)
            or str(row.day.year) == archived.period
        ]
        fresh = period_stats(account, indices[0], indices[-1], archived.period)
        for key in Metrics.model_fields:
            if key == "period":
                continue
            actual, expected = float(getattr(fresh, key)), float(getattr(archived, key))
            if not isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"metric mismatch: {account.name}/{archived.period}/{key}")
            checked += 1
    full_growth = account.daily[-1].equity / account.initial_cash
    for rows in (annual_stats(account), monthly_stats(account)):
        if not isclose(prod(1 + row.total_return for row in rows), full_growth, rel_tol=1e-12):
            raise ValueError("period returns do not compound to full account")
    return checked


def summarize(account: Account) -> Summary:
    full = period_stats(account, 0, len(account.daily) - 1, "full")
    months = monthly_stats(account)
    return Summary(
        account=full.account,
        rule=full.rule,
        cost_bps=full.cost_bps,
        period=full.period,
        year=full.year,
        start=full.start,
        end=full.end,
        days=full.days,
        opening_equity=full.opening_equity,
        end_equity=full.end_equity,
        profit=full.profit,
        total_return=full.total_return,
        cagr=full.cagr,
        max_dd=full.max_dd,
        sharpe=full.sharpe,
        annual_volatility=full.annual_volatility,
        orders=full.orders,
        fees=full.fees,
        dividends=full.dividends,
        turnover=full.turnover,
        positive_years=full.positive_years,
        end_nav=full.end_equity / account.initial_cash * 100,
        calmar=full.cagr / full.max_dd if full.max_dd else None,
        worst_month=min(row.total_return for row in months),
        best_month=max(row.total_return for row in months),
        positive_months=sum(row.total_return > 0 for row in months),
        months=len(months),
    )


def write_csv[T: BaseModel](path: Path, rows: list[T]) -> None:
    if not rows:
        raise ValueError("cannot write an empty report table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(type(rows[0]).model_fields), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)


def write_markdown(
    root: Path,
    summaries: list[Summary],
    annual: list[PeriodStats],
    details: list[DrawdownDetail],
    verification: Verification,
) -> None:
    lines = [
        "# C10 历史账户结果（2022–2025）",
        "",
        "**这是条件性历史研究，不是可部署策略或未使用的样本外验证。** "
        "C10 是已比较的规则族中的候选；"
        "资产池使用当前规模筛选，存在当前 AUM 与存活偏差。原生 Backtrader 账户结果仅说明这些数据、"
        "规则和成本假设下的历史表现，不保证未来收益。",
        "",
        f"所有账户使用相同 2022-01-03 至 2025-12-31 窗口、"
        f"初始资金 ${summaries[0].opening_equity:,.0f}、95% 目标部署预算。"
        "价格变化导致持仓权重漂移，实际投资比例可超过 95%。"
        "收益含现金分红并扣交易费用与滑点；不含税、未使用现金利息。5 bp 为默认单边滑点，"
        "20 bp 为压力情景，均另计费用。QQQ/SPY 为同账户假设下的买入持有，非指数公布收益。",
        "",
        "年度收益 = 年末权益 / 上年末权益 − 1；首年分母为任务配置初始资金。年度最大回撤"
        "从年度期初资金开始测量，跨年历史回撤见下表；均以正数表示损失幅度。"
        "Sharpe 采用日收益、无风险利率 0、样本标准差；波动率乘 √252；"
        "全期 CAGR 按 252/交易日数折算（1003 天），不能代替实际年度收益。",
        "",
        "## 全期比较：默认 5 bp",
        "",
    ]
    for cost in (5, 20):
        if cost == 20:
            lines.extend(["", "## 全期比较：20 bp 单边滑点压力情景", ""])
        lines.extend(
            [
                "| 策略 | 累计收益 | CAGR | 最大回撤 | 年化波动 | Sharpe | 期末权益 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in summaries:
            if summary.cost_bps == cost:
                lines.append(
                    f"| {summary.rule} | {summary.total_return:.2%} | {summary.cagr:.2%} | "
                    f"{summary.max_dd:.2%} | {summary.annual_volatility:.2%} | "
                    f"{summary.sharpe:.3f} | ${summary.end_equity:,.2f} |"
                )
    lines.extend(
        [
            "",
            "## 各年 C10 / QQQ / SPY 对照：默认 5 bp",
            "",
            "| 年度 | C10 收益 | C10 回撤 | QQQ 收益 | QQQ 回撤 | SPY 收益 | SPY 回撤 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for year in sorted({row.year for row in annual if row.year is not None}):
        values = [
            next(
                row for row in annual if row.year == year and row.rule == rule and row.cost_bps == 5
            )
            for rule in ("C10", "QQQ_BH", "SPY_BH")
        ]
        lines.append(
            f"| {year} | "
            + " | ".join(f"{row.total_return:.2%} | {row.max_dd:.2%}" for row in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## C10 年度账户明细：默认 5 bp",
            "",
            "| 年度 | 期初权益 | 期末权益 | 净利润 | 收益 | 最大回撤 | "
            "Sharpe | 波动率 | 订单 | 费用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in annual:
        if row.rule == "C10" and row.cost_bps == 5:
            lines.append(
                f"| {row.year} | ${row.opening_equity:,.2f} | ${row.end_equity:,.2f} | "
                f"${row.profit:,.2f} | {row.total_return:.2%} | {row.max_dd:.2%} | "
                f"{row.sharpe:.3f} | {row.annual_volatility:.2%} | "
                f"{row.orders} | ${row.fees:.2f} |"
            )
    lines.extend(
        [
            "",
            "## C10 最大历史回撤",
            "",
            "| 单边滑点 | 回撤 | 峰值日期 | 谷值日期 | 恢复峰值日期 | 峰值至恢复交易日 |",
            "|---|---:|---|---|---|---:|",
        ]
    )
    for detail in details:
        duration = detail.trading_days_to_recovery
        lines.append(
            f"| {detail.cost_bps} bp | {detail.max_dd:.2%} | "
            f"{detail.peak_date or '初始资金'} | {detail.trough_date} | "
            f"{detail.recovery_date or '期末未恢复'} | "
            f"{duration if duration is not None else '未恢复'} |"
        )
    lines.extend(
        [
            "",
            "## 图表",
            "",
            "![权益与回撤](../figures/performance.png)",
            "",
            "![年度收益与回撤](../figures/annual_comparison.png)",
            "",
            "![C10 月度收益](../figures/monthly_heatmap.png)",
            "",
            "## 复核与数据口径",
            "",
            f"已校验 {verification.source_hashes_checked} 个来源文件 SHA-256；"
            f"独立复算 {verification.accounts} 个账户的 "
            f"{verification.periods_checked} 个时期、"
            f"{verification.metric_values_checked} 个指标值，"
            "与归档 metrics.json 在 1e-10 容差内一致。"
            "另检验年度和月度收益连乘等于全期收益，所有日期不晚于 2025-12-31。",
            "",
            "- [年度明细](annual.csv)：48 行，6 个账户 × 2 种滑点 × 4 年。",
            "- [全期汇总](summary.csv)：12 行，每种滑点内按累计收益降序，"
            "含 Calmar、月度极值和正收益月数。",
            "- [月度明细](monthly.csv)：576 行，含每月期初与期末权益。",
            "- [历史回撤路径](drawdowns.csv) 与 [验证结果](verification.json)。",
            "- 来源：[账户清单](../data/ACCOUNTS.json)、"
            "各账户 `data/reference/<account>/daily.csv.gz`、"
            "`orders.csv.gz`、`dividends.csv.gz`、`task.json`、`metrics.json`。",
            "- 重建：`python scripts/build_report.py`。只读已保存账户，不重新运行策略或联网取数。",
            "",
        ]
    )
    (root / "results/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def plot_reports(
    root: Path, accounts: list[Account], annual: list[PeriodStats], monthly: list[MonthlyStats]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
            "figure.facecolor": "white",
        }
    )
    colors = {
        "C10": "#0072B2",
        "C11": "#56B4E9",
        "C00": "#009E73",
        "QQQ_BH": "#D55E00",
        "SPY_BH": "#CC79A7",
    }
    labels = {
        "C10": "C10 (top 1)",
        "C11": "C11 (top 2)",
        "C00": "C00 (top 3)",
        "QQQ_BH": "QQQ buy & hold",
        "SPY_BH": "SPY buy & hold",
    }
    primary = ("C10", "QQQ_BH", "SPY_BH")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for account in accounts:
        if account.cost_bps != 5 or account.rule not in colors:
            continue
        dates = [account.daily[0].day - timedelta(days=1)] + [row.day for row in account.daily]
        nav = np.asarray([account.initial_cash] + [row.equity for row in account.daily])
        axes[0].plot(
            dates,
            nav / account.initial_cash * 100,
            color=colors[account.rule],
            label=labels[account.rule],
            linewidth=2.2 if account.rule == "C10" else 1.6,
        )
        if account.rule in primary:
            axes[1].plot(
                dates,
                nav / np.maximum.accumulate(nav) - 1,
                color=colors[account.rule],
                label=labels[account.rule],
                linewidth=1.5,
            )
    axes[0].set_title("Historical account NAV and drawdown | 2022-2025 | 5 bp one-way slippage")
    axes[0].set_ylabel("NAV (initial capital = 100)")
    axes[0].legend(loc="upper left", ncols=2, frameon=False)
    axes[1].set_ylabel("Drawdown from prior peak")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlim(date(2022, 1, 1), date(2025, 12, 31))
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.savefig(root / "figures/performance.png")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    years = sorted({row.year for row in annual if row.year is not None})
    positions = np.arange(len(years))
    for j, rule in enumerate(primary):
        rows = [
            next(
                row for row in annual if row.rule == rule and row.cost_bps == 5 and row.year == year
            )
            for year in years
        ]
        for axis, values in zip(
            axes, ([row.total_return for row in rows], [row.max_dd for row in rows]), strict=True
        ):
            bars = axis.bar(
                positions + (j - 1) * 0.25, values, 0.24, label=labels[rule], color=colors[rule]
            )
            axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], fontsize=9, padding=3)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(axis="y", alpha=0.18)
            axis.set_axisbelow(True)
    axes[0].set_title("Calendar-year return and maximum drawdown | 5 bp one-way slippage")
    axes[0].set_ylabel("Net annual return")
    axes[0].legend(loc="upper left", ncols=3, frameon=False)
    axes[0].margins(y=0.2)
    axes[0].axhline(0, color="#555555", linewidth=0.7)
    axes[1].set_ylabel("Annual maximum drawdown\n(loss magnitude)")
    axes[1].margins(y=0.2)
    axes[1].set_xticks(positions, [str(year) for year in years])
    figure.savefig(root / "figures/annual_comparison.png")
    plt.close(figure)

    grid = np.asarray(
        [
            [
                next(
                    row.total_return
                    for row in monthly
                    if row.rule == "C10"
                    and row.cost_bps == 5
                    and row.month == f"{year}-{month:02d}"
                )
                for month in range(1, 13)
            ]
            for year in years
        ]
    )
    figure, axis = plt.subplots(figsize=(12, 4.2), layout="constrained")
    limit = float(np.max(np.abs(grid)))
    heat = axis.imshow(grid, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(years)):
        for j in range(12):
            axis.text(
                j,
                i,
                f"{grid[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if abs(grid[i, j]) > 0.75 * limit else "#111111",
                fontsize=10,
            )
    axis.set_yticks(range(len(years)), [str(year) for year in years])
    axis.set_xticks(
        range(12),
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    axis.set_title("C10 net monthly returns | 5 bp one-way slippage")
    figure.colorbar(heat, ax=axis, format=PercentFormatter(1), label="Monthly return")
    figure.savefig(root / "figures/monthly_heatmap.png")
    plt.close(figure)


def build_report(root: Path) -> Verification:
    accounts, hashes = read_accounts(root)
    checked = sum(verify_account(account) for account in accounts)
    annual = [row for account in accounts for row in annual_stats(account)]
    monthly = [row for account in accounts for row in monthly_stats(account)]
    summaries = sorted(
        (summarize(account) for account in accounts),
        key=lambda row: (row.cost_bps, -row.total_return),
    )
    details = [drawdown_detail(account) for account in accounts if account.rule == "C10"]
    verification = Verification(
        accounts=len(accounts),
        source_hashes_checked=hashes,
        periods_checked=sum(len(account.reference) for account in accounts),
        metric_values_checked=checked,
        annual_rows=len(annual),
        monthly_rows=len(monthly),
    )
    for name in ("results", "figures"):
        (root / name).mkdir(exist_ok=True)
    write_csv(root / "results/annual.csv", annual)
    write_csv(root / "results/monthly.csv", monthly)
    write_csv(root / "results/summary.csv", summaries)
    write_csv(root / "results/drawdowns.csv", details)
    (root / "results/verification.json").write_text(
        json.dumps(verification.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(root, summaries, annual, details, verification)
    plot_reports(root, accounts, annual, monthly)
    return verification
