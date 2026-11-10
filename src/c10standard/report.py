"""Describe frozen V2 account books offline; no simulation or parameter selection."""

from datetime import date, timedelta
from math import fsum, isclose, prod, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

import exchange_calendars as xcals
from beartype import beartype
from pydantic import Field, TypeAdapter

from c10share.report import read_csv, write_csv
from rotbt.models import DailyRecord, Distribution, Metrics, Model, OrderRecord

from .audit import Description
from .io import ROOT, load_accounts, load_strategy, reference_folder
from .models import COHORTS, LAST, Cohort, Contract, NativeScalar, Rule


class Nav(Model):
    day: date
    equity: float = Field(gt=0, allow_inf_nan=False)


class Book(Model):
    contract: Contract
    initial_cash: float = Field(gt=0, allow_inf_nan=False)
    daily: tuple[Nav, ...]
    orders: tuple[OrderRecord, ...] = ()
    dividends: tuple[Distribution, ...] = ()
    archived_metrics: tuple[Metrics, ...] = ()
    archived_ytd: Description | None = None


class Statistics(Model):
    account: str
    rule: Rule
    cost_bps: int
    cohort: Cohort
    period: str
    start: date
    end: date
    days: int
    opening_equity: float
    closing_equity: float
    profit: float
    total_return: float
    max_drawdown: float
    cagr_252: float | None
    annualized_volatility: float
    sharpe_rf0: float
    fees: float
    orders: int
    dividends: float
    turnover: float


class Verification(Model):
    status: Literal["PASS"] = "PASS"
    accounts: int
    source_hashes_checked: int
    archived_metric_values_checked: int
    annual_rows: int
    summary_rows: int
    monthly_rows: int
    native_attempts_by_report: Literal[0] = 0
    source: str = "standard/ACCOUNTS.json"
    scope: str = "continuous summaries are 2026 YTD; annual/monthly tables retain all account dates"


@beartype
def summarize(book: Book, first: date, last: date, period: str) -> Statistics:
    """Ported NAV statistics: period-opening capital participates in returns and drawdown."""
    days = [row.day for row in book.daily]
    if not days or days != sorted(set(days)):
        raise ValueError("NAV must be nonempty and strictly increasing")
    if days[-1] > LAST:
        raise ValueError("NAV exceeds the frozen publication cutoff")
    indices = [i for i, row in enumerate(book.daily) if first <= row.day <= last]
    if not indices:
        raise ValueError("empty native NAV period")
    selected = book.daily[indices[0] : indices[-1] + 1]
    opening = book.daily[indices[0] - 1].equity if indices[0] else book.initial_cash
    previous = [opening] + [row.equity for row in selected[:-1]]
    changes = [row.equity / prior - 1 for row, prior in zip(selected, previous, strict=True)]
    sigma = stdev(changes) if len(changes) > 1 else 0.0
    peak, drawdown = opening, 0.0
    for row in selected:
        peak = max(peak, row.equity)
        drawdown = max(drawdown, 1 - row.equity / peak)
    orders = [row for row in book.orders if first <= row.day <= last]
    growth = selected[-1].equity / opening
    return Statistics(
        account=book.contract.key,
        rule=book.contract.rule,
        cost_bps=book.contract.cost_bps,
        cohort=book.contract.cohort,
        period=period,
        start=selected[0].day,
        end=selected[-1].day,
        days=len(selected),
        opening_equity=opening,
        closing_equity=selected[-1].equity,
        profit=selected[-1].equity - opening,
        total_return=growth - 1,
        max_drawdown=drawdown,
        cagr_252=growth ** (252 / len(selected)) - 1 if selected[-1].day.year < 2026 else None,
        annualized_volatility=sigma * sqrt(252),
        sharpe_rf0=mean(changes) / sigma * sqrt(252) if sigma else 0.0,
        fees=fsum(row.fee for row in orders),
        orders=len(orders),
        dividends=fsum(row.net_amount for row in book.dividends if first <= row.ex_date <= last),
        turnover=fsum(abs(row.size * row.price) for row in orders)
        / mean(row.equity for row in selected),
    )


@beartype
def annual_rows(book: Book) -> list[Statistics]:
    return [
        summarize(book, date(year, 1, 1), date(year, 12, 31), str(year))
        for year in sorted({row.day.year for row in book.daily})
    ]


@beartype
def monthly_rows(book: Book) -> list[Statistics]:
    output = []
    for month in sorted({row.day.strftime("%Y-%m") for row in book.daily}):
        days = [row.day for row in book.daily if row.day.strftime("%Y-%m") == month]
        row = summarize(book, days[0], days[-1], month)
        output.append(row.model_copy(update={"cagr_252": None}))
    return output


@beartype
def summary_row(book: Book) -> Statistics:
    first = book.contract.start if book.contract.terminal else date(2026, 1, 2)
    return summarize(
        book, first, book.contract.through, "2022_2025" if book.contract.terminal else "2026_ytd"
    )


@beartype
def load_books(root: Path) -> tuple[list[Book], int]:
    strategy = load_strategy(root)
    references = load_accounts(root)
    # Hash the complete set before any account is summarized or output is written.
    folders = [(account, reference_folder(account, root)) for account in references]
    books = []
    for account, folder in folders:
        daily = read_csv(folder / "daily.csv.gz", DailyRecord)
        expected = [
            row.date()
            for row in xcals.get_calendar("XNYS").sessions_in_range(
                account.contract.start, account.contract.through
            )
        ]
        if [row.day for row in daily] != expected:
            raise ValueError(f"wrong or empty frozen account calendar: {account.key}")
        native = NativeScalar.model_validate_json(
            (folder / "native_total_return.json").read_bytes()
        )
        if not isclose(
            daily[-1].equity / strategy.initial_cash - 1,
            native.native_total_return,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"saved native total return mismatch: {account.key}")
        if account.contract.terminal and daily[-1].positions != 0:
            raise ValueError("historical terminal account must be liquidated")
        book = Book(
            contract=account.contract,
            initial_cash=strategy.initial_cash,
            daily=tuple(Nav(day=row.day, equity=row.equity) for row in daily),
            orders=read_csv(folder / "orders.csv.gz", OrderRecord),
            dividends=read_csv(folder / "dividends.csv.gz", Distribution),
            archived_metrics=TypeAdapter(tuple[Metrics, ...]).validate_json(
                (folder / "metrics.json").read_bytes()
            )
            if account.contract.terminal
            else (),
            archived_ytd=None
            if account.contract.terminal
            else Description.model_validate_json((folder / "ytd.json").read_bytes()),
        )
        books.append(book)
    return books, sum(len(account.reference_files) for account in references)


def equal(actual: float, expected: float, label: str) -> None:
    if not isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"archived statistic mismatch: {label}")


@beartype
def verify_book(book: Book) -> int:
    checks = 0
    for archived in book.archived_metrics:
        first, last = (
            (2022, 2023)
            if archived.period == "early"
            else (
                (2024, 2025)
                if archived.period == "late"
                else (2022, 2025)
                if archived.period == "full"
                else (int(archived.period), int(archived.period))
            )
        )
        actual = summarize(book, date(first, 1, 1), date(last, 12, 31), archived.period)
        for observed, expected in (
            (actual.days, archived.days),
            (actual.total_return, archived.total_return),
            (actual.max_drawdown, archived.max_dd),
            (actual.sharpe_rf0, archived.sharpe),
            (actual.fees, archived.fees),
            (actual.orders, archived.orders),
            (actual.dividends, archived.dividends),
            (actual.turnover, archived.turnover),
        ):
            equal(observed, expected, f"{book.contract.key}/{archived.period}")
            checks += 1
        if actual.cagr_252 is None:
            raise ValueError("historical CAGR unexpectedly unavailable")
        equal(actual.cagr_252, archived.cagr, "historical CAGR")
        checks += 1
    if book.archived_ytd is not None:
        ytd = summary_row(book)
        for field in (
            "days",
            "opening_equity",
            "closing_equity",
            "total_return",
            "max_drawdown",
            "sharpe_rf0",
            "annualized_volatility",
            "fees",
            "orders",
            "dividends",
        ):
            equal(float(getattr(ytd, field)), float(getattr(book.archived_ytd, field)), field)
            checks += 1
        for row in monthly_rows(book):
            if row.start.year == 2026:
                equal(row.total_return, book.archived_ytd.monthly_returns[row.period], row.period)
                checks += 1
    growth = book.daily[-1].equity / book.initial_cash
    for rows in (annual_rows(book), monthly_rows(book)):
        equal(prod(1 + row.total_return for row in rows), growth, "compounded returns")
    return checks


def write_report(
    root: Path, summaries: list[Statistics], annual: list[Statistics], verification: Verification
) -> None:
    default_rule = load_strategy(root).default_rule
    default_k = {"C10": 1, "C11": 2, "C00": 3}[default_rule]
    default = next(
        row
        for row in summaries
        if row.rule == default_rule and row.cost_bps == 5 and row.cohort == "reset2026"
    )
    lines = [
        "# V2 标准历史与 2026 已见样本结果",
        "",
        f"当前默认是 **{default_rule} / K={default_k}**：历史 2022 前 AUM 条件池"
        "（34 风险资产 + SHY），21 个经济风险组，每组最多 1 个持仓。"
        "用户于2026-11-11明确将默认从C10前1只改为C11前2只；C10与C00仍为完整对照。"
        "这是看过历史结果后的规则选择，不是新的独立样本外确认。",
        "",
        f"**默认 {default_rule}，5 bp 单边滑点，2026 重置账户累计收益 {default.total_return:.2%}，"
        f"最大回撤 {default.max_drawdown:.2%}。** "
        "回撤仍然较大；成为研究标准不代表获准实盘。"
        "2026 已参与研究，不能再称全新 OOS。历史 AUM 证据仅支持 conditional PIT，"
        "不能抹去候选集构造、存活与反复研究偏差。",
        "",
        "口径：所有账户初始 $50,000，目标部署预算 95%（价格漂移可超过该比例），"
        "同一费用表，另计 5/20 bp 单边滑点，含分红、无税、现金不计息。"
        "历史账户 2025-12-31 开盘终清；2026 重置账户从 $50,000 起算；"
        "连续账户的 2026 收益以其自身 2025-12-31 收盘 NAV 为分母，之前未终清。"
        "因此连续账户 2025 年数据与历史终清账户并非相同结算口径。",
        "",
        "年度和月度收益包含首个交易日相对期初资本的变化；最大回撤也包含期初资本。"
        "2026 截至 2026-11-06，11 月是不完整月。2026 收益不年化，CAGR 留空；"
        "波动率与 Sharpe 仅按日收益标准化（252 日、样本标准差、无风险利率 0）。"
        "历史 CAGR 使用 252/交易日数；年度收益使用实际期末/期初比值。",
        "",
    ]
    titles = ("历史 2022–2025（期末终清）", "2026 重置账户", "连续账户的 2026 YTD")
    for cohort, title in zip(COHORTS, titles, strict=True):
        for cost in (5, 20):
            lines.extend(
                [
                    f"## {title} · {cost} bp 单边滑点",
                    "",
                    "| 策略 | 累计收益 | 最大回撤 | CAGR | 期初权益 | 期末权益 |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in summaries:
                if row.cohort == cohort and row.cost_bps == cost:
                    cagr = f"{row.cagr_252:.2%}" if row.cagr_252 is not None else "—"
                    lines.append(
                        f"| {row.rule} | {row.total_return:.2%} | {row.max_drawdown:.2%} | "
                        f"{cagr} | ${row.opening_equity:,.2f} | ${row.closing_equity:,.2f} |"
                    )
            lines.append("")
    lines.extend(
        [
            f"## 历史年度 {default_rule} / QQQ / SPY · 5 bp",
            "",
            f"| 年度 | {default_rule} 收益 / 回撤 | QQQ 收益 / 回撤 | SPY 收益 / 回撤 |",
            "|---|---:|---:|---:|",
        ]
    )
    for year in range(2022, 2026):
        rows = [
            next(
                row
                for row in annual
                if row.rule == rule
                and row.cost_bps == 5
                and row.cohort == "historical2022_2025"
                and row.period == str(year)
            )
            for rule in (default_rule, "QQQ_BH", "SPY_BH")
        ]
        lines.append(
            f"| {year} | "
            + " | ".join(f"{row.total_return:.2%} / {row.max_drawdown:.2%}" for row in rows)
            + " |"
        )
    lines.extend(
        [
            "",
            "![历史净值与回撤](../figures/historical.png)",
            "",
            "![2026 重置账户净值与回撤](../figures/reset2026.png)",
            "",
            "## 可复核文件",
            "",
            "- [summary.csv](summary.csv)：36 行；连续账户只汇总 2026 YTD。",
            "- [annual.csv](annual.csv) 和 [monthly.csv](monthly.csv)："
            "保存完整账本的年度/月度切片，按 cohort、成本与 account 明确区分。",
            "- [verification.json](verification.json)：全部来源哈希与归档指标核对。",
            f"- 已验证 {verification.accounts} 账户、{verification.source_hashes_checked} "
            f"来源文件、{verification.archived_metric_values_checked} 个归档统计值。",
            "- 来源：[账户清单](../ACCOUNTS.json)、`standard/reference/<account>/` 原生五账本；"
            "[标准配置](../STRATEGY.json)。",
            "- 离线重建：`python -m c10standard.report`；报告不运行 native，不请求 API。",
            "",
        ]
    )
    (root / "standard/results/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def plot_reports(root: Path, books: list[Book]) -> None:
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
            "figure.facecolor": "white",
            "savefig.dpi": 170,
        }
    )
    default_rule = load_strategy(root).default_rule
    default_k = {"C10": 1, "C11": 2, "C00": 3}[default_rule]
    rules = ("C10", "C11", "C00", "QQQ_BH", "SPY_BH")
    colors = ("#0072B2", "#56B4E9", "#009E73", "#D55E00", "#CC79A7")
    for cohort, filename, title in (
        ("historical2022_2025", "historical", "Historical 2022-2025 | terminal liquidation"),
        ("reset2026", "reset2026", "2026 reset account | through 2026-11-06 | seen sample"),
    ):
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [2, 1]},
        )
        for rule, color in zip(rules, colors, strict=True):
            book = next(
                row
                for row in books
                if row.contract.rule == rule
                and row.contract.cost_bps == 5
                and row.contract.cohort == cohort
            )
            dates = [book.daily[0].day - timedelta(days=1)] + [row.day for row in book.daily]
            nav = np.asarray([book.initial_cash] + [row.equity for row in book.daily])
            axes[0].plot(
                dates,
                nav / book.initial_cash * 100,
                color=color,
                label=rule,
                linewidth=2.3 if rule == default_rule else 1.5,
            )
            if rule in (default_rule, "QQQ_BH", "SPY_BH"):
                axes[1].plot(
                    dates,
                    nav / np.maximum.accumulate(nav) - 1,
                    color=color,
                    label=rule,
                    linewidth=1.5,
                )
        axes[0].set_title(
            f"V2 | {title}\n5 bp one-way slippage plus fees; "
            f"{default_rule} / K={default_k} default",
            pad=36,
        )
        axes[0].set_ylabel("NAV (initial capital = 100)")
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=5, frameon=False)
        axes[1].set_ylabel("Drawdown from prior peak")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=3, frameon=False)
        for axis in axes:
            axis.grid(alpha=0.18)
            axis.margins(x=0.01)
        figure.savefig(root / f"standard/figures/{filename}.png")
        plt.close(figure)


@beartype
def build_report(root: Path = ROOT) -> Verification:
    books, hashes = load_books(root)
    checks = sum(verify_book(book) for book in books)
    annual = [row for book in books for row in annual_rows(book)]
    monthly = [row for book in books for row in monthly_rows(book)]
    summaries = sorted(
        (summary_row(book) for book in books),
        key=lambda row: (COHORTS.index(row.cohort), row.cost_bps, -row.total_return),
    )
    verification = Verification(
        accounts=len(books),
        source_hashes_checked=hashes,
        archived_metric_values_checked=checks,
        annual_rows=len(annual),
        summary_rows=len(summaries),
        monthly_rows=len(monthly),
    )
    for folder in ("results", "figures"):
        (root / "standard" / folder).mkdir(exist_ok=True)
    write_csv(root / "standard/results/annual.csv", annual)
    write_csv(root / "standard/results/summary.csv", summaries)
    write_csv(root / "standard/results/monthly.csv", monthly)
    (root / "standard/results/verification.json").write_text(
        verification.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    write_report(root, summaries, annual, verification)
    plot_reports(root, books)
    return verification


if __name__ == "__main__":
    print(build_report().model_dump_json(indent=2))
