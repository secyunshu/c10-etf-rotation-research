"""Thin read-only reporting over audited 13612W and frozen public V2 account books."""

import csv
from datetime import date, timedelta
from math import isclose, prod
from pathlib import Path
from typing import Literal

import pandas as pd
from beartype import beartype
from pydantic import ConfigDict, TypeAdapter

from rotbt.models import Metrics, Model
from scripts.etf_13612w import OUT, RESEARCH, Reference, Trial, catalog, public_root, references
from scripts.etf_group_cap_audit import sha, verify_hashes
from scripts.etf_group_cap_report import summarize
from scripts.etf_group_cap_targets import Cohort

Score = Literal["calendar_13612w", "original252", "benchmark"]


class Binding(Model):
    key: str
    score: Score
    rule: str
    k: int | None
    cost_bps: Literal[5, 20]
    cohort: Cohort
    path: Path
    mode: Literal["new_native", "reuse"]
    source_hashes: dict[str, str]


class Registry(Model):
    roles: list[Binding]


class FullAudit(Model):
    status: Literal["PASS"]
    keys: list[str]
    science: Literal[18]
    bridges: Literal[2]
    reuse_roles: Literal[36]
    report_roles: Literal[54]
    journal: dict[str, int]
    hashes: dict[str, str]


class Labels(Model):
    key: str
    score: Score
    rule: str
    k: int | None
    cost_bps: Literal[5, 20]
    cohort: Cohort
    period: str


class Period(Labels):
    start: date
    end: date
    days: int
    opening: float
    closing: float
    total_return: float
    max_drawdown: float
    cagr_252: float | None
    annualized_volatility: float
    sharpe_rf0: float
    fees: float
    orders: int
    turnover: float


class Month(Labels):
    opening: float
    closing: float
    total_return: float


class Pair(Model):
    new_key: str
    original_key: str
    k: int
    cost_bps: Literal[5, 20]
    cohort: Cohort
    period: str
    new_return: float
    original_return: float
    return_difference: float
    new_drawdown: float
    original_drawdown: float
    drawdown_difference: float
    fees_difference: float
    orders_difference: int
    turnover_difference: float


class Ytd(Model):
    model_config = ConfigDict(extra="ignore")
    days: int
    opening_equity: float
    closing_equity: float
    total_return: float
    max_drawdown: float
    sharpe_rf0: float
    annualized_volatility: float
    fees: float
    orders: int
    monthly_returns: dict[str, float]


@beartype
def planned_bindings(root: Path, trials: list[Trial], reused: list[Reference]) -> list[Binding]:
    output = [
        Binding(
            key=trial.key,
            score=trial.score,
            rule={1: "C10", 2: "C11", 3: "C00"}[trial.k],
            k=trial.k,
            cost_bps=trial.cost_bps,
            cohort=trial.cohort,
            path=root / OUT / trial.key,
            mode="new_native",
            source_hashes={},
        )
        for trial in trials
        if trial.kind == "science"
    ]
    for account in reused:
        folder = public_root(root) / "standard/reference" / account.key
        output.append(
            Binding(
                key=account.key,
                score="original252" if account.rule in ("C10", "C11", "C00") else "benchmark",
                rule=account.rule,
                k={"C10": 1, "C11": 2, "C00": 3}.get(account.rule),
                cost_bps=account.cost_bps,
                cohort=account.cohort,
                path=folder,
                mode="reuse",
                source_hashes={
                    str(folder / name): checksum
                    for name, checksum in account.reference_files.items()
                },
            )
        )
    return output


@beartype
def validate_registry(actual: list[Binding], expected: list[Binding]) -> None:
    if len(actual) != len(expected) or sorted(actual, key=lambda r: r.key) != sorted(
        expected, key=lambda r: r.key
    ):
        raise ValueError("report registry differs from the frozen science and reuse roles")


@beartype
def make_period(
    binding: Binding,
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    first: date,
    last: date,
    period: str,
) -> Period:
    if daily.empty or daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise ValueError("empty or unordered native NAV")
    selected = daily.loc[str(first) : str(last)]
    if selected.empty:
        raise ValueError("empty reporting window")
    prior = daily.loc[daily.index < pd.Timestamp(first), "equity"]
    if binding.cohort == "continuous2022_2026" and first.year == 2026:
        if pd.Timestamp("2025-12-31") not in daily.index:
            raise ValueError("continuous account requires its own December 31 NAV")
    opening = float(prior.iloc[-1]) if len(prior) else 50_000.0
    stats = summarize(daily, orders, first, last, opening)
    return Period(
        key=binding.key,
        score=binding.score,
        rule=binding.rule,
        k=binding.k,
        cost_bps=binding.cost_bps,
        cohort=binding.cohort,
        period=period,
        start=pd.Timestamp(selected.index[0]).date(),
        end=pd.Timestamp(selected.index[-1]).date(),
        days=stats.days,
        opening=stats.opening,
        closing=stats.closing,
        total_return=stats.total_return,
        max_drawdown=stats.max_drawdown,
        cagr_252=stats.cagr_252 if last.year < 2026 else None,
        annualized_volatility=stats.annualized_volatility,
        sharpe_rf0=stats.sharpe_rf0,
        fees=stats.fees,
        orders=stats.orders,
        turnover=stats.turnover,
    )


@beartype
def paired_comparisons(rows: list[Period]) -> list[Pair]:
    output = []
    for new in rows:
        if new.score != "calendar_13612w":
            continue
        matches = [
            old
            for old in rows
            if old.score == "original252"
            and (old.k, old.cost_bps, old.cohort, old.period)
            == (new.k, new.cost_bps, new.cohort, new.period)
        ]
        if len(matches) != 1 or new.k is None:
            raise ValueError("each science row requires a unique matched original252 role")
        old = matches[0]
        if (new.start, new.end, new.days) != (old.start, old.end, old.days):
            raise ValueError("paired windows differ")
        output.append(
            Pair(
                new_key=new.key,
                original_key=old.key,
                k=new.k,
                cost_bps=new.cost_bps,
                cohort=new.cohort,
                period=new.period,
                new_return=new.total_return,
                original_return=old.total_return,
                return_difference=new.total_return - old.total_return,
                new_drawdown=new.max_drawdown,
                original_drawdown=old.max_drawdown,
                drawdown_difference=new.max_drawdown - old.max_drawdown,
                fees_difference=new.fees - old.fees,
                orders_difference=new.orders - old.orders,
                turnover_difference=new.turnover - old.turnover,
            )
        )
    return output


def equal(actual: float, expected: float) -> None:
    if not isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"independent archived metric mismatch: {actual} != {expected}")


def describe_binding(binding: Binding) -> tuple[list[Period], list[Month]]:
    daily = pd.read_csv(
        binding.path / "daily.csv.gz", parse_dates=["day"], float_precision="round_trip"
    ).set_index("day")
    orders = pd.read_csv(binding.path / "orders.csv.gz", float_precision="round_trip")
    historical = binding.cohort == "historical2022_2025"
    expected_start = date(2026, 1, 2) if binding.cohort == "reset2026" else date(2022, 1, 3)
    through = date(2025, 12, 31) if historical else date(2026, 11, 6)
    if (
        daily.empty
        or pd.Timestamp(daily.index[0]).date() != expected_start
        or pd.Timestamp(daily.index[-1]).date() != through
    ):
        raise ValueError("saved account boundary differs from frozen cohort")
    windows = [(str(y), date(y, 1, 1), date(y, 12, 31)) for y in range(2022, 2026)]
    windows.append(("full2022_2025", date(2022, 1, 3), through))
    if not historical:
        windows = [("2026YTD", date(2026, 1, 2), through)]
    rows = [
        make_period(binding, daily, orders, first, last, period) for period, first, last in windows
    ]
    first = date(2022, 1, 3) if historical else date(2026, 1, 2)
    selected = daily.loc[str(first) : str(through), "equity"]
    previous = rows[-1].opening
    months = []
    for month in sorted({pd.Timestamp(day).strftime("%Y-%m") for day in selected.index}):
        closing = float(selected.loc[month].iloc[-1])
        months.append(
            Month(
                key=binding.key,
                score=binding.score,
                rule=binding.rule,
                k=binding.k,
                cost_bps=binding.cost_bps,
                cohort=binding.cohort,
                period=month,
                opening=previous,
                closing=closing,
                total_return=closing / previous - 1,
            )
        )
        previous = closing
    equal(prod(1 + row.total_return for row in months), 1 + rows[-1].total_return)
    name = "METRICS.json" if historical else "YTD.json"
    if binding.mode == "reuse":
        name = name.lower()
    if historical:
        archived = TypeAdapter(list[Metrics]).validate_json((binding.path / name).read_text())
        if {r.period for r in archived} != {
            "early",
            "late",
            "full",
            "2022",
            "2023",
            "2024",
            "2025",
        }:
            raise ValueError("missing archived historical metric periods")
        for reference in archived:
            a, b = (
                (2022, 2023)
                if reference.period == "early"
                else (
                    (2024, 2025)
                    if reference.period == "late"
                    else (2022, 2025)
                    if reference.period == "full"
                    else (int(reference.period), int(reference.period))
                )
            )
            actual = make_period(
                binding, daily, orders, date(a, 1, 1), date(b, 12, 31), reference.period
            )
            for x, y in (
                (actual.total_return, reference.total_return),
                (actual.days, reference.days),
                (actual.max_drawdown, reference.max_dd),
                (actual.sharpe_rf0, reference.sharpe),
                (actual.fees, reference.fees),
                (actual.orders, reference.orders),
                (actual.turnover, reference.turnover),
            ):
                equal(x, y)
            if actual.cagr_252 is None:
                raise ValueError("historical CAGR missing")
            equal(actual.cagr_252, reference.cagr)
        equal(prod(1 + row.total_return for row in rows[:-1]), 1 + rows[-1].total_return)
    else:
        ytd = Ytd.model_validate_json((binding.path / name).read_text())
        actual = rows[0]
        for x, y in (
            (actual.days, ytd.days),
            (actual.opening, ytd.opening_equity),
            (actual.closing, ytd.closing_equity),
            (actual.total_return, ytd.total_return),
            (actual.max_drawdown, ytd.max_drawdown),
            (actual.sharpe_rf0, ytd.sharpe_rf0),
            (actual.annualized_volatility, ytd.annualized_volatility),
            (actual.fees, ytd.fees),
            (actual.orders, ytd.orders),
        ):
            equal(x, y)
        if {row.period for row in months} != set(ytd.monthly_returns):
            raise ValueError("archived month coverage differs")
        for monthly in months:
            equal(monthly.total_return, ytd.monthly_returns[monthly.period])
    return rows, months


def write_table[T: Model](path: Path, rows: list[T]) -> None:
    if not rows:
        raise ValueError("empty report table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(type(rows[0]).model_fields))
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)


def plots(root: Path, roles: list[Binding]) -> None:
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
    for cohort, filename in (
        ("historical2022_2025", "HISTORICAL.png"),
        ("reset2026", "RESET2026.png"),
    ):
        chosen = [
            role
            for role in roles
            if role.cohort == cohort
            and role.cost_bps == 5
            and (role.score == "calendar_13612w" or role.rule in ("QQQ_BH", "SPY_BH"))
        ]
        chosen.sort(key=lambda row: (row.score != "calendar_13612w", row.k or 0, row.rule))
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [2, 1]},
        )
        for role, color in zip(
            chosen, ("#0072B2", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"), strict=True
        ):
            daily = pd.read_csv(
                role.path / "daily.csv.gz", parse_dates=["day"], float_precision="round_trip"
            )
            dates = [pd.Timestamp(daily.day.iloc[0]).date() - timedelta(days=1)] + [
                pd.Timestamp(x).date() for x in daily.day
            ]
            values = np.r_[50_000.0, daily.equity.to_numpy(dtype=float)]
            label = f"13612W K={role.k}" if role.score == "calendar_13612w" else role.rule
            axes[0].plot(dates, values / values[0] * 100, color=color, label=label, linewidth=1.7)
            axes[1].plot(
                dates, values / np.maximum.accumulate(values) - 1, color=color, linewidth=1.3
            )
        title = (
            "2022-2025; last-day open liquidation"
            if cohort == "historical2022_2025"
            else "2026 reset; through Nov 6; seen sample"
        )
        axes[0].set_title(
            f"Calendar-month 13612W | {title}\n5 bp one-way slippage plus fees", pad=36
        )
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=5, frameon=False)
        axes[0].set_ylabel("NAV (initial capital = 100)")
        axes[1].set_ylabel("Drawdown from prior peak")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        for axis in axes:
            axis.grid(alpha=0.18)
            axis.margins(x=0.01)
        figure.savefig(root / RESEARCH / filename)
        plt.close(figure)


@beartype
def main(root: Path | None = None) -> None:
    root = root.resolve() if root is not None else Path(__file__).resolve().parents[1]
    audit_path = root / OUT / "FULL_AUDIT.json"
    if not audit_path.is_file():
        raise ValueError("FULL_AUDIT.json must pass before reporting any new results")
    audit_sha = sha(audit_path)
    audit = FullAudit.model_validate_json(audit_path.read_text())
    verify_hashes(audit.hashes)
    trials = catalog(root)
    if set(audit.keys) != {trial.key for trial in trials} or len(audit.keys) != 20:
        raise ValueError("full audit does not cover the exact20 frozen native trial identities")
    registry_path = root / OUT / "REGISTRY.json"
    if str(registry_path) not in audit.hashes:
        raise ValueError("full audit must bind the reporting registry")
    expected = planned_bindings(root, trials, references(root))
    roles = Registry.model_validate_json(registry_path.read_text()).roles
    validate_registry(roles, expected)
    if len(roles) != 54:
        raise ValueError("54 distinct science and reuse roles required")
    for role in roles:
        filenames = [
            "daily.csv.gz",
            "orders.csv.gz",
            "METRICS.json" if role.cohort == "historical2022_2025" else "YTD.json",
        ]
        required = [
            str(role.path / (name.lower() if role.mode == "reuse" else name)) for name in filenames
        ]
        if not set(required) <= audit.hashes.keys() or any(
            audit.hashes.get(p) != h for p, h in role.source_hashes.items()
        ):
            raise ValueError("full audit is missing exact reporting source hashes")
    periods: list[Period] = []
    months: list[Month] = []
    for role in roles:
        rows, monthly = describe_binding(role)
        periods.extend(rows)
        months.extend(monthly)
    summaries = sorted(
        (row for row in periods if row.period in ("full2022_2025", "2026YTD")),
        key=lambda row: (row.cohort, row.cost_bps, -row.total_return),
    )
    pairs = paired_comparisons(periods)
    if (len(periods), len(summaries), len(months), len(pairs)) != (126, 54, 1260, 42):
        raise ValueError("report coverage differs from the preregistered54 roles")
    verify_hashes(audit.hashes)
    if sha(audit_path) != audit_sha:
        raise ValueError("full audit changed during reporting")
    for name, data in (("PERIOD_METRICS", periods), ("SUMMARY", summaries)):
        write_table(root / RESEARCH / f"{name}.csv", data)
    write_table(root / RESEARCH / "MONTHLY.csv", months)
    write_table(root / RESEARCH / "PAIRED_COMPARISON.csv", pairs)
    plots(root, roles)
    verify_hashes(audit.hashes)
    if sha(audit_path) != audit_sha:
        raise ValueError("full audit changed while rendering")
    print("PASS: 54 roles,126 period rows,1260 months,42 matched comparisons; no native run")


if __name__ == "__main__":
    main()
