"""Fixed six-mechanism report: saved-account statistics and local-family inference only."""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter

from rotbt.models import BootstrapResult, Model
from rotbt.statistics import max_bootstrap
from scripts import etf_13612w_report as saved
from scripts.etf_group_cap_audit import sha, verify_hashes
from scripts.etf_group_cap_targets import Cohort

RESEARCH = "research/ETF_MOMENTUM_VARIANTS_20261110"
OUT = "etf_momentum_variants_outputs/20261110_v1"
FAMILY = ("E01", "E02", "E03", "E04", "E05", "E06")
Variant = Literal["E01", "E02", "E03", "E04", "E05", "E06", "BASE252", "W13612", "benchmark"]


class Role(Model):
    key: str
    variant: Variant
    rule: str
    k: int | None
    cost_bps: Literal[5, 20]
    cohort: Cohort
    path: Path
    mode: Literal["new_native", "reuse"]
    source_hashes: dict[str, str]


class Registry(Model):
    roles: list[Role]


class FullAudit(Model):
    status: Literal["PASS"]
    keys: list[str]
    science: Literal[36]
    bridges: Literal[2]
    reuse_roles: Literal[30]
    report_roles: Literal[66]
    journal: dict[str, int]
    hashes: dict[str, str]


class Metric(Model):
    key: str
    variant: Variant
    rule: str
    k: int | None
    cost_bps: Literal[5, 20]
    cohort: Cohort
    period: str
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


class Monthly(Model):
    key: str
    variant: Variant
    rule: str
    k: int | None
    cost_bps: Literal[5, 20]
    cohort: Cohort
    month: str
    opening: float
    closing: float
    total_return: float


class Pair(saved.Pair):
    variant: Variant
    baseline: Literal["BASE252", "W13612"]


class ExcessPanel(Model):
    names: tuple[str, ...]
    days: tuple[date, ...]
    differences: NDArray[np.float64]


class Statistic(Model):
    variant: str
    cost_bps: int
    block: int
    repetitions: Literal[10000] = 10000
    family_size: Literal[6] = 6
    annualized_arithmetic_excess: float
    pointwise_ci_low: float
    pointwise_ci_high: float
    adjusted_p: float
    family_p: float
    inference_scope: str = "local six rules; pointwise arithmetic CI, not CAGR or global-search CI"


class Pareto(Model):
    variant: str
    cost_bps: int
    cagr_difference: float
    drawdown_difference: float
    years_beating_baseline: int
    worst_year_difference: float
    pareto_at_cost: bool
    pareto_both_costs: bool
    local_statistical_support: bool


class Target(Model):
    model_config = ConfigDict(extra="ignore")
    day: date
    q: dict[str, float]


class SamePath(Model):
    key: str
    variant: Variant
    cost_bps: int
    cohort: Cohort
    target_sha256: str
    nav_sha256: str
    same_target_keys: str = ""
    same_nav_keys: str = ""


def parser_role(role: Role) -> saved.Binding:
    # The saved parser's mode chooses JSON filename case only; it never executes an account.
    return saved.Binding(
        key=role.key,
        score="original252",
        rule=role.rule,
        k=role.k,
        cost_bps=role.cost_bps,
        cohort=role.cohort,
        path=role.path,
        mode="new_native" if role.variant in (*FAMILY, "W13612") else "reuse",
        source_hashes=role.source_hashes,
    )


def relabel(role: Role, row: saved.Period) -> Metric:
    return Metric(
        key=role.key,
        variant=role.variant,
        rule=role.rule,
        k=role.k,
        cost_bps=role.cost_bps,
        cohort=role.cohort,
        period=row.period,
        start=row.start,
        end=row.end,
        days=row.days,
        opening=row.opening,
        closing=row.closing,
        total_return=row.total_return,
        max_drawdown=row.max_drawdown,
        cagr_252=row.cagr_252,
        annualized_volatility=row.annualized_volatility,
        sharpe_rf0=row.sharpe_rf0,
        fees=row.fees,
        orders=row.orders,
        turnover=row.turnover,
    )


@beartype
def period_row(
    role: Role, daily: pd.DataFrame, orders: pd.DataFrame, first: date, last: date, period: str
) -> Metric:
    return relabel(role, saved.make_period(parser_role(role), daily, orders, first, last, period))


@beartype
def paired_rows(rows: list[Metric]) -> list[Pair]:
    output = []
    for new in [row for row in rows if row.variant in FAMILY]:
        for baseline in ("BASE252", "W13612"):
            matched = [
                row
                for row in rows
                if row.variant == baseline
                and (row.k, row.cost_bps, row.cohort, row.period)
                == (new.k, new.cost_bps, new.cohort, new.period)
            ]
            if len(matched) != 1 or new.k != 2:
                raise ValueError("unique matched top2 baseline required")
            old = matched[0]
            if (new.start, new.end, new.days) != (old.start, old.end, old.days):
                raise ValueError("paired account periods differ")
            output.append(
                Pair(
                    variant=new.variant,
                    baseline=baseline,
                    new_key=new.key,
                    original_key=old.key,
                    k=2,
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


@beartype
def validate_panel(panel: ExcessPanel) -> None:
    expected = tuple(
        t.date() for t in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2025-12-31")
    )
    if panel.names != FAMILY or panel.differences.ndim != 2 or panel.differences.shape[1] != 6:
        raise ValueError("complete ordered six-rule family required, including identical paths")
    if (
        panel.days != expected
        or panel.differences.shape != (1003, 6)
        or not np.isfinite(panel.differences).all()
    ):
        raise ValueError("exact finite 1003-session historical window required")


def historical_panel(roles: list[Role], cost: int) -> ExcessPanel:
    arrays: list[NDArray[np.float64]] = []
    days: tuple[date, ...] = ()
    for variant in ("BASE252", *FAMILY):
        selected = [
            r
            for r in roles
            if r.variant == variant and r.cost_bps == cost and r.cohort == "historical2022_2025"
        ]
        if len(selected) != 1:
            raise ValueError("one historical account per fixed family member required")
        daily = pd.read_csv(selected[0].path / "daily.csv.gz", float_precision="round_trip")
        observed = tuple(pd.Timestamp(x).date() for x in daily.day)
        if days and observed != days:
            raise ValueError("historical comparison calendars differ")
        days = observed
        nav = np.r_[50_000.0, daily.equity.to_numpy(dtype=float)]
        if not np.isfinite(nav).all() or (nav <= 0).any():
            raise ValueError("invalid historical NAV")
        arrays.append(nav[1:] / nav[:-1] - 1)
    panel = ExcessPanel(
        names=FAMILY,
        days=days,
        differences=np.column_stack([row - arrays[0] for row in arrays[1:]]),
    )
    validate_panel(panel)
    return panel


@beartype
def statistics_rows(result: BootstrapResult, cost: int) -> list[Statistic]:
    if result.names != FAMILY or result.block not in (5, 20, 60):
        raise ValueError("unexpected statistical family or block")
    return [
        Statistic(
            variant=name,
            cost_bps=cost,
            block=result.block,
            annualized_arithmetic_excess=result.mean_bp[i] * 252 / 10000,
            pointwise_ci_low=result.ci_low_bp[i] * 252 / 10000,
            pointwise_ci_high=result.ci_high_bp[i] * 252 / 10000,
            adjusted_p=result.adjusted_p[i],
            family_p=result.family_p,
        )
        for i, name in enumerate(FAMILY)
    ]


def annual_support(statistics: list[Statistic], variant: str) -> bool:
    rows = [row for row in statistics if row.variant == variant]
    return (
        len(rows) == 6
        and {(r.cost_bps, r.block) for r in rows}
        == {(cost, block) for cost in (5, 20) for block in (5, 20, 60)}
        and all(row.adjusted_p < 0.05 for row in rows)
    )


def pareto_rows(rows: list[Metric], statistics: list[Statistic]) -> list[Pareto]:
    output: list[Pareto] = []
    for variant in FAMILY:
        local = []
        for cost in (5, 20):
            new = [
                r
                for r in rows
                if r.variant == variant and r.cost_bps == cost and r.cohort == "historical2022_2025"
            ]
            old = [
                r
                for r in rows
                if r.variant == "BASE252"
                and r.cost_bps == cost
                and r.cohort == "historical2022_2025"
            ]
            a, b = (
                next(r for r in new if r.period == "full2022_2025"),
                next(r for r in old if r.period == "full2022_2025"),
            )
            if a.cagr_252 is None or b.cagr_252 is None:
                raise ValueError("historical Pareto requires CAGR")
            cagr, dd = a.cagr_252 - b.cagr_252, a.max_drawdown - b.max_drawdown
            annual = [
                (
                    next(r.total_return for r in new if r.period == str(year)),
                    next(r.total_return for r in old if r.period == str(year)),
                )
                for year in range(2022, 2026)
            ]
            local.append(
                Pareto(
                    variant=variant,
                    cost_bps=cost,
                    cagr_difference=cagr,
                    drawdown_difference=dd,
                    years_beating_baseline=sum(x > y + 1e-10 for x, y in annual),
                    worst_year_difference=min(x for x, _ in annual) - min(y for _, y in annual),
                    pareto_at_cost=cagr >= -1e-10 and dd <= 1e-10 and (cagr > 1e-10 or dd < -1e-10),
                    pareto_both_costs=False,
                    local_statistical_support=annual_support(statistics, variant),
                )
            )
        output.extend(
            r.model_copy(update={"pareto_both_costs": all(x.pareto_at_cost for x in local)})
            for r in local
        )
    return output


@beartype
def target_hash(content: str) -> str:
    rows = TypeAdapter(list[Target]).validate_json(content)
    if [r.day for r in rows] != sorted({r.day for r in rows}):
        raise ValueError("target dates must be unique and ordered")
    canonical = [(row.day.isoformat(), sorted(row.q.items())) for row in rows]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def same_paths(roles: list[Role]) -> list[SamePath]:
    rows = []
    for role in roles:
        filename = "TARGETS.json" if role.variant in (*FAMILY, "W13612") else "targets.json"
        daily = pd.read_csv(role.path / "daily.csv.gz", float_precision="round_trip")
        canonical = [
            (str(day), float(nav).hex()) for day, nav in zip(daily.day, daily.equity, strict=True)
        ]
        rows.append(
            SamePath(
                key=role.key,
                variant=role.variant,
                cost_bps=role.cost_bps,
                cohort=role.cohort,
                target_sha256=target_hash((role.path / filename).read_text()),
                nav_sha256=hashlib.sha256(json.dumps(canonical).encode()).hexdigest(),
            )
        )
    return [
        row.model_copy(
            update={
                "same_target_keys": "|".join(
                    other.key
                    for other in rows
                    if other.key != row.key
                    and other.cohort == row.cohort
                    and other.target_sha256 == row.target_sha256
                ),
                "same_nav_keys": "|".join(
                    other.key
                    for other in rows
                    if other.key != row.key
                    and other.cohort == row.cohort
                    and other.nav_sha256 == row.nav_sha256
                ),
            }
        )
        for row in rows
    ]


def figures(root: Path, summaries: list[Metric], pairs: list[Pair]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = (*FAMILY, "BASE252", "W13612")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, cost in zip(axes, (5, 20), strict=True):
        for i, name in enumerate(names):
            row = next(
                r
                for r in summaries
                if r.variant == name and r.cost_bps == cost and r.cohort == "historical2022_2025"
            )
            axis.scatter(
                row.max_drawdown,
                row.total_return,
                color=f"C{i}",
                marker=("o", "s", "^", "v", "P", "X", "D", "*")[i],
                s=60,
                label=name,
                facecolors="none",
            )
        axis.set(
            title=f"2022-2025 | {cost} bp per side",
            xlabel="Maximum drawdown",
            ylabel="Total net return",
        )
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.2)
        axis.margins(0.2)
        axis.legend(loc="best", ncols=2, fontsize=8, frameon=False)
    figure.savefig(root / RESEARCH / "RETURN_DRAWDOWN.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for axis, cost in zip(axes, (5, 20), strict=True):
        values = np.asarray(
            [
                [
                    next(
                        r.return_difference
                        for r in pairs
                        if r.variant == name
                        and r.baseline == "BASE252"
                        and r.cost_bps == cost
                        and r.period == str(year)
                    )
                    for year in range(2022, 2026)
                ]
                for name in FAMILY
            ]
        )
        bound = max(float(np.abs(values).max()), 0.001)
        heat = axis.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_xticks(range(4), ["2022", "2023", "2024", "2025"])
        axis.set_yticks(range(6), FAMILY)
        axis.set_title(f"Annual return minus BASE252 | {cost} bp")
        for i in range(6):
            for j in range(4):
                axis.text(
                    j,
                    i,
                    f"{values[i, j]:+.1%}",
                    ha="center",
                    va="center",
                    color="white" if abs(values[i, j]) > bound * 0.6 else "black",
                )
        figure.colorbar(heat, ax=axis, format=PercentFormatter(1))
    figure.savefig(root / RESEARCH / "ANNUAL_EXCESS.png", dpi=170)
    plt.close(figure)


@beartype
def main(root: Path | None = None) -> None:
    from scripts.etf_momentum_variants import catalog, expected_registry

    root = root.resolve() if root else Path(__file__).resolve().parents[1]
    audit_path, registry_path = root / OUT / "FULL_AUDIT.json", root / OUT / "REGISTRY.json"
    if not audit_path.is_file():
        raise ValueError("FULL_AUDIT required before reading new results")
    audit_sha, audit = sha(audit_path), FullAudit.model_validate_json(audit_path.read_text())
    verify_hashes(audit.hashes)
    roles = Registry.model_validate_json(registry_path.read_text()).roles
    expected = TypeAdapter(list[Role]).validate_python(expected_registry(root))
    if len(roles) != 66 or sorted(roles, key=lambda r: r.key) != sorted(
        expected, key=lambda r: r.key
    ):
        raise ValueError("exact frozen66 report identities required")
    if (
        len(audit.keys) != 38
        or set(audit.keys) != {r.key for r in catalog(root)}
        or str(registry_path) not in audit.hashes
        or not 38 <= audit.journal.get("started", 0) <= 48
    ):
        raise ValueError("full audit must bind actual38 trial identities and registry")
    if str(root / RESEARCH / "CAUSALITY_RECEIPT.json") not in audit.hashes:
        raise ValueError("full audit must bind causality evidence")
    rows: list[Metric] = []
    months: list[Monthly] = []
    for role in roles:
        parser = parser_role(role)
        filenames = (
            "daily.csv.gz",
            "orders.csv.gz",
            "TARGETS.json",
            "METRICS.json" if role.cohort == "historical2022_2025" else "YTD.json",
        )
        required = {
            str(role.path / (name.lower() if parser.mode == "reuse" else name))
            for name in filenames
        }
        if not required <= audit.hashes.keys() or any(
            audit.hashes.get(p) != h for p, h in role.source_hashes.items()
        ):
            raise ValueError("missing bound report source")
        periods, monthly = saved.describe_binding(parser)
        rows.extend(relabel(role, row) for row in periods)
        months.extend(
            Monthly(
                key=role.key,
                variant=role.variant,
                rule=role.rule,
                k=role.k,
                cost_bps=role.cost_bps,
                cohort=role.cohort,
                month=m.period,
                opening=m.opening,
                closing=m.closing,
                total_return=m.total_return,
            )
            for m in monthly
        )
    summaries = sorted(
        (r for r in rows if r.period in ("full2022_2025", "2026YTD")),
        key=lambda r: (r.cohort, r.cost_bps, -r.total_return),
    )
    pairs = paired_rows(rows)
    if (len(rows), len(summaries), len(months), len(pairs)) != (154, 66, 1540, 168):
        raise ValueError("fixed report coverage mismatch")
    statistics = []
    for cost in (5, 20):
        panel = historical_panel(roles, cost)
        for block in (5, 20, 60):
            print(f"Bootstrap: {cost} bp, block={block}, B=10000, six rules, 1003 days", flush=True)
            statistics.extend(
                statistics_rows(max_bootstrap(panel.differences, FAMILY, block, 10000), cost)
            )
    pareto, paths = pareto_rows(rows, statistics), same_paths(roles)
    verify_hashes(audit.hashes)
    if sha(audit_path) != audit_sha:
        raise ValueError("full audit changed during report")
    saved.write_table(root / RESEARCH / "PERIOD_METRICS.csv", rows)
    saved.write_table(root / RESEARCH / "SUMMARY.csv", summaries)
    saved.write_table(root / RESEARCH / "MONTHLY.csv", months)
    saved.write_table(root / RESEARCH / "PAIRED_COMPARISON.csv", pairs)
    saved.write_table(root / RESEARCH / "STATISTICS.csv", statistics)
    saved.write_table(root / RESEARCH / "PARETO.csv", pareto)
    saved.write_table(root / RESEARCH / "SAME_PATH.csv", paths)
    figures(root, summaries, pairs)
    verify_hashes(audit.hashes)
    if sha(audit_path) != audit_sha:
        raise ValueError("full audit changed during rendering")
    print("PASS:66roles/154periods/1540months/168pairs/36statistics; native0")


if __name__ == "__main__":
    main()
