"""Recompute both published experiments from included books without native simulation."""

import argparse
import hashlib
import io
import json
from datetime import date
from math import fsum, isclose
from pathlib import Path
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype
from pydantic import TypeAdapter

from c10share.report import read_csv
from c10standard.audit import Description
from c10standard.io import ROOT, load_strategy
from c10standard.models import NAMES, NativeScalar, TargetRecord
from c10standard.report import Book, Nav, Statistics, load_books, summarize, verify_book
from rotbt.models import DailyRecord, Distribution, HoldingRecord, Metrics, Model, OrderRecord

from .statistics import max_bootstrap

FAMILY = ("E01", "E02", "E03", "E04", "E05", "E06")
STUDIES = ("calendar13612w", "top2-momentum-variants")
METRIC_COLUMNS = [
    "key",
    "label",
    "rule",
    "k",
    "cost_bps",
    "cohort",
    "period",
    "start",
    "end",
    "days",
    "opening",
    "closing",
    "total_return",
    "max_drawdown",
    "cagr_252",
    "annualized_volatility",
    "sharpe_rf0",
    "fees",
    "orders",
    "turnover",
]


class Case(Model):
    key: str
    variant: str
    folder: Path
    book: Book
    targets: tuple[TargetRecord, ...]
    holdings: tuple[HoldingRecord, ...]

    @property
    def k(self) -> int | None:
        return {"C10": 1, "C11": 2, "C00": 3}.get(self.book.contract.rule)


class Table(Model):
    name: str
    keys: tuple[str, ...]
    frame: pd.DataFrame


class TableCheck(Model):
    study: str
    table: str
    rows: int
    columns: int
    cells: int


class Fingerprints(Model):
    hashes: dict[str, str]


class Receipt(Model):
    status: Literal["PASS"] = "PASS"
    experiment_accounts: Literal[54] = 54
    standard_accounts: Literal[36] = 36
    calendar_report_roles: Literal[54] = 54
    variant_report_roles: Literal[66] = 66
    checked_book_metric_values: int
    bootstrap_runs: Literal[6] = 6
    bootstrap_repetitions: Literal[10000] = 10000
    native_attempts: Literal[0] = 0
    table_checks: list[TableCheck]
    input_sha256: dict[str, str]
    output_sha256: dict[str, str]
    notes: tuple[str, ...] = (
        "Recomputed from saved NAV, orders, dividends, holdings and targets; "
        "no native account run.",
        "2026 continuous denominators are own December 31 NAV; YTD CAGR remains blank.",
        "Six-rule bootstrap includes identical paths, uses B10000 and seed20261105+block.",
        "CI is pointwise annualized arithmetic excess, not CAGR or global-search adjustment.",
        "Original published CSV files are comparison inputs and remain unchanged.",
    )


@beartype
def report_period(book: Book, period: str) -> Statistics:
    if period == "2026YTD":
        if book.contract.cohort == "continuous2022_2026" and not any(
            row.day == date(2025, 12, 31) for row in book.daily
        ):
            raise ValueError("continuous account requires own December 31 NAV")
        first, last = date(2026, 1, 2), date(2026, 11, 6)
    elif period == "full2022_2025":
        first, last = date(2022, 1, 3), date(2025, 12, 31)
    else:
        first, last = date(int(period), 1, 1), date(int(period), 12, 31)
    return summarize(book, first, last, period)


@beartype
def historical_returns(book: Book) -> NDArray[np.float64]:
    expected = [
        x.date() for x in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2025-12-31")
    ]
    if [row.day for row in book.daily] != expected or len(expected) != 1003:
        raise ValueError("bootstrap requires exact 1003 historical sessions")
    nav = np.asarray([book.initial_cash] + [row.equity for row in book.daily], dtype=float)
    if not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError("invalid historical NAV")
    return np.asarray(nav[1:] / nav[:-1] - 1, dtype=float)


@beartype
def excess_matrix(cases: list[Case], cost: int) -> NDArray[np.float64]:
    arrays = []
    for name in ("BASE252", *FAMILY):
        selected = [
            c
            for c in cases
            if c.variant == name
            and c.book.contract.terminal
            and c.book.contract.cost_bps == cost
            and c.k == 2
        ]
        if len(selected) != 1:
            raise ValueError("one BASE252 and all six distinct top2 cases required")
        arrays.append(historical_returns(selected[0].book))
    return np.column_stack([row - arrays[0] for row in arrays[1:]])


@beartype
def compare_table(table: Table, archived: Path) -> int:
    actual = pd.read_csv(archived, dtype=str, keep_default_na=False)
    expected = pd.read_csv(
        io.StringIO(table.frame.to_csv(index=False)), dtype=str, keep_default_na=False
    )
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ValueError(f"CSV columns or row count changed: {archived.name}")
    keys = list(table.keys)
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"duplicate CSV identity: {archived.name}")
    actual, expected = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not actual.index.equals(expected.index):
        raise ValueError(f"CSV case or period identity changed: {archived.name}")
    for column in actual:
        left, right = actual[column], expected[column]
        if not (left.eq("") == right.eq("")).all():
            raise ValueError(f"CSV blank value changed: {archived.name}/{column}")
        if is_numeric_dtype(table.frame[column]) and table.frame[column].dtype != bool:
            keep = ~right.eq("")
            a, b = (
                pd.to_numeric(left[keep]).to_numpy(dtype=float),
                pd.to_numeric(right[keep]).to_numpy(dtype=float),
            )
            if not np.isfinite(a).all() or not np.allclose(a, b, rtol=1e-10, atol=1e-8):
                raise ValueError(f"CSV numeric drift: {archived.name}/{column}")
        elif column in ("same_target_keys", "same_nav_keys"):
            if [sorted(x.split("|")) for x in left] != [sorted(x.split("|")) for x in right]:
                raise ValueError(f"CSV same-path identity drift: {archived.name}/{column}")
        elif not left.equals(right):
            raise ValueError(f"CSV value drift: {archived.name}/{column}")
    return len(expected) * len(table.frame.columns)


def load_cases(root: Path) -> tuple[list[Case], int]:
    from .io import load_experiments, reference_folder

    strategy = load_strategy(root)
    books, _ = load_books(root)
    cases = [
        Case(
            key=b.contract.key,
            variant="BASE252" if b.contract.rule in ("C10", "C11", "C00") else "benchmark",
            folder=root / "standard/reference" / b.contract.key,
            book=b,
            targets=TypeAdapter(tuple[TargetRecord, ...]).validate_json(
                (root / "standard/reference" / b.contract.key / "targets.json").read_bytes()
            ),
            holdings=read_csv(
                root / "standard/reference" / b.contract.key / "holdings.csv.gz", HoldingRecord
            ),
        )
        for b in books
    ]
    for experiment in load_experiments(root):
        folder = reference_folder(experiment, root)
        daily = read_csv(folder / "daily.csv.gz", DailyRecord)
        contract = experiment.contract
        expected = [
            x.date()
            for x in xcals.get_calendar("XNYS").sessions_in_range(contract.start, contract.through)
        ]
        if [row.day for row in daily] != expected or (contract.terminal and daily[-1].positions):
            raise ValueError(
                f"experiment calendar or terminal liquidation changed: {experiment.key}"
            )
        scalar = NativeScalar.model_validate_json(
            (folder / "NATIVE_TOTAL_RETURN.json").read_bytes()
        )
        if not isclose(
            daily[-1].equity / strategy.initial_cash - 1,
            scalar.native_total_return,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"native scalar mismatch: {experiment.key}")
        book = Book(
            contract=contract,
            initial_cash=strategy.initial_cash,
            daily=tuple(Nav(day=row.day, equity=row.equity) for row in daily),
            orders=read_csv(folder / "orders.csv.gz", OrderRecord),
            dividends=read_csv(folder / "dividends.csv.gz", Distribution),
            archived_metrics=TypeAdapter(tuple[Metrics, ...]).validate_json(
                (folder / "METRICS.json").read_bytes()
            )
            if contract.terminal
            else (),
            archived_ytd=None
            if contract.terminal
            else Description.model_validate_json((folder / "YTD.json").read_bytes()),
        )
        cases.append(
            Case(
                key=experiment.key,
                variant=experiment.variant,
                folder=folder,
                book=book,
                targets=TypeAdapter(tuple[TargetRecord, ...]).validate_json(
                    (folder / "TARGETS.json").read_bytes()
                ),
                holdings=read_csv(folder / "holdings.csv.gz", HoldingRecord),
            )
        )
    checks = 0
    if len(cases) != 90 or len({c.key for c in cases}) != 90:
        raise ValueError("90 unique explicit reference identities required")
    for case in cases:
        if case.book.contract.terminal and {m.period for m in case.book.archived_metrics} != {
            "early",
            "late",
            "full",
            "2022",
            "2023",
            "2024",
            "2025",
        }:
            raise ValueError("missing archived metric period")
        if [row.day for row in case.targets] != sorted({row.day for row in case.targets}):
            raise ValueError("unordered or duplicate targets")
        for target in case.targets:
            if not isclose(fsum(target.q.values()), 1.0, abs_tol=1e-10) or any(
                not np.isfinite(w) or w < 0 for w in target.q.values()
            ):
                raise ValueError("invalid target allocation")
        checks += verify_book(case.book)
    return cases, checks


def study_cases(cases: list[Case], study: str) -> list[Case]:
    if study == STUDIES[0]:
        return [c for c in cases if c.variant == "W13612"] + [
            c for c in cases if c.variant in ("BASE252", "benchmark")
        ]
    return (
        [c for c in cases if c.variant in FAMILY]
        + [c for c in cases if c.variant == "benchmark" or (c.variant == "BASE252" and c.k == 2)]
        + [c for c in cases if c.variant == "W13612" and c.k == 2]
    )


def core_tables(cases: list[Case], study: str) -> list[Table]:
    label = "score" if study == STUDIES[0] else "variant"
    periods, months = [], []
    for case in cases:
        value = (
            {"W13612": "calendar_13612w", "BASE252": "original252"}.get(case.variant, "benchmark")
            if label == "score"
            else case.variant
        )
        labels = {
            "key": case.key,
            label: value,
            "rule": case.book.contract.rule,
            "k": case.k,
            "cost_bps": case.book.contract.cost_bps,
            "cohort": case.book.contract.cohort,
        }
        names = (
            ["2022", "2023", "2024", "2025", "full2022_2025"]
            if case.book.contract.terminal
            else ["2026YTD"]
        )
        for name in names:
            row = report_period(case.book, name).model_dump(mode="json")
            row.update(labels)
            row["opening"], row["closing"] = row.pop("opening_equity"), row.pop("closing_equity")
            periods.append(row)
        selected = [r for r in case.book.daily if case.book.contract.terminal or r.day.year == 2026]
        previous = report_period(case.book, names[-1]).opening_equity
        for month in sorted({r.day.strftime("%Y-%m") for r in selected}):
            closing = [r.equity for r in selected if r.day.strftime("%Y-%m") == month][-1]
            months.append(
                {
                    **labels,
                    "period" if label == "score" else "month": month,
                    "opening": previous,
                    "closing": closing,
                    "total_return": closing / previous - 1,
                }
            )
            previous = closing
    columns = [label if c == "label" else c for c in METRIC_COLUMNS]
    frame = pd.DataFrame(periods)[columns]
    summary = frame[frame.period.isin(["full2022_2025", "2026YTD"])].sort_values(
        ["cohort", "cost_bps", "total_return"], ascending=[True, True, False]
    )
    return [
        Table(name="PERIOD_METRICS.csv", keys=("key", "period"), frame=frame),
        Table(name="SUMMARY.csv", keys=("key", "period"), frame=summary),
        Table(
            name="MONTHLY.csv",
            keys=("key", "period" if label == "score" else "month"),
            frame=pd.DataFrame(months),
        ),
    ]


def paired_table(periods: pd.DataFrame, study: str) -> Table:
    label = "score" if study == STUDIES[0] else "variant"
    new_values = ["calendar_13612w"] if label == "score" else list(FAMILY)
    baseline_values = ["original252"] if label == "score" else ["BASE252", "W13612"]
    rows = []
    for new in periods[periods[label].isin(new_values)].itertuples(index=False):
        for baseline in baseline_values:
            old_rows = periods[
                (periods[label] == baseline)
                & (periods.k == new.k)
                & (periods.cost_bps == new.cost_bps)
                & (periods.cohort == new.cohort)
                & (periods.period == new.period)
            ]
            if len(old_rows) != 1:
                raise ValueError("unique matched baseline required")
            old = next(old_rows.itertuples(index=False))
            row = {
                "new_key": new.key,
                "original_key": old.key,
                "k": new.k,
                "cost_bps": new.cost_bps,
                "cohort": new.cohort,
                "period": new.period,
                "new_return": new.total_return,
                "original_return": old.total_return,
                "return_difference": number(new.total_return) - number(old.total_return),
                "new_drawdown": new.max_drawdown,
                "original_drawdown": old.max_drawdown,
                "drawdown_difference": number(new.max_drawdown) - number(old.max_drawdown),
                "fees_difference": number(new.fees) - number(old.fees),
                "orders_difference": int(number(new.orders) - number(old.orders)),
                "turnover_difference": number(new.turnover) - number(old.turnover),
            }
            if label == "variant":
                row.update(variant=new.variant, baseline=baseline)
            rows.append(row)
    return Table(
        name="PAIRED_COMPARISON.csv",
        keys=("new_key", "original_key", "period"),
        frame=pd.DataFrame(rows),
    )


def inference_tables(cases: list[Case], periods: pd.DataFrame) -> list[Table]:
    records = []
    for cost in (5, 20):
        differences = excess_matrix(cases, cost)
        if differences.shape != (1003, 6):
            raise ValueError("fixed six-rule historical family required")
        for block in (5, 20, 60):
            print(
                f"Bootstrap {cost}bp block={block}, B=10000, six rules / 1003 sessions", flush=True
            )
            result = max_bootstrap(differences, FAMILY, block=block, repetitions=10000)
            for i, name in enumerate(FAMILY):
                records.append(
                    {
                        "variant": name,
                        "cost_bps": cost,
                        "block": block,
                        "repetitions": 10000,
                        "family_size": 6,
                        "annualized_arithmetic_excess": result.mean_bp[i] * 252 / 10000,
                        "pointwise_ci_low": result.ci_low_bp[i] * 252 / 10000,
                        "pointwise_ci_high": result.ci_high_bp[i] * 252 / 10000,
                        "adjusted_p": result.adjusted_p[i],
                        "family_p": result.family_p,
                        "inference_scope": "local six rules; pointwise arithmetic CI, "
                        "not CAGR or global-search CI",
                    }
                )
    statistics = pd.DataFrame(records)
    pareto = []
    for name in FAMILY:
        local = []
        for cost in (5, 20):
            sample = periods[(periods.cohort == "historical2022_2025") & (periods.cost_bps == cost)]
            new = sample[sample.variant == name].set_index("period")
            old = sample[sample.variant == "BASE252"].set_index("period")
            cagr = number(new.loc["full2022_2025", "cagr_252"]) - number(
                old.loc["full2022_2025", "cagr_252"]
            )
            dd = number(new.loc["full2022_2025", "max_drawdown"]) - number(
                old.loc["full2022_2025", "max_drawdown"]
            )
            years = [str(year) for year in range(2022, 2026)]
            pvalues = statistics[statistics.variant == name].adjusted_p
            local.append(
                {
                    "variant": name,
                    "cost_bps": cost,
                    "cagr_difference": cagr,
                    "drawdown_difference": dd,
                    "years_beating_baseline": int(
                        (
                            new.loc[years, "total_return"] > old.loc[years, "total_return"] + 1e-10
                        ).sum()
                    ),
                    "worst_year_difference": float(
                        new.loc[years, "total_return"].min() - old.loc[years, "total_return"].min()
                    ),
                    "pareto_at_cost": cagr >= -1e-10
                    and dd <= 1e-10
                    and (cagr > 1e-10 or dd < -1e-10),
                    "pareto_both_costs": False,
                    "local_statistical_support": len(pvalues) == 6 and bool((pvalues < 0.05).all()),
                }
            )
        for row in local:
            row["pareto_both_costs"] = all(r["pareto_at_cost"] for r in local)
        pareto.extend(local)
    return [
        Table(name="STATISTICS.csv", keys=("variant", "cost_bps", "block"), frame=statistics),
        Table(name="PARETO.csv", keys=("variant", "cost_bps"), frame=pd.DataFrame(pareto)),
    ]


def same_path_table(cases: list[Case]) -> Table:
    rows = []
    for case in cases:
        target = [(row.day.isoformat(), sorted(row.q.items())) for row in case.targets]
        nav = [(r.day.isoformat(), r.equity.hex()) for r in case.book.daily]
        rows.append(
            {
                "key": case.key,
                "variant": case.variant,
                "cost_bps": case.book.contract.cost_bps,
                "cohort": case.book.contract.cohort,
                "target_sha256": hashlib.sha256(
                    json.dumps(target, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest(),
                "nav_sha256": hashlib.sha256(json.dumps(nav).encode()).hexdigest(),
                "same_target_keys": "",
                "same_nav_keys": "",
            }
        )
    for row in rows:
        for kind in ("target", "nav"):
            row[f"same_{kind}_keys"] = "|".join(
                str(other["key"])
                for other in rows
                if other["key"] != row["key"]
                and other["cohort"] == row["cohort"]
                and other[f"{kind}_sha256"] == row[f"{kind}_sha256"]
            )
    return Table(name="SAME_PATH.csv", keys=("key",), frame=pd.DataFrame(rows))


def diagnostic_tables(cases: list[Case]) -> list[Table]:
    profits, weights, january = [], [], []
    for case in cases:
        if case.k is None or case.book.contract.cost_bps != 5:
            continue
        label = "calendar_13612w" if case.variant == "W13612" else "original252"
        labels = {"key": case.key, "score": label, "k": case.k, "cohort": case.book.contract.cohort}
        for target in case.targets:
            for symbol, value in target.q.items():
                weights.append(
                    {
                        **labels,
                        "decision_day": target.day.isoformat(),
                        "symbol": symbol,
                        "predeployment_weight": value,
                        "target_account_weight": value * 0.95,
                    }
                )
        periods = (
            [str(y) for y in range(2022, 2026)] if case.book.contract.terminal else ["2026YTD"]
        )
        for period in periods:
            stats = report_period(case.book, period)
            previous = [row.day for row in case.book.daily if row.day < stats.start]
            opening_day = previous[-1] if previous else None
            for symbol in NAMES:
                amount = fsum(
                    h.size * h.close
                    for h in case.holdings
                    if h.symbol == symbol and h.day == stats.end
                )
                amount -= fsum(
                    h.size * h.close
                    for h in case.holdings
                    if h.symbol == symbol and h.day == opening_day
                )
                amount -= fsum(
                    o.size * o.price + o.fee
                    for o in case.book.orders
                    if o.symbol == symbol and stats.start <= o.day <= stats.end
                )
                amount += fsum(
                    d.net_amount
                    for d in case.book.dividends
                    if d.symbol == symbol and stats.start <= d.ex_date <= stats.end
                )
                profits.append({**labels, "period": period, "symbol": symbol, "profit_usd": amount})
            total = fsum(number(row["profit_usd"]) for row in profits[-len(NAMES) :])
            if not isclose(total, stats.profit, rel_tol=1e-10, abs_tol=1e-5):
                raise ValueError("asset profit attribution does not reconcile")
        if case.book.contract.cohort == "reset2026":
            holdings = sorted(
                f"{h.symbol}:{h.size}"
                for h in case.holdings
                if h.day == date(2026, 1, 29) and h.size
            )
            before = next(r.equity for r in case.book.daily if r.day == date(2026, 1, 29))
            after = next(r.equity for r in case.book.daily if r.day == date(2026, 1, 30))
            january.append(
                {
                    "key": case.key,
                    "score": label,
                    "k": case.k,
                    "jan29_holdings": ";".join(holdings),
                    "jan30_return": after / before - 1,
                }
            )
    return [
        Table(
            name="ASSET_PROFIT_5BP.csv",
            keys=("key", "period", "symbol"),
            frame=pd.DataFrame(profits),
        ),
        Table(
            name="TARGET_WEIGHTS_5BP.csv",
            keys=("key", "decision_day", "symbol"),
            frame=pd.DataFrame(weights),
        ),
        Table(name="JAN30_CHECK_5BP.csv", keys=("key",), frame=pd.DataFrame(january)),
    ]


def number(value: object) -> float:
    return TypeAdapter(float).validate_python(value)


def fingerprints(root: Path) -> Fingerprints:
    paths = [
        *root.glob("src/**/*.py"),
        root / "standard/ACCOUNTS.json",
        root / "standard/STRATEGY.json",
        *root.glob("standard/reference/**/*"),
    ]
    for study in STUDIES:
        folder = root / "experiments" / study
        paths.extend(
            [folder / "ACCOUNTS.json", *folder.glob("*.csv"), *folder.glob("reference/**/*")]
        )
    return Fingerprints(
        hashes={
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
            if path.is_file()
        }
    )


@beartype
def reproduce(root: Path = ROOT, output: Path | None = None) -> Receipt:
    root = root.resolve()
    destination = output.resolve() if output else root / "experiments/reproduction_results"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output directory must be new or empty; use --output")
    before = fingerprints(root)
    cases, checks = load_cases(root)
    prepared: list[tuple[str, Table]] = []
    checked: list[TableCheck] = []
    for study, count in zip(STUDIES, (54, 66), strict=True):
        selected = study_cases(cases, study)
        if len(selected) != count or len({c.key for c in selected}) != count:
            raise ValueError("published study role coverage mismatch")
        tables = core_tables(selected, study)
        tables.append(paired_table(tables[0].frame, study))
        if study == STUDIES[0]:
            tables.extend(diagnostic_tables(selected))
        else:
            tables.extend(inference_tables(selected, tables[0].frame))
            tables.append(same_path_table(selected))
        for table in tables:
            cells = compare_table(table, root / "experiments" / study / table.name)
            checked.append(
                TableCheck(
                    study=study,
                    table=table.name,
                    rows=len(table.frame),
                    columns=len(table.frame.columns),
                    cells=cells,
                )
            )
            prepared.append((study, table))
    if fingerprints(root) != before:
        raise ValueError("input or source changed during reproduction")
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for study, table in prepared:
        (destination / study).mkdir(exist_ok=True)
        path = destination / study / table.name
        table.frame.to_csv(path, index=False)
        outputs[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if fingerprints(root) != before:
        raise ValueError("input or source changed while writing outputs")
    receipt = Receipt(
        checked_book_metric_values=checks,
        table_checks=checked,
        input_sha256=before.hashes,
        output_sha256=outputs,
    )
    (destination / "VERIFICATION.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new or empty output directory")
    args = parser.parse_args()
    result = reproduce(output=args.output)
    print(
        f"PASS:90 unique books,54/66 study roles,{len(result.table_checks)} CSVs,"
        "six B10000 bootstraps,native0"
    )


if __name__ == "__main__":
    main()
