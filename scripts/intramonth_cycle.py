"""Measure monthly patterns from frozen Backtrader ledgers and ETF histories."""

import csv
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal, cast

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard.io import ROOT
from c10standard.run import sha
from rotbt.models import Dataset, Model
from scripts import annual_io, ytd_annual_io
from scripts.annual_data import load_prepared as load_history
from scripts.annual_models import Pool
from scripts.ytd_annual_data import load_prepared as load_ytd
from scripts.ytd_annual_models import Pool as YtdPool

STUDY = Path("experiments/intramonth-cycle")
US_GROUPS = frozenset({
    "us_broad_style", "technology_semiconductors", "financials", "healthcare", "energy",
    "industrials", "utilities", "consumer_discretionary", "consumer_staples", "communications",
    "materials", "real_estate",
})
REFORM = pd.Period("2024-05", freq="M")
Metric = Literal["strategy_spy", "strategy_qqq", "loser_spy", "winner_spy", "far_near"]


class Contrast(Model):
    month: str
    metric: Metric
    sort: str
    window: str
    inside_bp: float
    outside_bp: float
    difference_bp: float


class Estimate(Model):
    metric: Metric
    sort: str
    window: str
    sample: str
    months: int
    inside_bp: float
    outside_bp: float
    difference_bp: float
    low95_bp: float
    high95_bp: float


class Groups(Model):
    month: str
    sort: str
    signal: date
    losers: list[str]
    winners: list[str]
    far: list[str]
    near: list[str]
    high_ratios: dict[str, float]


class Attribution(Model):
    sample: str
    window: str
    days: int
    strategy_log_pp: float
    spy_log_pp: float
    qqq_log_pp: float


@beartype
def labels(size: int, shift: int = 0) -> list[str]:
    if size < 16 or shift not in (0, 1):
        raise ValueError("require a full exchange month and shift 0 or 1")
    result = ["other"] * size
    result[-4:] = ["turn"] * 4
    result[size - 10 + shift:size - 4 + shift] = ["pretom"] * 6
    return result


@beartype
def log_returns(nav: pd.Series, opening: float) -> pd.Series:
    values = nav.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(opening) or opening <= 0:
        raise ValueError("require an opening balance and at least one net asset value")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("net asset values must be positive and finite")
    return pd.Series(np.diff(np.log(np.r_[opening, values])), index=nav.index)


@beartype
def split_close(data: Dataset, signal: pd.Timestamp) -> pd.Series:
    prices = data.frame.loc[:signal, "close"].astype(float).copy()
    for event in data.splits:
        if event.day <= signal.date():
            prices.loc[pd.Timestamp(event.day):] *= event.shares_after_per_before
    return prices


def ledger(study: str, last: str) -> pd.DataFrame:
    series = []
    for role, name in (("C11", "strategy"), ("SPY_BH", "spy"), ("QQQ_BH", "qqq")):
        folder = ROOT / "experiments" / study / "reference" / role
        proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, proof.hashes)
        nav = pd.read_csv(folder / "daily.csv.gz", index_col="day", parse_dates=True)["equity"]
        returns = log_returns(nav, 50_000.).loc[:last].rename(name)
        if not np.isclose(np.exp(returns.sum()), nav.loc[last] / 50_000., atol=1e-12):
            raise ValueError("return contributions do not reconcile to verified NAV")
        series.append(returns)
    frame = pd.concat(series, axis=1)
    expected = xcals.get_calendar("XNYS").sessions_in_range(frame.index[0], last)
    if frame.isna().any().any() or not frame.index.equals(expected):
        raise ValueError("ledgers must share the exact exchange calendar")
    return frame


def group_month(month: pd.Period, dates: pd.DatetimeIndex, pool: Pool | YtdPool,
                datasets: list[Dataset], sort: str) -> tuple[Groups, pd.DataFrame]:
    calendar = xcals.get_calendar("XNYS")
    signal = calendar.previous_session(dates[0])
    names = {s for g, symbols in pool.groups.groups.items() if g in US_GROUPS for s in symbols}
    datasets_by_symbol = {data.symbol: data for data in datasets}
    scores: list[tuple[float, str]] = []
    highs: dict[str, float] = {}
    returns: list[pd.Series] = []
    for symbol in sorted(names | {"SPY"}):
        data = datasets_by_symbol[symbol]
        prior = data.frame.loc[:signal, "tr"]
        if symbol in names:
            if sort == "skip_month":
                end = calendar.date_to_session((month - 2).end_time.normalize(),
                                               direction="previous")
                start = calendar.date_to_session((month - 13).end_time.normalize(),
                                                 direction="previous")
                score = float(prior.loc[end] / prior.loc[start])
            else:
                score = float(prior.iloc[-1] / prior.iloc[-253])
            scores.append((score, symbol))
            prices = split_close(data, signal).iloc[-252:]
            if len(prices) != 252 or prices.index[-1] != signal:
                raise ValueError("incomplete high-distance history")
            highs[symbol] = float(prices.iloc[-1] / prices.max())
        tr = data.frame.loc[signal:dates[-1], "tr"]
        returns.append(log_returns(tr.iloc[1:], float(tr.iloc[0])).rename(symbol))
    panel = pd.concat(returns, axis=1)
    if not panel.index.equals(dates) or panel.isna().any().any():
        raise ValueError("incomplete ETF outcome panel")
    size = len(scores) // 3
    ordered = [s for _, s in sorted(scores)]
    losers, winners = ordered[:size], ordered[-size:]
    ordered_high = sorted(losers, key=lambda s: (highs[s], s))
    half = size // 2
    if half < 2:
        raise ValueError("too few ETFs for a meaningful loser split")
    far, near = ordered_high[:half], ordered_high[-half:]
    record = Groups(month=str(month), sort=sort, signal=signal.date(), losers=losers,
                    winners=winners, far=far, near=near, high_ratios=highs)
    simple = panel.apply(np.expm1)
    loser_log = np.log1p(simple[losers].mean(axis=1))
    winner_log = np.log1p(simple[winners].mean(axis=1))
    far_log = np.log1p(simple[far].mean(axis=1))
    near_log = np.log1p(simple[near].mean(axis=1))
    output = pd.DataFrame({"loser_spy": loser_log - panel.SPY,
                           "winner_spy": winner_log - panel.SPY, "far_near": far_log - near_log})
    return record, output


def contrasts(month: str, series: pd.Series, metric: Metric, sort: str) -> list[Contrast]:
    size = len(series)
    fixed = np.array(labels(size)) == "pretom"
    shifted = np.array(labels(size, shift=int(pd.Period(month, freq="M") > REFORM))) == "pretom"
    placebo = (np.arange(size) >= 3) & (np.arange(size) < 9)
    if fixed.sum() != 6 or shifted.sum() != 6 or placebo.sum() != 6:
        raise ValueError("invalid six-day control")
    rows = []
    for name, mask in (("fixed", fixed), ("settlement_shift", shifted), ("early_placebo", placebo)):
        inside, outside = float(series[mask].mean() * 1e4), float(series[~mask].mean() * 1e4)
        rows.append(Contrast(month=month, metric=metric, sort=sort, window=name, inside_bp=inside,
                             outside_bp=outside, difference_bp=inside - outside))
    return rows


def summarize(rows: list[Contrast]) -> list[Estimate]:
    result = []
    keys: set[tuple[Metric, str, str]] = {(r.metric, r.sort, r.window) for r in rows}
    cases = sorted(keys)
    for metric, sort, window in cases:
        subset = [r for r in rows if (r.metric, r.sort, r.window) == (metric, sort, window)]
        periods = (
            ("2022–2025", [r for r in subset if r.month < "2026-01"]),
            ("T+2", [r for r in subset if r.month < "2024-05"]),
            ("T+1 through 2025", [r for r in subset if "2024-06" <= r.month < "2026-01"]),
            ("2026 Jan–Oct", [r for r in subset if r.month >= "2026-01"]),
        )
        for sample, months in periods:
            x = np.array([r.difference_bp for r in months])
            n = len(x)
            starts = np.random.default_rng(20261113).integers(0, n, (10_000, (n + 2) // 3))
            idx = ((starts[..., None] + np.arange(3)) % n).reshape(10_000, -1)[:, :n]
            lo, hi = np.quantile(x[idx].mean(axis=1), [.025, .975])
            result.append(Estimate(metric=metric, sort=sort, window=window, sample=sample, months=n,
                                   inside_bp=float(np.mean([r.inside_bp for r in months])),
                                   outside_bp=float(np.mean([r.outside_bp for r in months])),
                                   difference_bp=float(x.mean()), low95_bp=float(lo),
                                   high95_bp=float(hi)))
    return result


def write_table(path: Path, records: Sequence[Model]) -> None:
    rows = [row.model_dump(mode="json") for row in records]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    annual_io.verify_lock(ROOT)
    ytd_annual_io.verify_lock(ROOT)
    history, ytd = load_history("ANNUAL"), load_ytd("ANNUAL", "science")
    ledger_studies = ("annual-qqq-spy-sleeve", "conditional-sleeve-2026")
    checks = [ROOT / "experiments" / s / "reference" / k / "VERIFIED.json"
              for s in ledger_studies for k in ("C11", "SPY_BH", "QQQ_BH")]
    sources = [ROOT / annual_io.STUDY / "SOURCE_LOCK.json", ROOT / "scripts/intramonth_cycle.py",
               ROOT / ytd_annual_io.STUDY / "SOURCE_LOCK.json", ROOT / STUDY / "PLAN.md", *checks]
    output = ROOT / STUDY
    (output / "SOURCE.json").write_text(Fingerprints(
        hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    ).model_dump_json(indent=2) + "\n")
    observations, groups, attribution = [], [], []
    for study, market, last, sample in (
        (ledger_studies[0], history, "2025-12-31", "2022–2025"),
        (ledger_studies[1], ytd, "2026-10-30", "2026 Jan–Oct"),
    ):
        frame = ledger(study, last)
        pools = {p.year: p for p in market.schedule.entries}
        all_labels: list[str] = []
        dates = pd.DatetimeIndex(frame.index)
        month_index = dates.to_period("M")
        for month in cast(Sequence[pd.Period], month_index.unique()):
            block = frame.loc[month_index == month]
            expected = xcals.get_calendar("XNYS").sessions_in_range(
                month.start_time, month.end_time.normalize())
            if not block.index.equals(expected):
                raise ValueError(f"partial month: {month}")
            all_labels.extend(labels(len(block)))
            observations.extend(contrasts(str(month), block.strategy - block.spy,
                                          "strategy_spy", "actual"))
            observations.extend(contrasts(str(month), block.strategy - block.qqq,
                                          "strategy_qqq", "actual"))
            for sort in ("skip_month", "252"):
                group, stats = group_month(month, pd.DatetimeIndex(block.index), pools[month.year],
                                           market.datasets, sort)
                groups.append(group)
                metrics: tuple[Metric, ...] = ("loser_spy", "winner_spy", "far_near")
                for metric in metrics:
                    observations.extend(contrasts(str(month), stats[metric], metric, sort))
        for name in ("pretom", "turn", "other"):
            bucket = frame.loc[np.array(all_labels) == name]
            sums = bucket.sum() * 100
            attribution.append(Attribution(sample=sample, window=name, days=len(bucket),
                                           strategy_log_pp=float(sums.strategy),
                                           spy_log_pp=float(sums.spy), qqq_log_pp=float(sums.qqq)))
    estimates = summarize(observations)
    main = [e for e in estimates if e.metric == "strategy_spy" and e.window == "fixed"]
    whole, t2, t1, _ = main
    placebo = next(e for e in estimates if e.metric == "strategy_spy"
                   and e.window == "early_placebo" and e.sample == "2022–2025")
    gate = (whole.low95_bp * whole.high95_bp > 0 and t1.difference_bp * t2.difference_bp > 0
            and not (whole.difference_bp * placebo.difference_bp > 0
                     and abs(placebo.difference_bp) >= abs(whole.difference_bp)))
    write_table(output / "MONTHLY.csv", observations)
    write_table(output / "ESTIMATES.csv", estimates)
    write_table(output / "ATTRIBUTION.csv", attribution)
    (output / "GROUPS.json").write_text(json.dumps([g.model_dump(mode="json") for g in groups],
                                                  indent=2) + "\n")
    gate_result = {"timing_backtest_admitted": gate, "complete_months": len(groups) // 2}
    (output / "GATE.json").write_text(json.dumps(gate_result, indent=2) + "\n")


if __name__ == "__main__":
    run()
