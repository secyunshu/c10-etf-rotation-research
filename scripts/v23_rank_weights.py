"""Only alter ordered Top2 weights; native Backtrader owns all account state."""
import argparse
import gzip
import json
from datetime import UTC, date, datetime
from math import fsum, isclose, isfinite
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd
from beartype import beartype
from pydantic import Field, TypeAdapter, model_validator

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import audit, native
from c10standard.io import ROOT, load_strategy, settings_for
from c10standard.models import Artifact, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Model
from rotbt.phase2.models import FeatureStore
from scripts import annual_io, ytd_annual_io, ytd_annual_report
from scripts.annual_data import load_prepared as load_history
from scripts.annual_models import Pool
from scripts.conditional_sleeve import CloseContract
from scripts.conditional_sleeve_ytd import ClosingBar, ResetContract, append_bar
from scripts.current_standard import YTD, select_target
from scripts.v23_band_trend import check_statistics
from scripts.ytd_annual_data import load_prepared as load_ytd
from scripts.ytd_annual_models import Pool as YtdPool
from scripts.ytd_annual_report import Statistics, summarize
from scripts.ytd_annual_signals import features_for

STUDY = ROOT / "experiments/rank-weights-20261119"
Weight = Literal["45_45", "50_40", "60_30"]
Window = Literal["HIST", "YTD2026"]
Cost = Literal[5, 20]
WEIGHTS: tuple[Weight, ...] = ("45_45", "50_40", "60_30")
WINDOWS: tuple[Window, ...] = ("HIST", "YTD2026")
COSTS: tuple[Cost, ...] = (5, 20)


class Candidate(Model):
    symbol: str
    group: str
    score: float = Field(allow_inf_nan=False)
    qualified: bool = True


class Candidates(Model):
    day: date
    rows: list[Candidate]


class OrderedPicks(Model):
    day: date
    first: str
    second: str

    @model_validator(mode="after")
    def valid(self) -> "OrderedPicks":
        if self.first == self.second or "SHY" in (self.first, self.second):
            raise ValueError("two distinct non-defender ranked ETFs required")
        return self


class Signals(Model):
    candidates: dict[Window, list[Candidates]]
    picks: dict[Window, list[OrderedPicks]]


class Context(Model):
    market: Prepared
    candidates: list[Candidates]
    picks: list[OrderedPicks]


class Row(Model):
    window: Window
    weight: Weight
    cost_bps: Cost
    stats: Statistics
    asset_profit: dict[str, float]


class Receipt(Model):
    window: Window
    weight: Weight
    cost_bps: Cost
    status: Literal["PASS", "FAIL"]
    reused: bool
    error: str = ""


@beartype
def select(values: Candidates) -> OrderedPicks:
    if len({r.symbol for r in values.rows}) != len(values.rows):
        raise ValueError("duplicate candidates")
    chosen: list[str] = []
    groups: set[str] = set()
    for r in sorted(values.rows, key=lambda r: (-r.score, r.symbol)):
        if r.qualified and r.group not in groups:
            chosen.append(r.symbol)
            groups.add(r.group)
        if len(chosen) == 2:
            return OrderedPicks(day=values.day, first=chosen[0], second=chosen[1])
    raise ValueError("fewer than two qualified groups")


@beartype
def allocation(picks: OrderedPicks, weight: Weight) -> TargetRecord:
    weights = {"45_45": (.45, .45), "50_40": (.5, .4), "60_30": (.6, .3)}
    a, b = weights[weight]
    sleeve = "SHY" if {"QQQ", "SPY"}.intersection((picks.first, picks.second)) else "QQQ"
    return TargetRecord(day=picks.day, q={picks.first: a, picks.second: b, sleeve: .1},
                        reason="oracle")


@beartype
def candidates(pool: Pool | YtdPool, features: FeatureStore, day: date) -> Candidates:
    groups = {s: g for g, names in pool.groups.groups.items() for s in names}
    rows = []
    for symbol in pool.risk_names:
        frame = features.frames[symbol].loc[:str(day)]
        if len(frame) < 253 or frame.index[-1].date() != day:
            raise ValueError(f"missing ranking prefix: {symbol} {day}")
        score = float(frame.tr.iloc[-1] / frame.tr.iloc[-253] - 1)
        row = frame.iloc[-1]
        close, turnover, positive = (float(row[k]) for k in
                                      ("raw_close", "turnover20", "positive20"))
        if not all(isfinite(v) for v in (score, close, turnover, positive)):
            raise ValueError("non-finite qualification")
        if not isclose(score, float(row.mom252), abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("raw total return and feature momentum disagree")
        rows.append(Candidate(symbol=symbol, group=groups[symbol], score=score,
                              qualified=close >= 5 and turnover >= 1e7 and positive == 20))
    values = Candidates(day=day, rows=rows)
    if allocation(select(values), "45_45").q != select_target(pool, features, day).q:
        raise ValueError("independent ranking differs from V2.3")
    return values


@beartype
def original(window: Window, cost: Cost) -> Artifact:
    if window == "HIST":
        folder = ROOT / f"experiments/v23-band-and-trend/reference/base252/{cost}bp"
        proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, proof.hashes)
        payload = gzip.decompress((folder / "ACCOUNT.json.gz").read_bytes())
        return Artifact.model_validate_json(payload)
    if cost != 5:
        raise ValueError("no frozen YTD20bp standard reference")
    folder = ROOT / YTD / "reference/C11"
    proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
    verify_hashes(folder, proof.hashes)
    return Artifact.model_validate_json((folder / "ARTIFACT.json").read_bytes())


@beartype
def contract_for(window: Window, cost: Cost) -> Contract:
    return CloseContract(cost_bps=cost) if window == "HIST" else ResetContract(cost_bps=cost)


@beartype
def context(window: Window) -> Context:
    data = load_history("ANNUAL") if window == "HIST" else load_ytd("ANNUAL", "science")
    contract = contract_for(window, 5)
    calendar = xcals.get_calendar("XNYS")
    days = (calendar.previous_session(pd.Timestamp(contract.start)).date(),
            *native.decision_days(contract))
    pools = {p.year: p for p in data.schedule.entries}
    values, picks = [], []
    for day in days:
        execution = calendar.next_session(pd.Timestamp(day)).date()
        pool = pools[execution.year]
        if pool.source_asof >= execution:
            raise ValueError("annual pool unavailable before execution")
        value = candidates(pool, data.features, day)
        prefix = FeatureStore(frames={s: data.features.frames[s].loc[:str(day)]
                                      for s in pool.risk_names})
        if value != candidates(pool, prefix, day):
            raise ValueError("future data altered earlier ranking")
        values.append(value)
        picks.append(select(value))
    archived = original(window, 5)
    if [(r.day, r.q) for r in archived.targets] != [(r.day, allocation(r, "45_45").q)
                                                  for r in picks]:
        raise ValueError("ranked baseline differs from frozen standard")
    datasets = data.datasets
    if window == "YTD2026":
        bar_path = ROOT / YTD / "NOV11_BARS.json"
        bars = TypeAdapter(list[ClosingBar]).validate_json(bar_path.read_bytes())
        required = {s for p in picks for s in allocation(p, "45_45").q} | {"SPY", "QQQ", "SHY"}
        if {b.symbol for b in bars} != required or len(bars) != len(required):
            raise ValueError("last-day marks do not cover exact required feeds")
        datasets = [append_bar(next(d for d in data.datasets if d.symbol == b.symbol), b)
                    for b in bars]
    prepared = Prepared(datasets=datasets, features=features_for(datasets),
        strategy=load_strategy(ROOT), first=data.first, last=contract.through,
        risk_names=tuple(sorted(d.symbol for d in datasets if d.symbol != "SHY")))
    return Context(market=prepared, candidates=values, picks=picks)


@beartype
def binding() -> Fingerprints:
    old = annual_io.verify_lock(ROOT), ytd_annual_io.verify_lock(ROOT)
    files = [Path(__file__), STUDY / "PLAN.md", STUDY / "REVIEW.md",
        STUDY / "tests/test_rank_weight_rules.py", STUDY / "tests/test_rank_weight_account.py",
        ROOT / "scripts/conditional_sleeve.py", ROOT / "scripts/conditional_sleeve_ytd.py",
        ROOT / "scripts/current_standard.py", ROOT / "requirements.lock",
        ROOT / YTD / "NOV11_BARS.json", ROOT / YTD / "NOV11_SOURCE.json"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    hashes[str(annual_io.STUDY / "SOURCE_LOCK.json")] = old[0]
    hashes[str(ytd_annual_io.STUDY / "SOURCE_LOCK.json")] = old[1]
    for window, cost in (("HIST", 5), ("HIST", 20), ("YTD2026", 5)):
        p = (ROOT / YTD / "reference/C11/VERIFIED.json" if window == "YTD2026" else
             ROOT / f"experiments/v23-band-and-trend/reference/base252/{cost}bp/VERIFIED.json")
        hashes[str(p.relative_to(ROOT))] = sha(p)
    return Fingerprints(hashes=hashes)


@beartype
def record_attempt(receipt: Receipt, event: str) -> None:
    with (STUDY / "raw/ATTEMPTS.jsonl").open("a") as f:
        f.write(json.dumps({"time": datetime.now(UTC).isoformat(), "window": receipt.window,
            "weight": receipt.weight, "cost_bps": receipt.cost_bps, "event": event,
            "error": receipt.error}) + "\n")


@beartype
def account(ctx: Context, window: Window, weight: Weight, cost: Cost,
            execute: bool) -> Artifact:
    contract = contract_for(window, cost)
    settings = settings_for(ctx.market.strategy, contract)
    settings.invested = 1.
    schedule = {p.day: allocation(p, weight) for p in ctx.picks}
    reused = weight == "45_45" and (window == "HIST" or cost == 5)
    receipt = Receipt(window=window, weight=weight, cost_bps=cost, status="PASS", reused=reused)
    folder = STUDY / f"reference/{window}/{weight}/{cost}bp"

    def target(actual: Contract, market: Prepared, day: date) -> TargetRecord:
        if actual != contract or market is not ctx.market:
            raise ValueError("wrong account callback identity")
        return schedule[day]

    if (folder / "ACCOUNT.json.gz").exists():
        proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, proof.hashes)
        payload = gzip.decompress((folder / "ACCOUNT.json.gz").read_bytes())
        artifact = Artifact.model_validate_json(payload)
    elif execute and reused:
        artifact = original(window, cost)
    elif execute:
        journal = STUDY / "raw/ATTEMPTS.jsonl"
        attempts = sum(json.loads(line)["event"] == "STARTED" for line in
                       journal.read_text().splitlines()) if journal.exists() else 0
        if attempts >= 11:
            raise ValueError("native attempt cap reached")
        folder.mkdir(parents=True)
        record_attempt(receipt, "STARTED")
        with patch.object(native, "target", target):
            artifact = native.simulate(ctx.market, contract, settings, folder)
        record_attempt(receipt, "NATIVE_COMPLETED")
    else:
        raise ValueError("missing completed native account")
    if artifact.contract.model_dump() != contract.model_dump():
        raise ValueError("stored account contract differs")
    artifact.contract = contract
    with patch.object(audit, "oracle", target):
        audit.audit(artifact, ctx.market, settings)
    if [(r.day, r.q) for r in artifact.targets] != [(r.day, r.q) for r in schedule.values()]:
        raise ValueError("native economic target trace differs")
    if reused:
        reference = original(window, cost)
        reference.contract = contract
        if artifact != reference:
            raise ValueError("reused standard account changed")
    if execute and not (folder / "ACCOUNT.json.gz").exists():
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "ACCOUNT.json.gz").open("xb") as f:
            f.write(gzip.compress(artifact.model_dump_json().encode(), mtime=0))
        native.write(folder / "VERIFIED.json", Fingerprints(hashes={
            "ACCOUNT.json.gz": sha(folder / "ACCOUNT.json.gz")}))
        native.write(folder / "RECEIPT.json", receipt)
    return artifact


@beartype
def describe(artifact: Artifact, window: Window, weight: Weight, cost: Cost) -> list[Row]:
    book = artifact.account
    periods = ("2022", "2023", "2024", "2025", "full") if window == "HIST" else ("full",)
    rows = []
    for period in periods:
        with patch.object(ytd_annual_report, "LAST", artifact.contract.through):
            stats = summarize(book.daily, book.orders, book.distributions, period)
        check_statistics(book, stats)
        profits: dict[str, float] = {}
        if period == "full":
            for o in book.orders:
                profits[o.symbol] = profits.get(o.symbol, 0.) - o.size * o.price - o.fee
            for d in book.distributions:
                profits[d.symbol] = profits.get(d.symbol, 0.) + d.net_amount
            for h in book.holdings:
                if h.day == book.daily[-1].day:
                    profits[h.symbol] = profits.get(h.symbol, 0.) + h.size * h.close
            if not isclose(fsum(profits.values()), book.daily[-1].equity - 50000., abs_tol=1e-6):
                raise ValueError("asset attribution does not explain net profit")
        if window == "YTD2026":
            stats = stats.model_copy(update={"period": "2026YTD", "cagr": None})
        rows.append(Row(window=window, weight=weight, cost_bps=cost, stats=stats,
                        asset_profit=profits))
    return rows


@beartype
def run(mode: Literal["prepare", "execute", "verify"]) -> None:
    bound = binding()
    if mode == "prepare":
        if (STUDY / "SIGNALS.json").exists():
            raise ValueError("signals already frozen")
        (STUDY / "raw").mkdir(exist_ok=True)
        contexts: dict[Window, Context] = {w: context(w) for w in WINDOWS}
        signals = Signals(candidates={w: c.candidates for w, c in contexts.items()},
                          picks={w: c.picks for w, c in contexts.items()})
        native.write(STUDY / "SOURCE.json", bound)
        native.write(STUDY / "SIGNALS.json", signals)
        print(f"Frozen {len(signals.picks['HIST'])} historical and "
              f"{len(signals.picks['YTD2026'])} YTD ranked targets", flush=True)
        return
    if Fingerprints.model_validate_json((STUDY / "SOURCE.json").read_bytes()) != bound:
        raise ValueError("source/input binding changed")
    signals = Signals.model_validate_json((STUDY / "SIGNALS.json").read_bytes())
    rows, receipts = [], []
    for window in WINDOWS:
        ctx = context(window)
        if ctx.picks != signals.picks[window] or ctx.candidates != signals.candidates[window]:
            raise ValueError("frozen causal signals changed")
        for weight in WEIGHTS:
            for cost in COSTS:
                reused = weight == "45_45" and (window == "HIST" or cost == 5)
                try:
                    artifact = account(ctx, window, weight, cost, mode == "execute")
                    rows.extend(describe(artifact, window, weight, cost))
                    receipt = Receipt(window=window, weight=weight, cost_bps=cost,
                                      status="PASS", reused=reused)
                    print(f"{window} {weight} {cost}bp PASS reused={reused}", flush=True)
                except Exception as error:
                    receipt = Receipt(window=window, weight=weight, cost_bps=cost,
                        status="FAIL", reused=reused, error=repr(error))
                    record_attempt(receipt, "FAIL")
                    print(f"{window} {weight} {cost}bp FAIL {error}", flush=True)
                receipts.append(receipt)
                if reused and receipt.status == "FAIL":
                    raise ValueError("standard bridge failed; no dependent accounts executed")
    for name, payload in [
        ("RESULTS.json", TypeAdapter(list[Row]).dump_json(rows, indent=2)),
        ("RECEIPTS.json", TypeAdapter(list[Receipt]).dump_json(receipts, indent=2)),
    ]:
        if mode == "execute":
            with (STUDY / name).open("xb") as f:
                f.write(payload + b"\n")
        elif (STUDY / name).read_bytes() != payload + b"\n":
            raise ValueError("recomputed metrics or receipts differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "verify"))
    run(parser.parse_args().mode)
