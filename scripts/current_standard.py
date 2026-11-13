"""Verify or reproduce the current annual Top 2 conditional-sleeve policy offline."""

import argparse
import csv
import json
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd
from pydantic import Field, TypeAdapter

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import audit as financial
from c10standard import native
from c10standard.io import ROOT, load_strategy, settings_for
from c10standard.models import Artifact, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Model
from rotbt.phase2.models import FeatureStore
from scripts import annual_io, ytd_annual_io, ytd_annual_report
from scripts.annual_data import load_prepared as load_history
from scripts.annual_models import Pool
from scripts.conditional_sleeve import RULES, CloseContract, allocation
from scripts.conditional_sleeve_ytd import ClosingBar, ResetContract, append_bar
from scripts.ytd_annual_data import load_prepared as load_ytd
from scripts.ytd_annual_models import Pool as YtdPool

RELEASE = Path("standard/v2_3")
HISTORY = Path("experiments/annual-qqq-spy-sleeve")
YTD = Path("experiments/conditional-sleeve-2026")


class Policy(Model):
    policy_version: Literal["2.3.0"]
    id: Literal["ANNUAL_TOP2_CONDITIONAL_SLEEVE"]
    adopted_date: date
    status: Literal["RESEARCH_AND_MANUAL_PAPER_STANDARD"]
    universe_refresh: Literal["annual"]
    rebalance_frequency: Literal["monthly"]
    momentum_lookback_trading_days: Literal[252]
    aum_min_usd: Literal[5000000000]
    adv20_min_usd: Literal[10000000]
    top_k: Literal[2]
    risk_group_cap: Literal[1]
    shy_momentum_gate: Literal[False]
    risk_slot_weight: float = Field(ge=.45, le=.45)
    sleeve_weight: float = Field(ge=.10, le=.10)
    sleeve_trigger: tuple[Literal["QQQ"], Literal["SPY"]]
    sleeve_if_triggered: Literal["SHY"]
    sleeve_otherwise: Literal["QQQ"]
    initial_cash: Literal[50000]
    reference_cost_bps: Literal[5]
    signal_timing: Literal["month_end_close"]
    execution_timing: Literal["next_session_open"]
    rebalance_same_names: Literal[True]
    history_schedule: Literal["experiments/v2-annual-repool/ANNUAL.json"]
    ytd_schedule: Literal["experiments/annual-2026/SCHEDULE.json"]


class Pointer(Model):
    policy_version: Literal["2.3.0"]
    strategy_path: Literal["standard/v2_3/STRATEGY.json"]
    strategy_sha256: str
    entrypoint: Literal["scripts.current_standard"]
    prior_policy: Literal["2.2.0"]


def load_policy(root: Path) -> Policy:
    pointer = Pointer.model_validate_json((root / "standard/CURRENT.json").read_bytes())
    strategy = root / pointer.strategy_path
    if sha(strategy) != pointer.strategy_sha256:
        raise ValueError("current policy hash mismatch")
    return Policy.model_validate_json(strategy.read_bytes())


def select_target(pool: Pool | YtdPool, features: FeatureStore, day: date) -> TargetRecord:
    groups = {symbol: group for group, names in pool.groups.groups.items() for symbol in names}
    ranked: list[tuple[float, str]] = []
    for symbol in pool.risk_names:
        prefix = features.frames[symbol].loc[:day.isoformat()]
        if len(prefix) < 253 or prefix.index[-1].date() != day:
            raise ValueError(f"incomplete ranking data: {symbol} at {day}")
        row = prefix.iloc[-1]
        score, close, adv, positive = (
            float(row[k]) for k in ("mom252", "raw_close", "turnover20", "positive20")
        )
        if not all(isfinite(v) for v in (score, close, adv, positive)):
            raise ValueError(f"invalid ranking data: {symbol} at {day}")
        if close >= 5 and adv >= 1e7 and positive == 20:
            ranked.append((-score, symbol))
    picks: list[str] = []
    for _, symbol in sorted(ranked):
        if not any(groups[symbol] == groups[s] for s in picks):
            picks.append(symbol)
            if len(picks) == 2:
                break
    return TargetRecord(day=day, q=allocation(tuple(picks)), reason="oracle")


def verify_release(root: Path) -> None:
    annual_io.verify_lock(root)
    ytd_annual_io.verify_lock(root)
    proof = Fingerprints.model_validate_json((root / RELEASE / "RELEASE_LOCK.json").read_bytes())
    for name, expected in proof.hashes.items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or sha(file) != expected:
            raise ValueError(f"release file hash mismatch: {name}")


def verify_report(root: Path, study: Path, anchor: Contract, results: list[Artifact]) -> None:
    with patch.object(ytd_annual_report, "LAST", anchor.through):
        rows = []
        for artifact in results:
            book = artifact.account
            periods = ("2022", "2023", "2024", "2025", "full") if study == HISTORY else ("full",)
            for period in periods:
                stats = ytd_annual_report.summarize(book.daily, book.orders, book.distributions,
                                                    period)
                rows.append({"strategy": artifact.contract.rule, **stats.model_dump(mode="json")})
    if study == HISTORY:
        expected = [{k: str(v) for k, v in row.items()} for row in rows]
        with (root / HISTORY / "RESULTS.csv").open(newline="", encoding="utf-8") as handle:
            if list(csv.DictReader(handle)) != expected:
                raise ValueError("historical report differs from the verified daily ledger")
    else:
        summary = {row["strategy"]: {k: v for k, v in row.items()
                                     if k not in {"strategy", "cagr", "period"}} for row in rows}
        if json.loads((root / YTD / "RESULTS.json").read_text()) != summary:
            raise ValueError("2026 report differs from the verified daily ledger")


def run_current(root: Path, output: Path, *, execute: bool = False) -> None:
    policy = load_policy(root)
    verify_release(root)
    history, ytd = load_history("ANNUAL", root), load_ytd("ANNUAL", "science", root)
    strategy = load_strategy(root)
    calendar = xcals.get_calendar("XNYS")
    for study, data, anchor in ((HISTORY, history, CloseContract()), (YTD, ytd, ResetContract())):
        pools = {p.year: p for p in data.schedule.entries}
        initial = calendar.previous_session(pd.Timestamp(anchor.start)).date()
        signals: dict[date, TargetRecord] = {}
        for day in (initial, *native.decision_days(anchor)):
            next_open = calendar.next_session(pd.Timestamp(day)).date()
            pool = pools[next_open.year]
            if pool.source_asof >= next_open:
                raise ValueError("annual pool was not known before execution")
            signals[day] = select_target(pool, data.features, day)
        datasets = data.datasets
        if study == YTD:
            bars = TypeAdapter(list[ClosingBar]).validate_json(
                (root / YTD / "NOV11_BARS.json").read_bytes()
            )
            required = {s for target in signals.values() for s in target.q} | {"SPY", "QQQ", "SHY"}
            if len(bars) != len(required) or {bar.symbol for bar in bars} != required:
                raise ValueError("supplemental marks must cover the actual execution feeds")
            datasets = [append_bar(next(d for d in datasets if d.symbol == bar.symbol), bar)
                        for bar in bars]
        market = Prepared(datasets=datasets, features=data.features, strategy=strategy,
                          risk_names=tuple(d.symbol for d in datasets if d.symbol != "SHY"),
                          first=data.first, last=anchor.through)

        def target(
            contract: Contract, prepared: Prepared, day: date, *,
            market: Prepared = market, through: date = anchor.through,
            signals: dict[date, TargetRecord] = signals, initial: date = initial,
        ) -> TargetRecord:
            if prepared is not market or contract.through != through:
                raise ValueError("target called with an unregistered market")
            if contract.rule == "C11":
                return signals[day]
            if day != initial or contract.rule not in RULES:
                raise ValueError("unexpected benchmark decision")
            return TargetRecord(day=day, q={contract.rule.removesuffix("_BH"): 1.}, reason="oracle")

        results: list[Artifact] = []
        for rule in RULES:
            contract = anchor.model_copy(update={"rule": rule})
            settings = settings_for(strategy, contract)
            settings.invested = 1.
            reference = root / study / "reference" / rule
            proof = Fingerprints.model_validate_json((reference / "VERIFIED.json").read_bytes())
            verify_hashes(reference, proof.hashes)
            original = Artifact.model_validate_json((reference / "ARTIFACT.json").read_bytes())
            if original.contract.model_dump() != contract.model_dump():
                raise ValueError("reference contract identity differs")
            saved = output / study.name / rule
            artifact = original
            if execute:
                path = saved / "ARTIFACT.json"
                if path.exists():
                    artifact = Artifact.model_validate_json(path.read_bytes())
                else:
                    saved.mkdir(parents=True)
                    with patch.object(native, "target", target):
                        artifact = native.simulate(market, contract, settings, saved)
                if Artifact.model_validate_json(artifact.model_dump_json()) != original:
                    raise ValueError("native account differs from the published reference")
            artifact.contract = contract
            with patch.object(financial, "oracle", target):
                financial.audit(artifact, market, settings)
            due = {calendar.next_session(pd.Timestamp(r.day)).date() for r in artifact.targets}
            orders = artifact.account.orders
            if (not orders or orders[0].day != contract.start
                    or any(o.day not in due for o in orders)
                    or (rule.endswith("_BH") and len(orders) != 1)):
                raise ValueError("orders differ from the registered next-open schedule")
            results.append(artifact)
            print(study.name, rule, "PASS", flush=True)
        verify_report(root, study, anchor, results)
    print(f"Current policy {policy.policy_version}: six Backtrader accounts verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/current-standard-v2-3"))
    parser.add_argument("--execute", action="store_true", help="reproduce six accounts")
    args = parser.parse_args()
    root = args.root.resolve()
    run_current(root, root / args.output, execute=args.execute)


if __name__ == "__main__":
    main()
