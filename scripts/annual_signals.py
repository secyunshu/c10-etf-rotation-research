"""Original V2 ranking restricted to the next execution session's annual pool."""

from datetime import date
from math import fsum, isfinite

import exchange_calendars as xcals
import pandas as pd
from beartype import beartype

from c10standard.audit import equal
from c10standard.models import TargetRecord
from rotbt.models import Dataset
from scripts.annual_models import Contract, DecisionTiming, Pool, Prepared, Schedule
from scripts.v2_2023_signals import features_for as features_for


def feed_names(contract: Contract, prepared: Prepared) -> tuple[str, ...]:
    Schedule.model_validate(prepared.schedule)
    if (contract.schedule_sha256 != prepared.schedule.sha256
            or contract.mode != prepared.schedule.mode):
        raise ValueError("contract does not bind prepared annual schedule")
    return tuple(sorted({s for p in prepared.schedule.entries for s in p.risk_names} | {"SHY"}))


def active_pool(contract: Contract, prepared: Prepared, day: date) -> Pool:
    feed_names(contract, prepared)
    execution = xcals.get_calendar("XNYS").next_session(pd.Timestamp(day)).date()
    if not contract.start <= execution < contract.through:
        raise ValueError("signal execution outside the registered 2022-2025 window")
    return next(pool for pool in prepared.schedule.entries if pool.year == execution.year)


def decision_timing(contract: Contract, prepared: Prepared, day: date) -> DecisionTiming:
    pool = active_pool(contract, prepared, day)
    execution = xcals.get_calendar("XNYS").next_session(pd.Timestamp(day)).date()
    # TargetRecord.day is the price bar label. Weekend selection is finalized later.
    selection = pool.selection_asof if execution == pool.execute_on else day
    return DecisionTiming(pool_year=pool.year, price_asof=day, selection_asof=selection,
                          source_asof=pool.source_asof, execute_on=execution,
                          pool_sha256=pool.sha256)


def group_for(pool: Pool, symbol: str) -> str:
    return next(group for group, members in pool.groups.groups.items() if symbol in members)


@beartype
def target(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
    pool = active_pool(contract, prepared, day)
    scores: dict[str, float] = {}
    for symbol in (*pool.risk_names, "SHY"):
        frame = prepared.features.frames[symbol].loc[:str(day)]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("missing exact prior signal close")
        if len(frame) < 253:
            if symbol == "SHY":
                raise ValueError("insufficient 253-price cash warmup")
            continue
        row = frame.iloc[-1]
        values = [float(row[k]) for k in ("mom252", "raw_close", "turnover20", "positive20")]
        if not all(isfinite(v) for v in values):
            raise ValueError("nonfinite signal input")
        if symbol == "SHY" or (values[1] >= 5 and values[2] >= 1e7 and values[3] == 20):
            scores[symbol] = values[0]
    selected: list[str] = []
    used: set[str] = set()
    for symbol in sorted((s for s in scores if s != "SHY"), key=lambda s: (-scores[s], s)):
        group = group_for(pool, symbol)
        if group in used:
            continue
        selected.append(symbol)
        used.add(group)
        if len(selected) == 2:
            break
    q = {s: .45 for s in selected if scores[s] > scores["SHY"]}
    q["SHY"] = max(0., 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


@beartype
def oracle(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
    pool = active_pool(contract, prepared, day)
    frames = {d.symbol: d.frame.loc[:str(day)] for d in prepared.datasets
              if d.symbol in {*pool.risk_names, "SHY"}}
    shy = frames["SHY"]
    if len(shy) < 253 or shy.index[-1].date() != day:
        raise ValueError("raw cash warmup/calendar requires 253 prices")
    cash = float(shy.tr.iloc[-1] / shy.tr.iloc[-253] - 1)
    if not isfinite(cash):
        raise ValueError("nonfinite raw cash return")
    scores: list[tuple[float, str, str]] = []
    for symbol in pool.risk_names:
        frame = frames[symbol]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("raw exact signal close missing")
        if len(frame) < 253:
            continue
        recent = frame.iloc[-20:]
        score = float(frame.tr.iloc[-1] / frame.tr.iloc[-253] - 1)
        if not isfinite(score):
            raise ValueError("nonfinite raw return")
        if (fsum(float(v) for v in recent.turnover) / 20 >= 1e7
                and all(float(v) > 0 for v in recent.volume) and float(frame.close.iloc[-1]) >= 5):
            scores.append((score, symbol, group_for(pool, symbol)))
    picks: list[tuple[float, str]] = []
    occupied: set[str] = set()
    for score, symbol, group in sorted(scores, key=lambda x: (-x[0], x[1])):
        if group in occupied:
            continue
        picks.append((score, symbol))
        occupied.add(group)
        if len(picks) == 2:
            break
    q = {symbol: .9 / 2 for score, symbol in picks if score > cash}
    q["SHY"] = max(0., 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


def verify_targets(contract: Contract, prepared: Prepared, days: tuple[date, ...]) -> None:
    for day in days:
        prefix = [Dataset(symbol=d.symbol, frame=d.frame.loc[:str(day)].copy(),
                          dividends=[e for e in d.dividends if e.ex_date <= day],
                          splits=[s for s in d.splits if s.day <= day]) for d in prepared.datasets]
        bounded = Prepared(schedule=prepared.schedule, datasets=prefix,
                           features=features_for(prefix), fees=prepared.fees)
        full = target(contract, prepared, day)
        for expected in (target(contract, bounded, day), oracle(contract, bounded, day)):
            if set(full.q) != set(expected.q):
                raise ValueError("full/prefix/raw target membership differs")
            for symbol, weight in full.q.items():
                equal(weight, expected.q[symbol], "full/prefix/raw target weight", 1e-12)
