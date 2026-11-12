"""Describe continuous native NAV through Nov10, with an explicit2026 capital base."""

import argparse
import hashlib
import json
from datetime import date
from math import fsum, isclose, isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from beartype import beartype
from pydantic import Field

from c10experiments.report import Fingerprints, Table, compare_table
from c10share.io import verify_hashes
from c10share.report import read_csv
from c10standard.models import NativeScalar, Receipt
from rotbt.models import (
    ClosedTrade,
    DailyRecord,
    Distribution,
    HoldingRecord,
    Model,
    OrderRecord,
    Result,
    Spec,
)
from scripts.ytd_annual_data import load_prepared
from scripts.ytd_annual_io import ROOT, STUDY, verify_lock
from scripts.ytd_annual_models import LAST, START, Artifact, Binding
from scripts.ytd_annual_native import audit


class Statistics(Model):
    period: str
    days: int
    first: date
    last: date
    opening_equity: float = Field(gt=0)
    closing_equity: float = Field(gt=0)
    total_return: float
    max_drawdown: float
    cagr: float | None
    sharpe_rf0: float
    annualized_volatility: float
    fees: float
    orders: int
    dividends: float
    turnover: float


@beartype
def summarize(
    daily: list[DailyRecord], orders: list[OrderRecord], dividends: list[Distribution], period: str
) -> Statistics:
    days = [d.day for d in daily]
    if not days or days != sorted(set(days)) or days[-1] > LAST:
        raise ValueError("invalid NAV calendar or cutoff")
    if any(not isfinite(d.equity) or d.equity <= 0 for d in daily):
        raise ValueError("invalid NAV")
    if period == "2026YTD":
        first, last = date(2026, 1, 2), LAST
    elif period == "full":
        first, last = START, LAST
    elif "-" in period:
        yy, mm = (int(x) for x in period.split("-"))
        month = pd.Period(f"{yy}-{mm:02d}", freq="M")
        first, last = month.start_time.date(), min(month.end_time.date(), LAST)
    else:
        yy = int(period)
        first, last = date(yy, 1, 1), min(date(yy, 12, 31), LAST)
    indices = [i for i, row in enumerate(daily) if first <= row.day <= last]
    if not indices:
        raise ValueError("empty report period")
    lo, hi = indices[0], indices[-1]
    if period == "2026YTD" and (lo == 0 or daily[lo - 1].day != date(2025, 12, 31)):
        raise ValueError("continuous YTD requires own December31 NAV")
    opening = daily[lo - 1].equity if lo else 50000.0
    selected = daily[lo : hi + 1]
    previous = [opening] + [x.equity for x in selected[:-1]]
    returns = [d.equity / prior - 1 for d, prior in zip(selected, previous, strict=True)]
    peak, drawdown = opening, 0.0
    for row in selected:
        peak = max(peak, row.equity)
        drawdown = max(drawdown, 1 - row.equity / peak)
    sigma = stdev(returns) if len(returns) > 1 else 0.0
    fills = [o for o in orders if first <= o.day <= last]
    growth = selected[-1].equity / opening
    return Statistics(
        period=period,
        days=len(selected),
        first=selected[0].day,
        last=selected[-1].day,
        opening_equity=opening,
        closing_equity=selected[-1].equity,
        total_return=growth - 1,
        max_drawdown=drawdown,
        cagr=growth ** (252 / len(selected)) - 1
        if period == "full" or (period.isdigit() and int(period) < 2026)
        else None,
        sharpe_rf0=mean(returns) / sigma * sqrt(252) if sigma else 0.0,
        annualized_volatility=sigma * sqrt(252),
        fees=fsum(o.fee for o in fills),
        orders=len(fills),
        dividends=fsum(d.net_amount for d in dividends if first <= d.ex_date <= last),
        turnover=fsum(abs(o.size * o.price) for o in fills) / mean(d.equity for d in selected),
    )


def build(root: Path, native_output: Path, output: Path) -> None:
    locked = verify_lock(root)
    annual: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    from scripts.ytd_annual_run import catalog

    annual_market = load_prepared("ANNUAL", "science", root)
    bridge_market = load_prepared("ANNUAL", "bridge", root)
    contracts = catalog(science_hash=annual_market.schedule.sha256,
                        bridge_hash=bridge_market.schedule.sha256)
    contracts = tuple(c for c in contracts if c.stage == "science")
    for contract in contracts:
        prepared = (
            annual_market
            if contract.role == "ANNUAL"
            else load_prepared(contract.role, "science", root)
        )
        folder = native_output / contract.key
        binding = Binding.model_validate_json((folder / "BINDING.json").read_bytes())
        if binding.contract != contract or binding.source_sha256 != locked:
            raise ValueError("report identity/source binding mismatch")
        if (folder / "REFERENCE.json").exists():
            fp = Fingerprints.model_validate_json((folder / "REFERENCE.json").read_bytes())
            if "BINDING.json" not in fp.hashes:
                raise ValueError("reference lacks identity")
            verify_hashes(folder, fp.hashes)
        else:
            verify_hashes(
                folder, Receipt.model_validate_json((folder / "VERIFIED.json").read_bytes()).hashes
            )
        daily = list(read_csv(folder / "daily.csv.gz", DailyRecord))
        expected = [d.date() for d in xcals.get_calendar("XNYS").sessions_in_range(START, LAST)]
        if [d.day for d in daily] != expected:
            raise ValueError("report requires full continuous account calendar")
        scalar = NativeScalar.model_validate_json(
            (folder / "native_total_return.json").read_bytes()
        )
        result = Result(
            spec=Spec(name=contract.key, layer="hold", top=3, lookback=252),
            daily=daily,
            orders=list(read_csv(folder / "orders.csv.gz", OrderRecord)),
            holdings=list(read_csv(folder / "holdings.csv.gz", HoldingRecord)),
            distributions=list(read_csv(folder / "dividends.csv.gz", Distribution)),
            trades=list(read_csv(folder / "trades.csv.gz", ClosedTrade)),
            native_total_return=scalar.native_total_return,
            splits=[s for d in prepared.datasets for s in d.splits if START <= s.day <= LAST],
        )
        artifact = Artifact.model_validate(
            {
                "contract": contract,
                "account": result,
                "targets": json.loads((folder / "targets.json").read_text()),
                "decision_times": json.loads((folder / "timings.json").read_text()),
                "native_returns": json.loads((folder / "native_returns.json").read_text()),
                "open_trades": json.loads((folder / "open_trades.json").read_text()),
            }
        )
        audit(artifact, prepared)
        for period in ("2022", "2023", "2024", "2025", "2026YTD", "full"):
            row = summarize(daily, result.orders, result.distributions, period)
            record = {
                "key": contract.key,
                "role": contract.role,
                "cost_bps": contract.cost_bps,
                **row.model_dump(mode="json"),
            }
            (summary if period in ("2026YTD", "full") else annual).append(record)
            if period == "2026YTD":
                annual.append(record)
        for month in sorted({d.day.strftime("%Y-%m") for d in daily}):
            row = summarize(daily, result.orders, result.distributions, month)
            monthly.append(
                {
                    "key": contract.key,
                    "role": contract.role,
                    "cost_bps": contract.cost_bps,
                    **row.model_dump(mode="json"),
                }
            )
        opening = next(d.equity for d in daily if d.day == date(2025, 12, 31))
        profit = 0.0
        for symbol in sorted(d.symbol for d in prepared.datasets):
            before = sum(
                h.size * h.close
                for h in result.holdings
                if h.day == date(2025, 12, 31) and h.symbol == symbol
            )
            current = [h for h in result.holdings if h.day == LAST and h.symbol == symbol]
            after = sum(h.size * h.close for h in current)
            flows = fsum(
                -o.size * o.price - o.fee
                for o in result.orders
                if o.day.year == 2026 and o.symbol == symbol
            )
            income = fsum(
                d.net_amount
                for d in result.distributions
                if d.ex_date.year == 2026 and d.symbol == symbol
            )
            value = after - before + flows + income
            profit += value
            if value:
                contributions.append(
                    {
                        "key": contract.key,
                        "symbol": symbol,
                        "profit_usd": value,
                        "return_contribution_pp": value / opening * 100,
                    }
                )
            if current and current[0].size:
                positions.append(
                    {
                        "key": contract.key,
                        "symbol": symbol,
                        "day": str(LAST),
                        "size": current[0].size,
                        "market_value": after,
                        "nav_weight": after / daily[-1].equity,
                    }
                )
        if not isclose(profit, daily[-1].equity - opening, abs_tol=1e-6, rel_tol=1e-10):
            raise ValueError("YTD cashflows and positions do not explain profit")
    definitions = (
        ("ANNUAL.csv", annual, ("key", "period")),
        ("SUMMARY.csv", summary, ("key", "period")),
        ("MONTHLY.csv", monthly, ("key", "period")),
        ("END_POSITIONS.csv", positions, ("key", "symbol")),
        ("YTD_CONTRIBUTIONS.csv", contributions, ("key", "symbol")),
    )
    output.mkdir(parents=True, exist_ok=True)
    cells, hashes = 0, {}
    for name, rows, keys in definitions:
        table = Table(name=name, keys=keys, frame=pd.DataFrame(rows))
        file = output / name
        if not file.exists():
            table.frame.to_csv(file, index=False, float_format="%.17g")
        cells += compare_table(table, file)
        hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    receipt = {
        "status": "PASS",
        "accounts": len(contracts),
        "through": str(LAST),
        "source_lock": locked,
        "checked_table_cells": cells,
        "native_attempts": 0,
        "tables": hashes,
    }
    file = output / "REPORT_VERIFICATION.json"
    if file.exists():
        if json.loads(file.read_text()) != receipt:
            raise ValueError("immutable report receipt differs")
    else:
        file.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(
        args.root,
        args.native_output or args.root / STUDY / "reference",
        args.output or args.root / STUDY / "results",
    )


if __name__ == "__main__":
    main()
