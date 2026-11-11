"""Recompute fresh2023 reports from saved native ledgers; never execute a broker."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import stdev
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from beartype import beartype
from pydantic import TypeAdapter

from c10experiments.report import Fingerprints, Table, compare_table
from c10share.io import verify_hashes
from c10share.report import read_csv
from c10share.verify import compare_json
from c10standard.io import ROOT
from c10standard.models import NativeScalar, Receipt, TargetRecord
from rotbt.analysis import analyze
from rotbt.models import (
    ClosedTrade,
    DailyRecord,
    Distribution,
    HoldingRecord,
    OrderRecord,
    Result,
    Spec,
)
from scripts.v2_2023_data import load_prepared
from scripts.v2_2023_io import STUDY, verify_lock
from scripts.v2_2023_models import Artifact, Binding, Contract
from scripts.v2_2023_native import audit, settings_for
from scripts.v2_2023_run import catalog


@beartype
def validate_period(contract: Contract, daily: list[DailyRecord]) -> None:
    if contract.stage != "science":
        raise ValueError("only fresh2023 science accounts enter the new comparison")
    days = [
        x.date()
        for x in xcals.get_calendar("XNYS").sessions_in_range(contract.start, contract.through)
    ]
    if [d.day for d in daily] != days:
        raise ValueError("fresh2023 calendar mismatch")
    if daily[-1].positions:
        raise ValueError("historical terminal account not flat")


def build(root: Path, native_output: Path, output: Path) -> None:
    source = verify_lock(root)
    markets = {role: load_prepared(role, root) for role in ("NEW_POOL", "OLD_POOL")}
    contracts = catalog(markets["NEW_POOL"].pool.sha256, markets["OLD_POOL"].pool.sha256)[2:]
    annual: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    initial_orders: list[dict[str, Any]] = []
    checks = 0
    for contract in contracts:
        prepared = markets["NEW_POOL" if contract.role == "NEW_POOL" else "OLD_POOL"]
        place = native_output / contract.key
        binding = Binding.model_validate_json((place / "BINDING.json").read_bytes())
        if binding.contract != contract or binding.source_sha256 != source:
            raise ValueError("report source or pool identity mismatch")
        if (place / "REFERENCE.json").exists():
            fp = Fingerprints.model_validate_json((place / "REFERENCE.json").read_bytes())
            if "BINDING.json" not in fp.hashes:
                raise ValueError("public reference lacks bound identity")
            verify_hashes(place, fp.hashes)
        else:
            native_receipt = Receipt.model_validate_json((place / "VERIFIED.json").read_bytes())
            verify_hashes(place, native_receipt.hashes)
        if Contract.model_validate_json((place / "CONTRACT.json").read_bytes()) != contract:
            raise ValueError("saved contract differs from requested role")
        daily = list(read_csv(place / "daily.csv.gz", DailyRecord))
        validate_period(contract, daily)
        names = {h.symbol for h in read_csv(place / "holdings.csv.gz", HoldingRecord)}
        result = Result(
            spec=Spec(name=contract.key, layer="hold", top=3, lookback=252),
            daily=daily,
            orders=list(read_csv(place / "orders.csv.gz", OrderRecord)),
            holdings=list(read_csv(place / "holdings.csv.gz", HoldingRecord)),
            distributions=list(read_csv(place / "dividends.csv.gz", Distribution)),
            trades=list(read_csv(place / "trades.csv.gz", ClosedTrade)),
            native_total_return=NativeScalar.model_validate_json(
                (place / "native_total_return.json").read_bytes()
            ).native_total_return,
            splits=[
                s
                for d in prepared.datasets
                if d.symbol in names
                for s in d.splits
                if contract.start <= s.day <= contract.through
            ],
        )
        # Artifact validates target, actualTimeReturn and open trade types independently.
        artifact = Artifact.model_validate(
            {
                "contract": contract,
                "account": result,
                "targets": json.loads((place / "targets.json").read_text()),
                "native_returns": json.loads((place / "native_returns.json").read_text()),
                "open_trades": json.loads((place / "open_trades.json").read_text()),
            }
        )
        audit(artifact, prepared)
        computed = analyze(result, settings_for(prepared, contract))
        compare_json(
            json.loads((place / "metrics.json").read_text()),
            [m.model_dump(mode="json") for m in computed],
            "savedmetrics",
        )
        checks += len(computed) * 11
        previous = 50_000.0
        changes = []
        for day in daily:
            changes.append((day.day, day.equity / previous - 1))
            previous = day.equity
        for metric in computed:
            if metric.period not in ("2023", "2024", "2025", "full"):
                continue
            sample = [
                v for d, v in changes if metric.period == "full" or d.year == int(metric.period)
            ]
            record = {
                "key": contract.key,
                "role": contract.role,
                "cost_bps": contract.cost_bps,
                "pool_sha256": contract.pool_sha256,
                **metric.model_dump(mode="json"),
                "annualized_volatility": stdev(sample) * 252**0.5,
            }
            (summary if metric.period == "full" else annual).append(record)
        for target in TypeAdapter(list[TargetRecord]).validate_json(
            (place / "targets.json").read_bytes()
        ):
            for symbol, weight in target.q.items():
                targets.append(
                    {
                        "key": contract.key,
                        "day": str(target.day),
                        "symbol": symbol,
                        "predeployment_weight": weight,
                        "account_weight": weight * 0.95,
                    }
                )
        initial_orders.extend(
            {"key": contract.key, **o.model_dump(mode="json")}
            for o in result.orders
            if o.day == contract.start
        )
    output.mkdir(parents=True, exist_ok=True)
    cell_count = 0
    records = (
        ("ANNUAL.csv", annual, ("key", "period")),
        ("SUMMARY.csv", summary, ("key", "period")),
        ("TARGETS.csv", targets, ("key", "day", "symbol")),
        ("INITIAL_ORDERS.csv", initial_orders, ("key", "day", "symbol")),
    )
    hashes = {}
    for name, rows, keys in records:
        table = Table(name=name, keys=keys, frame=pd.DataFrame(rows))
        file = output / name
        if not file.exists():
            table.frame.to_csv(file, index=False, float_format="%.17g")
        cell_count += compare_table(table, file)
        hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    receipt = {
        "status": "PASS",
        "accounts": len(contracts),
        "annual_rows": len(annual),
        "summary_rows": len(summary),
        "target_rows": len(targets),
        "checked_metric_values": checks,
        "checked_table_cells": cell_count,
        "source_lock": source,
        "native_attempts": 0,
        "table_hashes": hashes,
    }
    path = output / "REPORT_VERIFICATION.json"
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError("immutable report verification differs")
    else:
        path.write_text(json.dumps(receipt, indent=2) + "\n")
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
