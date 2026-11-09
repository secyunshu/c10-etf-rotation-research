"""Run twelve fixed accounts on the unchanged native Backtrader broker."""

import argparse
import fcntl
import json
import traceback
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, override
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd

from rotbt.analysis import analyze
from rotbt.models import Model, Result
from rotbt.phase2 import runner
from rotbt.phase2.strategy import Phase2Rotation

from .actions import HistoricalRotation
from .io import ROOT, feed_names, load_accounts, load_market, load_task, weights_for
from .models import AttemptRecord, Prepared, TargetRecord, Task, Verified
from .verify import compare_run


def simulate(prepared: Prepared, task: Task) -> tuple[Result, list[TargetRecord]]:
    trace: list[TargetRecord] = []

    class PublicRotation(HistoricalRotation):
        @override
        def __init__(self) -> None:
            Phase2Rotation.__init__(self)
            if self.p2 is None:
                raise ValueError("missing fixed public rule")
            self.public_rule = self.p2.label
            feed_names(self.public_rule)
            self.decision_days = {
                stamp.date()
                for stamp in xcals.get_calendar("XNYS").sessions_in_range(
                    self.start_day, self.end_day
                )
                if stamp.date() < self.end_day
                and not self.public_rule.endswith("_BH")
                and self.scheduled(stamp.date(), "M", ())
            }

        @override
        def _normal_close(self, day: date) -> bool:
            return day in self.decision_days

        @override
        def _decide(self, day: date, reason: str, *, force: bool = False) -> None:
            q = weights_for(self.public_rule, prepared, day)
            trace.append(TargetRecord.model_validate({"day": day, "q": q, "reason": reason}))
            self._stage(
                {s: float(w) * self.settings.invested for s, w in q.items()},
                day,
                reason,
                force=force,
            )

    # The proven runner reads its strategy and universe at call time. Restore both
    # even on failure; serial invocation is intentional because these are module globals.
    module: Any = runner
    previous, old_names = module.Phase2Rotation, module.EXPANDED
    names = feed_names(task.spec.label)
    by_name = {data.symbol: data for data in prepared.datasets}
    module.Phase2Rotation, module.EXPANDED = PublicRotation, names
    try:
        simulation = runner.run_phase2(
            [by_name[name] for name in names],
            prepared.features,
            task.spec,
            task.settings,
            task.start,
            task.end,
        )
        if simulation.faults:
            raise ValueError("native simulation returned faults")
        return simulation.account, trace
    finally:
        module.Phase2Rotation, module.EXPANDED = previous, old_names


def export_result(result: Result, targets: list[TargetRecord], task: Task, folder: Path) -> None:
    metrics = analyze(result, task.settings)
    tables: dict[str, list[Model]] = {
        "daily": list(result.daily),
        "orders": list(result.orders),
        "holdings": list(result.holdings),
        "dividends": list(result.distributions),
        "trades": list(result.trades),
    }
    for name, rows in tables.items():
        pd.DataFrame([row.model_dump() for row in rows]).to_csv(
            folder / f"{name}.csv.gz", index=False, compression="gzip"
        )
    full = next(metric for metric in metrics if metric.period == "full")
    verified = Verified(
        status="PASS",
        native_return=result.native_total_return,
        equity_return=full.total_return,
        days=len(result.daily),
        orders=len(result.orders),
        cash_events=len(result.distributions),
        min_available=min(row.available for row in result.daily),
        final_locked=result.daily[-1].locked,
        terminal_positions=result.daily[-1].positions,
    )
    payloads = {
        "metrics": [row.model_dump(mode="json") for row in metrics],
        "targets": [row.model_dump(mode="json") for row in targets],
        "task": task.model_dump(mode="json"),
        "verified": verified.model_dump(mode="json"),
    }
    for name, payload in payloads.items():
        (folder / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def journal_event(
    journal: Path, event: str, account: str, attempt_id: str, output: Path, error: str = ""
) -> None:
    """Count every start, including failures, under an exclusive append lock."""
    record = AttemptRecord.model_validate(
        {
            "event": event,
            "account": account,
            "attempt_id": attempt_id,
            "utc": datetime.now(UTC),
            "output": str(output),
            "error": error,
        }
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        history = [AttemptRecord.model_validate_json(line) for line in stream if line.strip()]
        if event == "started" and sum(row.event == "started" for row in history) >= 48:
            raise ValueError("48 native attempt budget exhausted")
        stream.seek(0, 2)
        stream.write(record.model_dump_json() + "\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/reproduction"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--accounts", nargs="+", help="optional subset of the twelve fixed names")
    parser.add_argument(
        "--journal", type=Path, help="shared attempt journal; default: output/attempts.jsonl"
    )
    args = parser.parse_args()
    accounts = load_accounts(args.root)
    if args.accounts:
        if len(set(args.accounts)) != len(args.accounts) or not set(args.accounts) <= {
            a.name for a in accounts
        }:
            parser.error("--accounts must be unique fixed account names")
        accounts = [account for account in accounts if account.name in args.accounts]
    if args.output.exists():
        raise ValueError(
            "output already exists; preserve prior evidence and choose a new directory"
        )
    prepared = load_market(args.root)
    tasks = {account.name: load_task(account, args.root) for account in accounts}
    args.output.mkdir(parents=True)
    journal = args.journal or args.output / "attempts.jsonl"
    comparisons = []
    for account in accounts:
        folder = args.output / account.name
        folder.mkdir()
        attempt_id = uuid4().hex
        journal_event(journal, "started", account.name, attempt_id, folder)
        try:
            task = tasks[account.name]
            result, trace = simulate(prepared, task)
            export_result(result, trace, task, folder)
            comparisons.append(compare_run(account, folder, args.root).model_dump(mode="json"))
        except Exception as exc:
            (folder / "failure.txt").write_text(traceback.format_exc())
            journal_event(journal, "failed", account.name, attempt_id, folder, str(exc))
            raise
        journal_event(journal, "finished", account.name, attempt_id, folder)
        print(f"PASS {account.name}", flush=True)
    (args.output / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "native_attempts": len(comparisons),
                "comparison": str(args.output / "comparison.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
