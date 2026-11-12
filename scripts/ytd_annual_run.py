"""Eight registered accounts, independent hard12 journal, immutable native evidence."""

import argparse
import fcntl
import json
import os
import shutil
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import pandas as pd

from c10share.io import verify_hashes
from c10share.verify import compare_frames, compare_json
from c10standard.models import BOOKS, NativeScalar, Receipt
from c10standard.native import write
from c10standard.run import sha, write_json
from rotbt.analysis import analyze
from scripts import ytd_annual_native as native
from scripts.ytd_annual_io import ROOT, STUDY
from scripts.ytd_annual_models import (
    HISTORICAL_LAST,
    Artifact,
    Attempt,
    Binding,
    Contract,
    Event,
    NativeComplete,
    Prepared,
    Role,
    Stage,
)
from scripts.ytd_annual_signals import feed_names, verify_targets

EXPERIMENT = STUDY


def catalog(science_hash: str, bridge_hash: str) -> tuple[Contract, ...]:
    roles: tuple[Role, ...] = ("ANNUAL", "SPY_BH", "QQQ_BH")
    return tuple(Contract(role="ANNUAL", cost_bps=cost, stage="bridge",
                          schedule_sha256=bridge_hash) for cost in (5, 20)) + tuple(
        Contract(role=role, cost_bps=cost, stage="science", schedule_sha256=science_hash)
        for role in roles for cost in (5, 20))


def journal_path(root: Path = ROOT) -> Path:
    return root / EXPERIMENT / "evidence/attempts.jsonl"


def journal_event(root: Path, event: Event, account: str, attempt: str, output: Path,
                  error: str = "") -> None:
    path = journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = Attempt(event=event, account=account, attempt_id=attempt, utc=datetime.now(UTC),
                     output=os.path.relpath(output.resolve(), root.resolve()),
                     error=error.replace(str(root.resolve()), "<root>"))
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = [Attempt.model_validate_json(line) for line in handle.read().splitlines()]
        if event == "started":
            starts = [row for row in previous if row.event == "started"]
            if len(starts) >= 12:
                raise ValueError("hard12 native-attempt budget exhausted, including failures")
            if any(row.account == account for row in starts):
                raise ValueError("account already attempted; switching output cannot retry it")
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def execution_lock(root: Path) -> Iterator[None]:
    path = journal_path(root).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def binding(contract: Contract, root: Path) -> Binding:
    from scripts.ytd_annual_io import verify_lock

    return Binding(contract=contract, source_sha256=verify_lock(root))


def validate_saved(contract: Contract, prepared: Prepared, folder: Path,
                   root: Path = ROOT) -> Receipt:
    expected_binding = binding(contract, root)
    if Binding.model_validate_json((folder / "BINDING.json").read_text()) != expected_binding:
        raise ValueError("saved source/pool binding changed")
    artifact = Artifact.model_validate_json((folder / "ARTIFACT.json").read_text())
    if (artifact.contract != contract
            or Contract.model_validate_json((folder / "CONTRACT.json").read_text()) != contract):
        raise ValueError("saved native contract mismatch")
    complete = NativeComplete.model_validate_json((folder / "NATIVE_COMPLETE.json").read_text())
    if complete.artifact_sha256 != sha(folder / "ARTIFACT.json"):
        raise ValueError("saved account lacks its fault-free native completion proof")
    receipt_path = folder / "VERIFIED.json"
    prior = Receipt.model_validate_json(receipt_path.read_text()) if receipt_path.exists() else None
    if prior is not None:
        verify_hashes(folder, prior.hashes)
    native.audit(artifact, prepared)
    verify_targets(contract, prepared, native.signal_days(contract))
    with TemporaryDirectory(prefix="annual-2026-books-") as temporary:
        generated = Path(temporary)
        native.export_books(generated, artifact)
        for name in BOOKS:
            filename = f"{name}.csv.gz"
            if (folder / filename).exists():
                compare_frames(pd.read_csv(generated / filename, float_precision="round_trip"),
                               pd.read_csv(folder / filename, float_precision="round_trip"),
                               filename)
            else:
                shutil.copyfile(generated / filename, folder / filename)
    settings = native.settings_for(prepared, contract)
    outputs: tuple[tuple[str, Any], ...] = (
               ("targets.json", [r.model_dump(mode="json") for r in artifact.targets]),
               ("timings.json", [r.model_dump(mode="json") for r in artifact.decision_times]),
               ("native_returns.json", [r.model_dump(mode="json")
                                        for r in artifact.native_returns]),
               ("open_trades.json", [r.model_dump(mode="json") for r in artifact.open_trades]))
    if contract.terminal:
        metrics = [m.model_dump(mode="json") for m in analyze(artifact.account, settings)]
        outputs = (*outputs, ("metrics.json", metrics))
    for name, value in outputs:
        if (folder / name).exists():
            compare_json(json.loads((folder / name).read_text()), value, name)
        else:
            write_json(folder / name, value)
    scalar = NativeScalar(native_total_return=artifact.account.native_total_return)
    scalar_path = folder / "native_total_return.json"
    if scalar_path.exists():
        if NativeScalar.model_validate_json(scalar_path.read_text()) != scalar:
            raise ValueError("native return scalar changed")
    else:
        write(scalar_path, scalar)
    differences = compare_history(contract, artifact, folder, root)
    receipt = Receipt(account=contract.key, hashes={p.name: sha(p) for p in folder.iterdir()
                                                   if p.is_file() and p.name not in
                                                   {"VERIFIED.json", "FAILURE.txt"}},
                      max_differences=differences)
    if prior is not None:
        if prior != receipt:
            raise ValueError("saved receipt differs from revalidated evidence")
    else:
        write(receipt_path, receipt)
    return receipt


def historical_reference(contract: Contract, root: Path) -> Path:
    if contract.role == "ANNUAL":
        return (root / "experiments/v2-annual-repool/reference"
                / f"ANNUAL_{contract.cost_bps}bp_science")
    return (root / "standard/reference"
            / f"{contract.role}_{contract.cost_bps}bp_historical2022_2025")


def prefix_book(expected: pd.DataFrame, actual: pd.DataFrame, book: str) -> float:
    column = {"daily": "day", "orders": "day", "holdings": "day",
              "dividends": "ex_date", "trades": "closed"}[book]
    cutoff = str(HISTORICAL_LAST)
    left = expected.loc[expected[column] < cutoff].copy()
    right = actual.loc[actual[column] < cutoff].copy()
    if book == "holdings":
        extra = ~right.symbol.isin(left.symbol.unique())
        if right.loc[extra, "size"].ne(0).any():
            raise ValueError("future pool member has a nonzero historical holding")
        right = right.loc[~extra]
    if book == "trades" and left.empty and right.empty and left.columns.equals(right.columns):
        # A header-only CSV has no values from which to infer the column dtypes.
        return 0.0
    return compare_frames(left.reset_index(drop=True), right.reset_index(drop=True),
                          f"historical prefix {book}")


def compare_history(contract: Contract, artifact: Artifact, folder: Path,
                    root: Path) -> dict[str, float]:
    reference = historical_reference(contract, root)
    differences: dict[str, float] = {}
    for book in BOOKS:
        old = pd.read_csv(reference / f"{book}.csv.gz", float_precision="round_trip")
        new = pd.read_csv(folder / f"{book}.csv.gz", float_precision="round_trip")
        differences[book] = (compare_frames(old, new, book) if contract.terminal
                             else prefix_book(old, new, book))
    for name, records in (("targets", artifact.targets), ("timings", artifact.decision_times),
                          ("native_returns", artifact.native_returns)):
        if contract.role != "ANNUAL" and name != "targets":
            continue
        old_rows = json.loads((reference / f"{name}.json").read_text())
        new_rows = [row.model_dump(mode="json") for row in records]
        if not contract.terminal:
            field = "price_asof" if name == "timings" else "day"
            old_rows = [row for row in old_rows if row[field] < str(HISTORICAL_LAST)]
            new_rows = [row for row in new_rows if row[field] < str(HISTORICAL_LAST)]
        differences[name] = compare_json(old_rows, new_rows, name)
    if contract.terminal:
        for name in ("metrics", "native_total_return", "open_trades"):
            differences[name] = compare_json(json.loads((reference / f"{name}.json").read_text()),
                                             json.loads((folder / f"{name}.json").read_text()),
                                             name)
    return differences


def require_bridges(output: Path, root: Path = ROOT) -> None:
    from scripts.ytd_annual_data import load_prepared

    prepared = load_prepared("ANNUAL", "bridge", root)
    for cost in (5, 20):
        contract = Contract(role="ANNUAL", cost_bps=cost, stage="bridge",
                            schedule_sha256=prepared.schedule.sha256)
        folder = output / contract.key
        if not (folder / "VERIFIED.json").exists():
            raise ValueError("both old annual bridges must PASS before science")
        validate_saved(contract, prepared, folder, root)


def execute(contract: Contract, prepared: Prepared, output: Path,
            root: Path = ROOT) -> Receipt:
    folder = output / contract.key
    if folder.exists():
        if (folder / "ARTIFACT.json").exists() and (folder / "NATIVE_COMPLETE.json").exists():
            return validate_saved(contract, prepared, folder, root)
        raise ValueError(f"incomplete failed evidence preserved; no automatic retry: {folder}")
    feed_names(contract, prepared)
    frozen = binding(contract, root)
    if contract.stage == "science":
        require_bridges(output, root)
    verify_targets(contract, prepared, native.signal_days(contract))
    settings = native.settings_for(prepared, contract)
    with execution_lock(root):
        attempt = uuid4().hex
        journal_event(root, "started", contract.key, attempt, folder)
        try:
            folder.mkdir(parents=True, exist_ok=False)
            write(folder / "CONTRACT.json", contract)
            write(folder / "BINDING.json", frozen)
            native.simulate(prepared, contract, settings, folder)
            write(folder / "NATIVE_COMPLETE.json",
                  NativeComplete(artifact_sha256=sha(folder / "ARTIFACT.json")))
            journal_event(root, "native_completed", contract.key, attempt, folder)
            receipt = validate_saved(contract, prepared, folder, root)
        except BaseException as error:
            if folder.exists():
                with (folder / "FAILURE.txt").open("a", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc().replace(str(root.resolve()), "<root>"))
            journal_event(root, "failed", contract.key, attempt, folder, str(error))
            raise
        journal_event(root, "finished", contract.key, attempt, folder)
    return receipt


def run_stage(stage: Stage, output: Path, root: Path = ROOT, *, verify_only: bool = False) -> None:
    from scripts.ytd_annual_data import load_prepared

    bridge = load_prepared("ANNUAL", "bridge", root)
    science = load_prepared("ANNUAL", "science", root)
    if stage == "science":
        require_bridges(output, root)
    for contract in catalog(science.schedule.sha256, bridge.schedule.sha256):
        if contract.stage != stage:
            continue
        prepared = (bridge if stage == "bridge" else science if contract.role == "ANNUAL"
                    else load_prepared(contract.role, stage, root))
        if verify_only:
            validate_saved(contract, prepared, output / contract.key, root)
        else:
            execute(contract, prepared, output, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("bridge", "science"), required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run_stage(args.stage, args.output, args.root, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
