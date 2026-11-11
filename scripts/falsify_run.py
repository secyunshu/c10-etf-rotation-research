"""Eight registered native runs, independent hard12 budget, saved-result revalidation."""

import argparse
import fcntl
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from beartype import beartype

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10share.verify import compare_frames, compare_json
from c10standard.models import BOOKS, Receipt
from c10standard.native import write
from c10standard.run import sha
from scripts import annual_native as native
from scripts import annual_run
from scripts.annual_data import load_prepared
from scripts.annual_models import Binding, Contract, Event, NativeComplete, Prepared
from scripts.annual_signals import verify_targets
from scripts.falsify_io import ROOT, STUDY, verify_lock
from scripts.falsify_models import (
    Account,
    Attempt,
    CaseBinding,
    CaseReceipt,
    Stage,
    baseline_schedules,
    catalog,
    load_cases,
)
from scripts.falsify_signals import case_scope


@beartype
def accounts(root: Path) -> tuple[Account, ...]:
    return catalog(load_cases(root), *baseline_schedules(root))


@beartype
def journal_path(root: Path = ROOT) -> Path:
    return root / STUDY / "evidence/attempts.jsonl"


@beartype
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
            starts = [r for r in previous if r.event == "started"]
            if len(starts) >= 12:
                raise ValueError("hard12 native-attempt budget exhausted, including failures")
            if any(r.account == account for r in starts):
                raise ValueError("account already attempted; switching output cannot retry it")
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@beartype
def binding(account: Account, root: Path) -> CaseBinding:
    if account not in accounts(root):
        raise ValueError("actual account differs from registered case definitions")
    return CaseBinding(account=account, source_sha256=verify_lock(root))


@beartype
def bridge_receipt(account: Account, receipt: Receipt, folder: Path, root: Path) -> CaseReceipt:
    differences: dict[str, float] = {}
    if account.stage == "bridge":
        reference = root / "experiments/v2-annual-repool/reference" / account.envelope.key
        fingerprints = Fingerprints.model_validate_json((reference / "REFERENCE.json").read_bytes())
        verify_hashes(reference, fingerprints.hashes)
        if (Contract.model_validate_json((reference / "CONTRACT.json").read_bytes())
                != account.envelope
                or Binding.model_validate_json((reference / "BINDING.json").read_bytes()).contract
                != account.envelope):
            raise ValueError("bridge reference has a different registered envelope")
        for name in BOOKS:
            filename = f"{name}.csv.gz"
            differences[filename] = compare_frames(
                pd.read_csv(reference / filename, float_precision="round_trip"),
                pd.read_csv(folder / filename, float_precision="round_trip"), filename)
        for name in ("targets", "timings", "native_returns", "open_trades", "metrics",
                     "native_total_return"):
            filename = f"{name}.json"
            differences[filename] = compare_json(json.loads((reference / filename).read_text()),
                                                 json.loads((folder / filename).read_text()),
                                                 filename)
    return CaseReceipt(account=account.key, native=receipt, bridge_differences=differences)


@beartype
def validate_saved(account: Account, prepared: Prepared, folder: Path,
                   root: Path = ROOT) -> CaseReceipt:
    expected = binding(account, root)
    if CaseBinding.model_validate_json((folder / "CASE_BINDING.json").read_bytes()) != expected:
        raise ValueError("saved case binding changed; cannot relabel the native envelope")
    with case_scope(account, expected.source_sha256):
        receipt = annual_run.validate_saved(account.envelope, prepared, folder, root)
    return bridge_receipt(account, receipt, folder, root)


@beartype
def require_bridges(output: Path, root: Path = ROOT) -> None:
    for account in accounts(root):
        if account.stage == "bridge":
            folder = output / account.key
            if not (folder / "VERIFIED.json").exists():
                raise ValueError("both bridges A5 and D20 must PASS before science")
            validate_saved(account, load_prepared(account.envelope.mode, root), folder, root)


@beartype
def execute(account: Account, prepared: Prepared, output: Path,
            root: Path = ROOT) -> CaseReceipt:
    folder = output / account.key
    if folder.exists():
        if (folder / "ARTIFACT.json").exists() and (folder / "NATIVE_COMPLETE.json").exists():
            return validate_saved(account, prepared, folder, root)
        raise ValueError(f"incomplete failed evidence preserved; no automatic retry: {folder}")
    frozen = binding(account, root)
    if account.stage == "science":
        require_bridges(output, root)
    with case_scope(account, frozen.source_sha256):
        verify_targets(account.envelope, prepared, native.signal_days(account.envelope))
    settings = native.settings_for(prepared, account.envelope)
    lock = journal_path(root).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        attempt = uuid4().hex
        journal_event(root, "started", account.key, attempt, folder)
        try:
            folder.mkdir(parents=True, exist_ok=False)
            write(folder / "CASE_BINDING.json", frozen)
            write(folder / "CONTRACT.json", account.envelope)
            write(folder / "BINDING.json", Binding(contract=account.envelope,
                                                  source_sha256=frozen.source_sha256))
            with case_scope(account, frozen.source_sha256):
                native.simulate(prepared, account.envelope, settings, folder)
            write(folder / "NATIVE_COMPLETE.json",
                  NativeComplete(artifact_sha256=sha(folder / "ARTIFACT.json")))
            journal_event(root, "native_completed", account.key, attempt, folder)
            receipt = validate_saved(account, prepared, folder, root)
        except BaseException as error:
            if folder.exists():
                with (folder / "FAILURE.txt").open("a", encoding="utf-8") as failure:
                    failure.write(traceback.format_exc().replace(str(root.resolve()), "<root>"))
            journal_event(root, "failed", account.key, attempt, folder, str(error))
            raise
        journal_event(root, "finished", account.key, attempt, folder)
    return receipt


@beartype
def run_stage(stage: Stage, output: Path, root: Path = ROOT, *, verify_only: bool = False) -> None:
    if stage == "science":
        require_bridges(output, root)
    for account in accounts(root):
        if account.stage == stage:
            prepared = load_prepared(account.envelope.mode, root)
            receipt = (validate_saved(account, prepared, output / account.key, root) if verify_only
                       else execute(account, prepared, output, root))
            print(receipt.model_dump_json())


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
