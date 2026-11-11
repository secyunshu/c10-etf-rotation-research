"""Ten preregistered native accounts, a separate hard16 journal, no automatic retries."""

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
from uuid import uuid4

import pandas as pd

from c10share.io import verify_hashes
from c10share.verify import compare_frames, compare_json
from c10standard.io import ROOT
from c10standard.models import BOOKS, NativeScalar, Receipt
from c10standard.models import Artifact as LegacyArtifact
from c10standard.models import Contract as LegacyContract
from c10standard.native import write
from c10standard.run import compare_reference, sha, write_json
from rotbt.analysis import analyze
from scripts import v2_2023_native as native
from scripts.v2_2023_models import (
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
from scripts.v2_2023_signals import feed_names, verify_targets

EXPERIMENT = Path("experiments/v2-2023-start")


def catalog(new_hash: str, old_hash: str) -> tuple[Contract, ...]:
    roles: tuple[Role, ...] = ("NEW_POOL", "OLD_POOL", "SPY_BH", "QQQ_BH")
    return tuple(Contract(role="OLD_POOL", cost_bps=cost, stage="bridge", pool_sha256=old_hash)
                 for cost in (5, 20)) + tuple(
        Contract(role=role, cost_bps=cost, stage="science",
                 pool_sha256=new_hash if role == "NEW_POOL" else old_hash)
        for cost in (5, 20) for role in roles)


def journal_path(root: Path = ROOT) -> Path:
    return root / EXPERIMENT / "evidence/attempts.jsonl"


def journal_event(root: Path, event: Event, account: str, attempt: str, output: Path,
                  error: str = "") -> None:
    path = journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = Attempt(event=event, account=account, attempt_id=attempt, utc=datetime.now(UTC),
                     output=str(output.resolve()), error=error)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = [Attempt.model_validate_json(line) for line in handle.read().splitlines()]
        if event == "started":
            starts = [row for row in previous if row.event == "started"]
            if len(starts) >= 16:
                raise ValueError("hard16 native-attempt budget exhausted, including failures")
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
    from scripts.v2_2023_io import verify_lock

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
    with TemporaryDirectory(prefix="v2-2023-books-") as temporary:
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
    metrics = [m.model_dump(mode="json") for m in analyze(artifact.account, settings)]
    outputs = (("metrics.json", metrics),
               ("targets.json", [r.model_dump(mode="json") for r in artifact.targets]),
               ("native_returns.json", [r.model_dump(mode="json")
                                        for r in artifact.native_returns]),
               ("open_trades.json", [r.model_dump(mode="json") for r in artifact.open_trades]))
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
    differences: dict[str, float] = {}
    if contract.stage == "bridge":
        legacy = LegacyContract(rule="C11", cost_bps=contract.cost_bps,
                                cohort="historical2022_2025")
        original = LegacyArtifact(contract=legacy, account=artifact.account,
                                  targets=artifact.targets, native_returns=artifact.native_returns,
                                  open_trades=artifact.open_trades)
        differences = compare_reference(legacy, original, folder, root)
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


def require_bridges(output: Path, root: Path = ROOT) -> None:
    from scripts.v2_2023_data import load_prepared

    prepared = load_prepared("OLD_POOL", root)
    for cost in (5, 20):
        contract = Contract(role="OLD_POOL", cost_bps=cost, stage="bridge",
                            pool_sha256=prepared.pool.sha256)
        folder = output / contract.key
        if not (folder / "VERIFIED.json").exists():
            raise ValueError("both old V2 bridges must PASS before science")
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
                    handle.write(traceback.format_exc())
            journal_event(root, "failed", contract.key, attempt, folder, str(error))
            raise
        journal_event(root, "finished", contract.key, attempt, folder)
    return receipt


def run_stage(stage: Stage, output: Path, root: Path = ROOT, *, verify_only: bool = False) -> None:
    from scripts.v2_2023_data import load_prepared

    old = load_prepared("OLD_POOL", root)
    new = load_prepared("NEW_POOL", root)
    if stage == "science":
        require_bridges(output, root)
    for contract in catalog(new.pool.sha256, old.pool.sha256):
        if contract.stage != stage:
            continue
        prepared = new if contract.role == "NEW_POOL" else old
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
