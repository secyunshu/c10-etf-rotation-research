"""Reproduce fixed V2 accounts offline; every native attempt shares a96-run budget."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import TypeAdapter

from c10share.io import verify_hashes
from c10share.verify import compare_frames, compare_json
from rotbt.analysis import analyze

from . import native
from .audit import audit, describe, equal
from .io import ROOT, load_accounts, load_market, load_strategy, reference_folder, settings_for
from .models import (
    BOOKS,
    COHORTS,
    RULES,
    Artifact,
    Attempt,
    Contract,
    NativeScalar,
    Prepared,
    Receipt,
    TargetRecord,
)


def catalog() -> tuple[Contract, ...]:
    return tuple(
        Contract(rule=rule, cost_bps=cost, cohort=cohort)
        for cohort in COHORTS
        for cost in (5, 20)
        for rule in RULES
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def journal_event(
    journal: Path,
    event: Literal["started", "native_completed", "finished", "failed", "revalidated"],
    account: str,
    attempt_id: str,
    output: Path,
    error: str = "",
) -> None:
    record = Attempt(
        standard_id="C10_HISTORICAL_GROUP_CAP_V2",
        event=event,
        account=account,
        attempt_id=attempt_id,
        utc=datetime.now(UTC),
        output=str(output),
        error=error,
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = [Attempt.model_validate_json(line) for line in handle.read().splitlines()]
        if event == "started" and sum(row.event == "started" for row in previous) >= 96:
            raise ValueError(
                "V2 shared96-native-attempt budget exhausted, including failed attempts"
            )
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def compare_reference(
    contract: Contract, artifact: Artifact, folder: Path, root: Path = ROOT
) -> dict[str, float]:
    accounts = [a for a in load_accounts(root) if a.contract == contract]
    if len(accounts) != 1:
        raise ValueError("unregistered V2 reference contract")
    reference = reference_folder(accounts[0], root)
    differences = {
        name: compare_frames(
            pd.read_csv(reference / f"{name}.csv.gz", float_precision="round_trip"),
            pd.read_csv(folder / f"{name}.csv.gz", float_precision="round_trip"),
            name,
        )
        for name in BOOKS
    }
    targets = TypeAdapter(list[TargetRecord]).validate_json(
        (reference / "targets.json").read_text()
    )
    differences["targets"] = compare_json(
        [t.model_dump(mode="json") for t in targets],
        [t.model_dump(mode="json") for t in artifact.targets],
        "targets",
    )
    saved = NativeScalar.model_validate_json((reference / "native_total_return.json").read_text())
    equal(
        artifact.account.native_total_return,
        saved.native_total_return,
        "reference actual native scalar",
        1e-10,
    )
    differences["native_total_return"] = abs(
        artifact.account.native_total_return - saved.native_total_return
    )
    name = "metrics.json" if contract.terminal else "ytd.json"
    differences[name] = compare_json(
        json.loads((reference / name).read_text()), json.loads((folder / name).read_text()), name
    )
    for name, actual in (
        ("native_returns.json", artifact.native_returns),
        ("open_trades.json", artifact.open_trades),
    ):
        if name in accounts[0].reference_files:
            differences[name] = compare_json(
                json.loads((reference / name).read_text()),
                [r.model_dump(mode="json") for r in actual],
                name,
            )
    return differences


def validate_saved(
    contract: Contract, prepared: Prepared, folder: Path, root: Path = ROOT
) -> Receipt:
    artifact = Artifact.model_validate_json((folder / "ARTIFACT.json").read_text())
    if (
        artifact.contract != contract
        or Contract.model_validate_json((folder / "CONTRACT.json").read_text()) != contract
    ):
        raise ValueError("saved V2 native contract mismatch")
    if json.loads((folder / "NATIVE_COMPLETE.json").read_text()) != {
        "artifact_sha256": sha(folder / "ARTIFACT.json")
    }:
        raise ValueError("artifact has no proof native execution completed without faults")
    settings = settings_for(prepared.strategy, contract)
    audit(artifact, prepared, settings)
    # Re-export to a temporary directory to verify or complete saved books, never rerun native.
    with TemporaryDirectory(prefix="c10-v2-books-") as temporary:
        generated = Path(temporary)
        native.export_books(generated, artifact)
        for name in BOOKS:
            filename = f"{name}.csv.gz"
            if (folder / filename).exists():
                compare_frames(
                    pd.read_csv(generated / filename, float_precision="round_trip"),
                    pd.read_csv(folder / filename, float_precision="round_trip"),
                    filename,
                )
            else:
                shutil.copyfile(generated / filename, folder / filename)
    name = "metrics.json" if contract.terminal else "ytd.json"
    metrics: Any = (
        [m.model_dump(mode="json") for m in analyze(artifact.account, settings)]
        if contract.terminal
        else describe(
            artifact.account, continuous=contract.cohort == "continuous2022_2026"
        ).model_dump(mode="json")
    )
    if not (folder / name).exists():
        write_json(folder / name, metrics)
    else:
        compare_json(json.loads((folder / name).read_text()), metrics, name)
    if not (folder / "targets.json").exists():
        write_json(folder / "targets.json", [r.model_dump(mode="json") for r in artifact.targets])
    else:
        compare_json(
            json.loads((folder / "targets.json").read_text()),
            [r.model_dump(mode="json") for r in artifact.targets],
            "saved targets",
        )
    scalar = NativeScalar(native_total_return=artifact.account.native_total_return)
    if not (folder / "native_total_return.json").exists():
        native.write(folder / "native_total_return.json", scalar)
    else:
        saved_scalar = NativeScalar.model_validate_json(
            (folder / "native_total_return.json").read_text()
        )
        equal(
            saved_scalar.native_total_return,
            artifact.account.native_total_return,
            "saved native total return",
            1e-10,
        )
    differences = compare_reference(contract, artifact, folder, root)
    path = folder / "VERIFIED.json"
    if path.exists():
        receipt = Receipt.model_validate_json(path.read_text())
        if receipt.account != contract.key or receipt.max_differences != differences:
            raise ValueError("saved verification receipt identity differs")
        verify_hashes(folder, receipt.hashes)
        return receipt
    receipt = Receipt(
        account=contract.key,
        hashes={
            p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != "FAILURE.txt"
        },
        max_differences=differences,
    )
    native.write(path, receipt)
    return receipt


def execute(
    contract: Contract, prepared: Prepared, output: Path, journal: Path, root: Path = ROOT
) -> Receipt:
    if contract not in catalog():
        raise ValueError("unregistered V2 account")
    folder = output / contract.key
    if folder.exists():
        if (folder / "ARTIFACT.json").exists() and (folder / "NATIVE_COMPLETE.json").exists():
            was_verified = (folder / "VERIFIED.json").exists()
            receipt = validate_saved(contract, prepared, folder, root)
            if not was_verified:
                journal_event(journal, "revalidated", contract.key, uuid4().hex, folder)
            return receipt
        raise ValueError(
            f"incomplete failed evidence preserved; no automatic native retry: {folder}"
        )
    if prepared.last != contract.through:
        raise ValueError("market physical cutoff differs from account cohort")
    # Validate reference/data contracts before spending a native attempt.
    account = next(a for a in load_accounts(root) if a.contract == contract)
    reference_folder(account, root)
    folder.mkdir(parents=True, exist_ok=False)
    native.write(folder / "CONTRACT.json", contract)
    settings = settings_for(prepared.strategy, contract)
    attempt = uuid4().hex
    journal_event(journal, "started", contract.key, attempt, folder)
    try:
        native.simulate(prepared, contract, settings, folder)
        write_json(
            folder / "NATIVE_COMPLETE.json", {"artifact_sha256": sha(folder / "ARTIFACT.json")}
        )
        journal_event(journal, "native_completed", contract.key, attempt, folder)
        receipt = validate_saved(contract, prepared, folder, root)
    except BaseException as error:
        with (folder / "FAILURE.txt").open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        journal_event(journal, "failed", contract.key, attempt, folder, str(error))
        raise
    journal_event(journal, "finished", contract.key, attempt, folder)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--journal", type=Path, help="shared96-attempt journal for all reproduction copies"
    )
    parser.add_argument("--all36", action="store_true")
    parser.add_argument(
        "--rule", choices=RULES, help="default: standard/STRATEGY.json default_rule"
    )
    parser.add_argument("--cost-bps", type=int, choices=(5, 20), default=5)
    parser.add_argument("--cohort", choices=COHORTS, default="historical2022_2025")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="revalidate saved outputs without native execution",
    )
    args = parser.parse_args()
    selected = (
        catalog()
        if args.all36
        else (
            Contract.model_validate(
                {
                    "rule": args.rule or load_strategy(args.root).default_rule,
                    "cost_bps": args.cost_bps,
                    "cohort": args.cohort,
                }
            ),
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.journal or args.output / "attempts.jsonl"
    with (args.output / ".native.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache: dict[str, Prepared] = {}
        for contract in selected:
            cutoff = "2025" if contract.terminal else "2026"
            if cutoff not in cache:
                cache[cutoff] = load_market(args.root, cutoff)
            if args.verify_only:
                receipt = validate_saved(
                    contract, cache[cutoff], args.output / contract.key, args.root
                )
            else:
                receipt = execute(contract, cache[cutoff], args.output, journal, args.root)
            print(f"{receipt.status} {contract.key}", flush=True)


if __name__ == "__main__":
    main()
