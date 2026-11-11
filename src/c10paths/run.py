"""Thin path-signal hooks around the standard native executor and financial audit."""

import argparse
import fcntl
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Literal, cast

import c10standard.audit as audit_module
from c10share.io import verify_hashes
from c10standard import native
from c10standard import run as standard
from c10standard.io import ROOT, load_market
from c10standard.models import BOOKS, Artifact, Contract, Prepared, Receipt, TargetRecord
from rotbt.models import Model

from . import signals
from .io import (
    EXPERIMENT,
    IDENTITY,
    Trial,
    catalog,
    check_binding,
    folder,
    journal_path,
    verify_lock,
    write_binding,
)

Event = Literal["started", "native_completed", "finished", "failed", "revalidated"]
Stage = Literal["verify", "bridge", "science"]
TargetFunction = Callable[[Contract, Prepared, date], TargetRecord]


class Attempt(Model):
    experiment: Literal["WECHAT_MOMENTUM_PATHS_V1"] = IDENTITY
    event: Event
    trial: Trial
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""


def journal_event(
    root: Path, journal: Path, event: Event, trial: Trial,
    attempt_id: str, output: Path, error: str = "",
) -> None:
    if journal.resolve() != journal_path(root).resolve():
        raise ValueError("all copies must use the experiment shared journal")
    record = Attempt(
        event=event, trial=trial, attempt_id=attempt_id, utc=datetime.now(UTC),
        output=str(output.resolve().relative_to(root.resolve())),
        error=error.replace(str(root.resolve()), "<root>"),
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = [Attempt.model_validate_json(line) for line in handle.read().splitlines()]
        if event == "started" and sum(row.event == "started" for row in previous) >= 28:
            raise ValueError("path experiment shared28-native-attempt budget exhausted")
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def scoped_hooks(trial: Trial, root: Path) -> Iterator[None]:
    old_target = cast(TargetFunction, native.__dict__["target"])
    old_oracle = cast(TargetFunction, audit_module.__dict__["oracle"])
    old_compare, old_journal = standard.compare_reference, standard.journal_event

    def selected_target(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
        if contract != trial.contract:
            raise ValueError("signal hook contract mismatch")
        if trial.variant == "BASE":
            return old_target(contract, prepared, day)
        return signals.target(trial.variant, prepared, day)

    def selected_oracle(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
        if contract != trial.contract:
            raise ValueError("oracle hook contract mismatch")
        if trial.variant == "BASE":
            return old_oracle(contract, prepared, day)
        return signals.oracle(trial.variant, prepared, day)

    def compare(
        contract: Contract, artifact: Artifact, folder: Path, root: Path = ROOT,
    ) -> dict[str, float]:
        # Empty means no preregistered science reference, never an exact self-comparison.
        return old_compare(contract, artifact, folder, root) if trial.variant == "BASE" else {}

    def record_event(
        journal: Path, event: Event, account: str, attempt_id: str, output: Path, error: str = "",
    ) -> None:
        if account != trial.contract.key:
            raise ValueError("journal account mismatch")
        if event == "started":
            write_binding(trial, output, root)
        journal_event(root, journal, event, trial, attempt_id, output, error)

    native.__dict__["target"], audit_module.__dict__["oracle"] = selected_target, selected_oracle
    standard.compare_reference, standard.journal_event = compare, record_event
    try:
        yield
    finally:
        native.__dict__["target"], audit_module.__dict__["oracle"] = old_target, old_oracle
        standard.compare_reference, standard.journal_event = old_compare, old_journal


@contextmanager
def execution_lock(root: Path) -> Iterator[None]:
    path = root / EXPERIMENT / "evidence/.native.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def require_bridges(output: Path, root: Path) -> None:
    for bridge in catalog()[:2]:
        place = folder(output, bridge)
        if not (place / "VERIFIED.json").exists():
            raise ValueError("both independently verified bridge accounts are required")
        check_binding(bridge, place, root)
        receipt = Receipt.model_validate_json((place / "VERIFIED.json").read_text())
        metric = "metrics.json" if bridge.contract.terminal else "ytd.json"
        expected = set(BOOKS) | {"targets", "native_total_return", metric}
        hashes = {f"{book}.csv.gz" for book in BOOKS} | {
            "BINDING.json", "ARTIFACT.json", "CONTRACT.json", "NATIVE_COMPLETE.json",
            "targets.json", "native_total_return.json", metric,
        }
        if (
            receipt.account != bridge.contract.key
            or not expected <= receipt.max_differences.keys()
            or not hashes <= receipt.hashes.keys()
            or any(not isfinite(v) or not 0 <= v <= 1e-10 for v in receipt.max_differences.values())
        ):
            raise ValueError("bridge reference verification missing")
        verify_hashes(place, receipt.hashes)


def execute(
    trial: Trial, prepared: Prepared, output: Path, journal: Path, root: Path = ROOT,
) -> Receipt:
    if trial not in catalog():
        raise ValueError("unregistered path trial")
    if journal.resolve() != journal_path(root).resolve():
        raise ValueError("all copies must use the experiment shared journal")
    with execution_lock(root):
        verify_lock(root)
        output.resolve().relative_to(root.resolve())
        if trial.variant != "BASE":
            require_bridges(output, root)
        place = folder(output, trial)
        if place.exists():
            check_binding(trial, place, root)
        elif journal.exists():
            records = [
                Attempt.model_validate_json(line) for line in journal.read_text().splitlines()
            ]
            prior = next((r for r in records if r.trial == trial and r.event == "started"), None)
            if prior is not None:
                raise ValueError(
                    "completed or failed native evidence preserved; "
                    f"reuse original output: {prior.output}"
                )
        with scoped_hooks(trial, root):
            return standard.execute(trial.contract, prepared, output / trial.variant, journal, root)


def verify_saved(
    trial: Trial, prepared: Prepared, output: Path, root: Path = ROOT,
) -> Receipt:
    if trial not in catalog():
        raise ValueError("unregistered path trial")
    with execution_lock(root):
        place = folder(output, trial)
        check_binding(trial, place, root)
        with scoped_hooks(trial, root):
            return standard.validate_saved(trial.contract, prepared, place, root)


def run_stage(stage: Stage, output: Path, root: Path = ROOT) -> None:
    verify_lock(root)
    cache: dict[str, Prepared] = {}

    def market(trial: Trial) -> Prepared:
        cutoff = "2025" if trial.contract.terminal else "2026"
        if cutoff not in cache:
            cache[cutoff] = load_market(root, cutoff)
        return cache[cutoff]

    if stage == "science":
        for trial in catalog()[:2]:
            verify_saved(trial, market(trial), output, root)
    selected = (
        catalog()[:2] if stage == "bridge"
        else catalog()[2:] if stage == "science" else catalog()
    )
    for trial in selected:
        if stage == "verify":
            receipt = verify_saved(trial, market(trial), output, root)
        else:
            receipt = execute(trial, market(trial), output, journal_path(root), root)
        print(f"{trial.key}: {receipt.status}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("verify", "bridge", "science"), default="verify")
    args = parser.parse_args()
    run_stage(args.stage, args.output or args.root / "runs/wechat-paths", args.root)


if __name__ == "__main__":
    main()
