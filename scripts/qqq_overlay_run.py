"""Offline frozen QQQ overlay; default verification never starts a native account."""

import argparse
import fcntl
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Final, Literal, cast

from beartype import beartype
from pydantic import ConfigDict, model_validator

import c10standard.audit as audit_module
from c10share.io import verify_hashes
from c10standard import native
from c10standard import run as standard
from c10standard.io import ROOT, load_market
from c10standard.models import BOOKS, Artifact, Contract, Prepared, Receipt, Strategy, TargetRecord
from c10standard.native import write
from c10standard.run import sha
from rotbt.models import Model, Settings
from scripts import qqq_overlay_signals as signals

EXPERIMENT = Path("experiments/qqq-overlay")
IDENTITY: Final = "QQQ_OVERLAY_V1"
Event = Literal["started", "native_completed", "finished", "failed", "revalidated"]
Stage = Literal["verify", "bridge", "science"]
TargetFunction = Callable[[Contract, Prepared, date], TargetRecord]
SettingsFunction = Callable[[Strategy, Contract], Settings]


class Trial(Model):
    model_config = ConfigDict(frozen=True)
    variant: Literal["BASE", "OVERLAY"]
    contract: Contract

    @model_validator(mode="after")
    def historical_c11(self) -> "Trial":
        if self.contract.rule != "C11" or not self.contract.terminal:
            raise ValueError("overlay trials require historical C11")
        return self

    @property
    def key(self) -> str:
        return f"{self.variant}_{self.contract.cost}bp_{self.contract.cohort}"


class SourceLock(Model):
    experiment: Literal["QQQ_OVERLAY_V1"] = IDENTITY
    hashes: dict[str, str]


class Binding(Model):
    experiment: Literal["QQQ_OVERLAY_V1"] = IDENTITY
    trial: Trial
    source_lock_sha256: str


class Attempt(Model):
    experiment: Literal["QQQ_OVERLAY_V1"] = IDENTITY
    event: Event
    trial: Trial
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""


@beartype
def catalog() -> tuple[Trial, ...]:
    return tuple(Trial(variant=v, contract=Contract(cost_bps=c))
                 for v in ("BASE", "OVERLAY") for c in (5, 20))


@beartype
def folder(output: Path, trial: Trial) -> Path:
    return output / trial.variant / trial.contract.key


@beartype
def journal_path(root: Path = ROOT) -> Path:
    return root / EXPERIMENT / "evidence/attempts.jsonl"


@beartype
def source_snapshot(root: Path) -> SourceLock:
    paths = {root / name for name in ("pyproject.toml", "standard/STRATEGY.json",
             "standard/ACCOUNTS.json", str(EXPERIMENT / "PLAN.md"),
             str(EXPERIMENT / "PLAN_REVIEW.json"))}
    for directory in ("src", "tests", "typings", str(EXPERIMENT / "tests")):
        paths.update(p for p in (root / directory).rglob("*") if p.suffix in (".py", ".pyi"))
    paths.update((root / "scripts").glob("qqq_overlay_*.py"))
    directories = [root / "standard/data/2025"]
    directories += [root / "standard/reference" / f"{r}_{c}bp_historical2022_2025"
                    for r in ("C11", "SPY_BH", "QQQ_BH") for c in (5, 20)]
    for historical in directories:
        if not historical.is_dir():
            raise ValueError(f"missing historical source directory: {historical}")
        paths.update(p for p in historical.rglob("*") if p.is_file())
    return SourceLock(hashes={str(p.relative_to(root)): sha(p) for p in sorted(paths)})


@beartype
def freeze(root: Path = ROOT) -> str:
    path = root / EXPERIMENT / "SOURCE_LOCK.json"
    if path.exists():
        return verify_lock(root)
    if journal_path(root).exists() and journal_path(root).stat().st_size:
        raise ValueError("cannot freeze after native attempts")
    write(path, source_snapshot(root))
    return sha(path)


@beartype
def verify_lock(root: Path = ROOT) -> str:
    path = root / EXPERIMENT / "SOURCE_LOCK.json"
    if (not path.exists()
            or SourceLock.model_validate_json(path.read_text()) != source_snapshot(root)):
        raise ValueError("source lock differs from source, plan, inputs or historical references")
    return sha(path)


@beartype
def write_binding(trial: Trial, place: Path, root: Path) -> None:
    write(place / "BINDING.json", Binding(trial=trial, source_lock_sha256=verify_lock(root)))


@beartype
def check_binding(trial: Trial, place: Path, root: Path) -> None:
    path = place / "BINDING.json"
    expected = Binding(trial=trial, source_lock_sha256=verify_lock(root))
    if not path.exists() or Binding.model_validate_json(path.read_text()) != expected:
        raise ValueError("saved trial binding differs from variant, contract or source lock")


@beartype
def journal_event(root: Path, journal: Path, event: Event, trial: Trial,
                  attempt_id: str, output: Path, error: str = "") -> None:
    if journal.resolve() != journal_path(root).resolve():
        raise ValueError("all copies must use the experiment shared journal")
    record = Attempt(event=event, trial=trial, attempt_id=attempt_id, utc=datetime.now(UTC),
                     output=str(output.resolve().relative_to(root.resolve())),
                     error=error.replace(str(root.resolve()), "<root>"))
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = [Attempt.model_validate_json(line) for line in handle.read().splitlines()]
        if event == "started" and sum(row.event == "started" for row in previous) >= 8:
            raise ValueError("overlay shared8-native-attempt budget exhausted, including failures")
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
@beartype
def scoped_hooks(trial: Trial, root: Path) -> Iterator[None]:
    old_target = cast(TargetFunction, native.__dict__["target"])
    old_oracle = cast(TargetFunction, audit_module.__dict__["oracle"])
    old_compare, old_journal = standard.compare_reference, standard.journal_event
    old_settings = cast(SettingsFunction, standard.__dict__["settings_for"])

    @beartype
    def selected_target(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
        if contract != trial.contract:
            raise ValueError("signal hook contract mismatch")
        return (old_target(contract, prepared, day) if trial.variant == "BASE"
                else signals.target(prepared, day))

    @beartype
    def selected_oracle(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
        if contract != trial.contract:
            raise ValueError("oracle hook contract mismatch")
        return (old_oracle(contract, prepared, day) if trial.variant == "BASE"
                else signals.oracle(prepared, day))

    @beartype
    def selected_settings(strategy: Strategy, contract: Contract) -> Settings:
        if contract != trial.contract:
            raise ValueError("settings hook contract mismatch")
        settings = old_settings(strategy, contract)
        return (settings if trial.variant == "BASE"
                else settings.model_copy(update={"invested": 1.0}))

    @beartype
    def compare(contract: Contract, artifact: Artifact, folder: Path,
                root: Path = ROOT) -> dict[str, float]:
        # Mandatory existing framework callback; an empty mapping means no science reference.
        return old_compare(contract, artifact, folder, root) if trial.variant == "BASE" else {}

    @beartype
    def record_event(journal: Path, event: Event, account: str, attempt_id: str,
                     output: Path, error: str = "") -> None:
        if account != trial.contract.key:
            raise ValueError("journal account mismatch")
        if event == "started":
            write_binding(trial, output, root)
        journal_event(root, journal, event, trial, attempt_id, output, error)

    native.__dict__["target"], audit_module.__dict__["oracle"] = selected_target, selected_oracle
    standard.compare_reference, standard.journal_event = compare, record_event
    standard.__dict__["settings_for"] = selected_settings
    try:
        yield
    finally:
        native.__dict__["target"], audit_module.__dict__["oracle"] = old_target, old_oracle
        standard.compare_reference, standard.journal_event = old_compare, old_journal
        standard.__dict__["settings_for"] = old_settings


@contextmanager
@beartype
def execution_lock(root: Path) -> Iterator[None]:
    path = root / EXPERIMENT / "evidence/.native.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@beartype
def require_bridges(output: Path, root: Path) -> None:
    for bridge in catalog()[:2]:
        place = folder(output, bridge)
        if not (place / "VERIFIED.json").exists():
            raise ValueError("both independently verified bridge accounts are required")
        check_binding(bridge, place, root)
        receipt = Receipt.model_validate_json((place / "VERIFIED.json").read_text())
        expected = set(BOOKS) | {"targets", "native_total_return", "metrics.json"}
        hashes = {f"{book}.csv.gz" for book in BOOKS} | {
            "BINDING.json", "ARTIFACT.json", "CONTRACT.json", "NATIVE_COMPLETE.json",
            "targets.json", "native_total_return.json", "metrics.json"}
        if (receipt.account != bridge.contract.key or not expected <= receipt.max_differences.keys()
                or not hashes <= receipt.hashes.keys()
                or any(not isfinite(v) or not 0 <= v <= 1e-10
                       for v in receipt.max_differences.values())):
            raise ValueError("bridge reference verification missing")
        verify_hashes(place, receipt.hashes)


@beartype
def execute(trial: Trial, prepared: Prepared, output: Path, journal: Path,
            root: Path = ROOT) -> Receipt:
    if trial not in catalog():
        raise ValueError("unregistered overlay trial")
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
            records = [Attempt.model_validate_json(line)
                       for line in journal.read_text().splitlines()]
            if any(row.trial == trial and row.event == "started" for row in records):
                raise ValueError(
                    "completed or failed native evidence preserved; reuse original output")
        with scoped_hooks(trial, root):
            return standard.execute(trial.contract, prepared, output / trial.variant, journal, root)


@beartype
def verify_saved(trial: Trial, prepared: Prepared, output: Path, root: Path = ROOT) -> Receipt:
    with execution_lock(root):
        place = folder(output, trial)
        check_binding(trial, place, root)
        with scoped_hooks(trial, root):
            return standard.validate_saved(trial.contract, prepared, place, root)


@beartype
def run_stage(stage: Stage, output: Path, root: Path = ROOT) -> None:
    verify_lock(root)
    prepared = load_market(root, cutoff="2025")
    if stage == "science":
        for trial in catalog()[:2]:
            verify_saved(trial, prepared, output, root)
    selected = (catalog()[:2] if stage == "bridge"
                else catalog()[2:] if stage == "science" else catalog())
    for trial in selected:
        receipt = (verify_saved(trial, prepared, output, root) if stage == "verify"
                   else execute(trial, prepared, output, journal_path(root), root))
        print(f"{trial.key}: {receipt.status}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("verify", "bridge", "science"), default="verify")
    args = parser.parse_args()
    run_stage(args.stage, args.output or args.root / "runs/qqq-overlay", args.root)


if __name__ == "__main__":
    main()
