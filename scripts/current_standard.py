"""Current V2.2 default: verify annual Top2; --execute reproduces four native accounts."""

import argparse
import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from beartype import beartype
from pydantic import ConfigDict, Field

from c10share.io import verify_hashes
from c10share.verify import compare_frames, compare_json
from c10standard.io import ROOT
from c10standard.models import BOOKS
from c10standard.run import sha
from rotbt.models import Model
from scripts import annual_run
from scripts.annual_io import verify_lock as verify_annual_lock
from scripts.annual_models import Event, Schedule
from scripts.annual_report import build as build_report

CURRENT = Path("standard/current")


class Policy(Model):
    model_config = ConfigDict(extra="ignore", frozen=True)
    policy_version: Literal["2.2.0"]
    id: Literal["V2_TOP2_ANNUAL_POOL"]
    default_rule: Literal["C11"]
    top_k: Literal[2]
    universe_refresh: Literal["annual"]
    rebalance_frequency: Literal["monthly"]
    momentum_lookback_trading_days: Literal[252]
    aum_min_usd: Literal[5000000000]
    adv20_min_usd: Literal[10000000]
    risk_group_cap: Literal[1]
    defender: Literal["SHY"]
    risk_slot_weight: float = Field(ge=0.4275, le=0.4275)
    minimum_shy_weight: float = Field(ge=0.095, le=0.095)
    cash_target: float = Field(ge=0.05, le=0.05)
    initial_cash: Literal[50000]
    costs_bps: tuple[Literal[5], Literal[20]]
    schedule_path: Literal["experiments/v2-annual-repool/ANNUAL.json"]
    schedule_sha256: str
    schedule_identity_sha256: str
    verified_trade_years: tuple[Literal[2022], Literal[2023], Literal[2024], Literal[2025]]
    new_oos_validation: Literal[False]
    annual_2026_verified: Literal[False]


class Pointer(Model):
    model_config = ConfigDict(extra="ignore")
    policy_version: Literal["2.2.0"]
    strategy_path: Literal["standard/current/STRATEGY.json"]
    strategy_sha256: str
    entrypoint: Literal["scripts.current_standard"]


class ReleaseLock(Model):
    release: Literal["ANNUAL_DEFAULT_V2_2"] = "ANNUAL_DEFAULT_V2_2"
    hashes: dict[str, str]


class Attempt(Model):
    release: Literal["ANNUAL_DEFAULT_V2_2"] = "ANNUAL_DEFAULT_V2_2"
    event: Event
    account: str
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""


@beartype
def load_policy(root: Path = ROOT) -> Policy:
    pointer = Pointer.model_validate_json((root / "standard/CURRENT.json").read_bytes())
    file = root / pointer.strategy_path
    if sha(file) != pointer.strategy_sha256:
        raise ValueError("current policy hash changed")
    policy = Policy.model_validate_json(file.read_bytes())
    schedule_file = root / policy.schedule_path
    if sha(schedule_file) != policy.schedule_sha256:
        raise ValueError("default annual schedule hash changed")
    schedule = Schedule.model_validate_json(schedule_file.read_bytes())
    if schedule.mode != "ANNUAL" or schedule.sha256 != policy.schedule_identity_sha256:
        raise ValueError("current policy does not bind the annual schedule")
    if tuple(p.year for p in schedule.entries) != policy.verified_trade_years:
        raise ValueError("current verified years differ from available schedule")
    return policy


def release_snapshot(root: Path) -> ReleaseLock:
    files = {
        root / n
        for n in (
            "README.md",
            "standard/CURRENT.json",
            "standard/current/STRATEGY.json",
            "standard/current/PLAN.md",
            "standard/current/PLAN_REVIEW.json",
            "docs/CURRENT_STANDARD.md",
            "docs/STANDARD_V2.md",
            "docs/REPRODUCE_V2.md",
            "scripts/current_standard.py",
            "standard/current/tests/test_current.py",
            "experiments/v2-annual-repool/SOURCE_LOCK.json",
            "experiments/v2-annual-repool/ANNUAL.json",
        )
    }
    references = root / "experiments/v2-annual-repool/reference"
    expected = {
        f"{mode}_{cost}bp_{stage}"
        for mode, stage in (("FIXED", "bridge"), ("ANNUAL", "science"))
        for cost in (5, 20)
    }
    if {p.name for p in references.iterdir()} != expected:
        raise ValueError("release reference lock requires exactly four expected accounts")
    files.update(p for p in references.rglob("*") if p.is_file())
    return ReleaseLock(hashes={p.relative_to(root).as_posix(): sha(p) for p in sorted(files)})


def verify_release(root: Path) -> None:
    verify_annual_lock(root)
    saved = ReleaseLock.model_validate_json((root / CURRENT / "RELEASE_LOCK.json").read_bytes())
    if saved != release_snapshot(root):
        raise ValueError("default release source/doc/policy lock changed")


def freeze(root: Path = ROOT) -> None:
    load_policy(root)
    verify_annual_lock(root)
    path = root / CURRENT / "RELEASE_LOCK.json"
    if path.exists():
        verify_release(root)
        return
    if journal_path(root).exists() and journal_path(root).stat().st_size:
        raise ValueError("cannot freeze after native attempts")
    path.write_text(release_snapshot(root).model_dump_json(indent=2) + "\n")


def journal_path(root: Path) -> Path:
    return root / CURRENT / "evidence/attempts.jsonl"


def journal_event(
    root: Path, event: Event, account: str, attempt: str, output: Path, error: str = ""
) -> None:
    path = journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = Attempt(
        event=event,
        account=account,
        attempt_id=attempt,
        utc=datetime.now(UTC),
        output=os.path.relpath(output.resolve(), root.resolve()),
        error=error.replace(str(root.resolve()), "<root>"),
    )
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = [Attempt.model_validate_json(line) for line in handle.read().splitlines()]
        starts = [r for r in previous if r.event == "started"]
        if event == "started":
            if len(starts) >= 8:
                raise ValueError("current release hard8 attempts exhausted, including failures")
            if any(r.account == account for r in starts):
                raise ValueError("account already attempted; reuse original output")
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def budget_scope(root: Path) -> Iterator[None]:
    previous = annual_run.journal_path, annual_run.journal_event
    owned_root = root

    def path(root: Path = ROOT) -> Path:
        if root.resolve() != owned_root.resolve():
            raise ValueError("current release root mismatch")
        return journal_path(owned_root)

    annual_run.journal_path, annual_run.journal_event = path, journal_event
    try:
        yield
    finally:
        annual_run.journal_path, annual_run.journal_event = previous


def compare_reference_books(root: Path, output: Path) -> None:
    refs = root / "experiments/v2-annual-repool/reference"
    for reference in sorted(refs.iterdir()):
        if not reference.is_dir():
            continue
        recorded = json.loads((reference / "REFERENCE.json").read_text())
        verify_hashes(reference, recorded["hashes"])
        actual = output / reference.name
        for name in BOOKS:
            compare_frames(
                pd.read_csv(reference / f"{name}.csv.gz", float_precision="round_trip"),
                pd.read_csv(actual / f"{name}.csv.gz", float_precision="round_trip"),
                name,
            )
        for name in (
            "CONTRACT",
            "targets",
            "timings",
            "native_returns",
            "open_trades",
            "metrics",
            "native_total_return",
        ):
            compare_json(
                json.loads((reference / f"{name}.json").read_text()),
                json.loads((actual / f"{name}.json").read_text()),
                name,
            )


@beartype
def run_current(root: Path, output: Path, *, execute: bool = False) -> None:
    policy = load_policy(root)
    verify_release(root)
    print(
        f"Current policy {policy.policy_version}: {policy.default_rule}, "
        f"top{policy.top_k}, pool={policy.universe_refresh}, verified2022–2025",
        flush=True,
    )
    reference = root / "experiments/v2-annual-repool/reference"
    if execute:
        # The unchanged engine receives exactly the schedule validated from CURRENT/config.
        with budget_scope(root):
            annual_run.run_stage("bridge", output, root)
            annual_run.run_stage("science", output, root)
        compare_reference_books(root, output)
        reference = output
    build_report(root, reference, output / "report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/current-standard"))
    parser.add_argument("--execute", action="store_true", help="reproduce2bridges+2annual accounts")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    run_current(args.root, output, execute=args.execute)


if __name__ == "__main__":
    main()
