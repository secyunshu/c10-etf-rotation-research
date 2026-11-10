"""Raw-dataset signal oracle and immutable reference identity checks."""

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype

from rotbt.models import (
    ClosedTrade,
    DailyRecord,
    Dataset,
    Distribution,
    Fees,
    HoldingRecord,
    OrderRecord,
    ResearchPeriod,
    Result,
    Settings,
    Spec,
)
from scripts.etf_c10_2026_audit import FinancialScope, TargetRecord, reconcile
from scripts.etf_group_cap_targets import GroupMap, Role


@beartype
def raw_oracle(
    role: Role, datasets: list[Dataset], names: tuple[str, ...], groups: GroupMap, day: date
) -> TargetRecord:
    frames = {d.symbol: d.frame.loc[: str(day)] for d in datasets}
    scores: list[tuple[float, str, str]] = []
    shy = frames["SHY"]
    if len(shy) < 253 or shy.index[-1].date() != day:
        raise ValueError("raw oracle missing SHY signal day")
    cash_return = float(shy.tr.iloc[-1] / shy.tr.iloc[-253] - 1)
    for symbol in names:
        group = groups.group_for(symbol)
        frame = frames[symbol]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("raw oracle missing exact signal close")
        if len(frame) < 253:
            continue
        recent = frame.iloc[-20:]
        momentum = float(frame.tr.iloc[-1] / frame.tr.iloc[-253] - 1)
        if not math.isfinite(momentum) or not math.isfinite(cash_return):
            raise ValueError("nonfinite raw momentum")
        if (
            math.fsum(float(x) for x in recent.turnover) / 20 >= 1e7
            and all(float(x) > 0 for x in recent.volume)
            and float(frame.close.iloc[-1]) >= 5
        ):
            scores.append((momentum, symbol, group))
    picks: list[tuple[float, str]] = []
    occupied: set[str] = set()
    for score, symbol, group in sorted(scores, key=lambda x: (-x[0], x[1])):
        if role.cap_one_per_group and group in occupied:
            continue
        picks.append((score, symbol))
        occupied.add(group)
        if len(picks) >= role.top_k:
            break
    allocation = {s: 0.9 / role.top_k for score, s in picks if score > cash_return}
    allocation["SHY"] = max(0.0, 1 - sum(allocation.values()))
    return TargetRecord(day=day, q=allocation, reason="oracle")


def reference_path(root: Path, role: Role) -> Path:
    if role.pool != "legacy36" or role.cap_one_per_group:
        raise ValueError("only original uncapped legacy roles have old reference accounts")
    if role.cohort == "historical2022_2025":
        return (
            root.parent
            / "c10-etf-rotation-research/data/reference"
            / f"{role.rule}_{role.cost_bps}bp"
        )
    mode = "reset2026" if role.cohort == "reset2026" else "continuous2022"
    return root / "c10_2026_outputs/20261109_v1" / f"{mode}_{role.rule}_{role.cost_bps}bp"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes(hashes: dict[str, str]) -> None:
    for filename, expected in hashes.items():
        if sha(Path(filename)) != expected:
            raise ValueError(f"frozen source changed: {filename}")


def settings_for(root: Path, role: Role) -> Settings:
    return Settings(
        fees=Fees.model_validate_json((root / "fees_source.json").read_text()).fees,
        initial_cash=50000,
        invested=0.95,
        slip_bps=role.cost_bps,
        tax=0,
        research_period=ResearchPeriod(first_year=2022, last_year=2025, split_year=2023),
    )


def validate_reference_identity(role: Role, task: dict[str, Any]) -> None:
    """C10/C11/C00 are the old true K identities; spec.top=3 is only their shell."""
    if (
        task.get("key") != f"{role.rule}_{role.cost_bps}bp"
        or task.get("spec", {}).get("label") != role.rule
        or task.get("start") != str(role.start)
        or task.get("end") != str(role.through)
        or task.get("stage") != "full2022_2025"
        or task.get("spec", {}).get("windows") != [252]
        or task.get("spec", {}).get("frequency") != "monthly"
    ):
        raise ValueError("old reference economic identity mismatch")


def bind_reference(root: Path, role: Role) -> dict[str, str]:
    """Verify original account provenance before a reuse receipt is frozen."""
    from scripts import etf_c10_2026_run as native

    folder = reference_path(root, role)
    if role.cohort == "historical2022_2025":
        accounts = folder.parent.parent / "ACCOUNTS.json"
        entries = [e for e in json.loads(accounts.read_text()) if e["name"] == folder.name]
        if (
            len(entries) != 1
            or entries[0]["rule"] != role.rule
            or entries[0]["cost_bps"] != role.cost_bps
        ):
            raise ValueError("old reference registry identity mismatch")
        hashes = {str(folder / n): h for n, h in entries[0]["reference_files"].items()}
        verify_hashes(hashes)
        task = json.loads((folder / "task.json").read_text())
        validate_reference_identity(role, task)
        if Settings.model_validate(task["settings"]) != settings_for(root, role):
            raise ValueError("old reference settings mismatch")
        if json.loads((folder / "verified.json").read_text())["status"] != "PASS":
            raise ValueError("old reference not verified")
        hashes[str(accounts)] = sha(accounts)
        return hashes
    expected = native.Contract(
        rule=role.rule,
        cost=role.cost_bps,
        mode="reset2026" if role.cohort == "reset2026" else "continuous2022",
    )
    if native.Contract.model_validate_json((folder / "CONTRACT.json").read_text()) != expected:
        raise ValueError("old 2026 reference contract mismatch")
    artifact = native.Artifact.model_validate_json((folder / "ARTIFACT.json").read_text())
    verified = native.Receipt.model_validate_json((folder / "VERIFIED.json").read_text())
    if artifact.contract != expected or verified.keys != [expected.key]:
        raise ValueError("old 2026 reference identity mismatch")
    sources = verify_reference_lock(root)
    sources.update({str(p): sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    return sources


def verify_reference_lock(root: Path) -> dict[str, str]:
    """Validate the already approved V2 amendment chain without running its recovery."""
    from scripts import etf_c10_2026_run as native
    from scripts.etf_c10_2026_resume import RecoveryAmendment

    output = root / native.OUT
    prior_path = output / "RUN_LOCK.json"
    current_path = output / "RUN_LOCK_V2.json"
    prior = native.HashLock.model_validate_json(prior_path.read_text())
    if not current_path.exists():
        native.verify_lock(prior)
        return {str(prior_path): sha(prior_path)}
    amendment_path = output / "EMPTY_TRADES_AMENDMENT.json"
    amendment = RecoveryAmendment.model_validate_json(amendment_path.read_text())
    if sha(prior_path) != amendment.prior_lock_sha256:
        raise ValueError("original reference source lock changed")
    current = native.HashLock.model_validate_json(current_path.read_text())
    runner = str(root / "scripts/etf_c10_2026_run.py")
    if set(current.hashes) != set(prior.hashes) | {
        str(root / "scripts/etf_c10_2026_resume.py"),
        str(amendment_path),
    }:
        raise ValueError("unregistered reference V2 source scope")
    if any(current.hashes[p] != h for p, h in prior.hashes.items() if p != runner):
        raise ValueError("reference V2 changed frozen inputs or unrelated source")
    backup = output / "SOURCE_BEFORE_EMPTY_TRADE_FIX.py"
    if sha(backup) != prior.hashes[runner]:
        raise ValueError("original reference runner evidence changed")
    native.verify_lock(current)
    paths = [prior_path, current_path, amendment_path, backup]
    recovery = output / "RECOVERY_AUDIT.json"
    if recovery.exists():
        paths.append(recovery)
    return {**current.hashes, **{str(p): sha(p) for p in paths}}


def compare_books(reference: Path, actual: Path) -> None:
    from scripts.etf_c10_2026_run import BOOKS, compare_book_frames

    for name in BOOKS:
        compare_book_frames(
            pd.read_csv(reference / f"{name}.csv.gz", float_precision="round_trip"),
            pd.read_csv(actual / f"{name}.csv.gz", float_precision="round_trip"),
        )


def audit_historical_reuse(
    root: Path, role: Role, datasets: list[Dataset], names: tuple[str, ...], groups: GroupMap
) -> None:
    """Read the actual saved books and native scalar, never synthesize native daily returns."""
    from pydantic import TypeAdapter

    from rotbt.analysis import analyze
    from scripts import etf_c10_2026_run as native

    folder = reference_path(root, role)
    frames = {
        name: pd.read_csv(folder / f"{name}.csv.gz", float_precision="round_trip")
        for name in native.BOOKS
    }
    result = Result(
        spec=Spec(name=role.rule, layer="hold", lookback=252, top=3),
        daily=[DailyRecord.model_validate(r) for r in frames["daily"].to_dict("records")],
        orders=[OrderRecord.model_validate(r) for r in frames["orders"].to_dict("records")],
        holdings=[HoldingRecord.model_validate(r) for r in frames["holdings"].to_dict("records")],
        distributions=[
            Distribution.model_validate(r) for r in frames["dividends"].to_dict("records")
        ],
        trades=[ClosedTrade.model_validate(r) for r in frames["trades"].to_dict("records")],
        native_total_return=json.loads((folder / "verified.json").read_text())["native_return"],
        splits=[s for d in datasets for s in d.splits if role.start <= s.day <= role.through],
    )
    settings = settings_for(root, role)
    reconcile(
        result,
        datasets,
        FinancialScope(start=role.start, through=role.through, settings=settings, terminal=True),
    )
    actual = TypeAdapter(list[TargetRecord]).validate_json((folder / "targets.json").read_text())
    import exchange_calendars as xcals

    prior = xcals.get_calendar("XNYS").previous_session(pd.Timestamp(role.start)).date()
    contract = native.Contract(rule=role.rule, cost=role.cost_bps, mode="bridge")
    if [r.day for r in actual] != [prior, *native.decision_days(contract)]:
        raise ValueError("reused historical target calendar mismatch")
    for record in actual:
        expected = raw_oracle(role, datasets, names, groups, record.day)
        if record.q.keys() != expected.q.keys() or any(
            abs(w - expected.q[s]) > 1e-12 for s, w in record.q.items()
        ):
            raise ValueError("reused historical independent target mismatch")
    pd.testing.assert_frame_equal(
        pd.DataFrame(json.loads((folder / "metrics.json").read_text())),
        pd.DataFrame([m.model_dump(mode="json") for m in analyze(result, settings)]),
        check_exact=False,
        rtol=0,
        atol=1e-10,
    )
