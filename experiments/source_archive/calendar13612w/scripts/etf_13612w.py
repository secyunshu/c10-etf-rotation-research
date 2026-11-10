"""Fixed calendar13612W experiment; the unchanged native broker owns every account."""

import argparse
import fcntl
import json
import math
import os
import shutil
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import exchange_calendars as xcals
import pandas as pd
from beartype import beartype
from pydantic import ConfigDict, TypeAdapter, model_validator

from rotbt.analysis import analyze
from rotbt.models import Dataset, Model
from rotbt.phase2.models import FeatureStore, SignalContext
from scripts import etf_c10_2026_run as native
from scripts.etf_c10_2026_audit import TargetRecord, describe, equal
from scripts.etf_c10_2026_inputs import Prepared
from scripts.etf_group_cap_audit import compare_books, settings_for, sha, verify_hashes
from scripts.etf_group_cap_audit import raw_oracle as original_oracle
from scripts.etf_group_cap_run import (
    configuration,
    freeze,
    model_write,
    verify_joint_acceptance,
    write,
)
from scripts.etf_group_cap_run import prepared_for as group_prepared
from scripts.etf_group_cap_targets import Cohort, GroupMap, Role
from scripts.etf_group_cap_targets import decision as original_decision
from scripts.etf_large36_return_targets import Rule as LegacyRule

RESEARCH = "research/ETF13612W_20261110"
OUT = "etf13612w_outputs/20261110_v1"
IDENTITY = "ETF13612W_CALENDAR_MONTHENDS_GROUPED_20261110"


class Trial(Model):
    model_config = ConfigDict(frozen=True)
    key: str
    k: Literal[1, 2, 3]
    cost_bps: Literal[5, 20]
    cohort: Cohort
    kind: Literal["bridge", "science"]
    score: Literal["original252", "calendar_13612w"]

    @model_validator(mode="after")
    def fixed_identity(self) -> "Trial":
        prefix = "bridge" if self.kind == "bridge" else "W13612"
        expected = f"{prefix}_K{self.k}_{self.cost_bps}bp_{self.cohort}"
        if self.key != expected or (self.kind == "bridge") != (self.score == "original252"):
            raise ValueError("trial economic identity differs from frozen score/K/cohort")
        return self


class Reference(Model):
    model_config = ConfigDict(extra="ignore")
    key: str
    rule: native.Rule
    cost_bps: Literal[5, 20]
    cohort: Cohort
    reference_files: dict[str, str]

    @model_validator(mode="after")
    def fixed_identity(self) -> "Reference":
        expected = {f"{n}.csv.gz" for n in native.BOOKS} | {
            "targets.json",
            "native_total_return.json",
            "metrics.json" if self.cohort == "historical2022_2025" else "ytd.json",
        }
        if (
            self.key != f"{self.rule}_{self.cost_bps}bp_{self.cohort}"
            or not expected <= self.reference_files.keys()
            or set(self.reference_files) - expected - {"native_returns.json", "open_trades.json"}
        ):
            raise ValueError("public V2 reference identity or filenames changed")
        return self


class Attempt(Model):
    study: Literal["ETF13612W_CALENDAR_MONTHENDS_GROUPED_20261110"] = (
        "ETF13612W_CALENDAR_MONTHENDS_GROUPED_20261110"
    )
    key: str
    state: Literal["started", "native_completed", "finished", "failed", "revalidated"]
    utc: datetime
    error: str = ""


def catalog(root: Path) -> list[Trial]:
    trials = TypeAdapter(list[Trial]).validate_json((root / RESEARCH / "CATALOG.json").read_text())
    expected = {
        f"W13612_K{k}_{cost}bp_{cohort}"
        for k in (1, 2, 3)
        for cost in (5, 20)
        for cohort in ("historical2022_2025", "reset2026", "continuous2022_2026")
    } | {"bridge_K1_5bp_historical2022_2025", "bridge_K2_5bp_continuous2022_2026"}
    if len(trials) != 20 or {t.key for t in trials} != expected:
        raise ValueError("exact frozen18 science and2 bridges required")
    return trials


def role_for(trial: Trial) -> Role:
    return Role(
        key=f"historical34_cap1_K{trial.k}_{trial.cost_bps}bp_{trial.cohort}",
        pool="historical34",
        cap_one_per_group=True,
        top_k=trial.k,
        cost_bps=trial.cost_bps,
        cohort=trial.cohort,
        mode="new_native",
        alias_of=None,
    )


def contract_for(trial: Trial) -> native.Contract:
    role = role_for(trial)
    return native.Contract(
        rule=role.rule,
        cost=role.cost_bps,
        mode=(
            "bridge"
            if trial.cohort == "historical2022_2025"
            else "reset2026"
            if trial.cohort == "reset2026"
            else "continuous2022"
        ),
    )


@beartype
def calendar_endpoints(day: date) -> dict[int, date]:
    """Use last exchange sessions of calendar months, including exact signal month-end."""
    calendar = xcals.get_calendar("XNYS")
    month = pd.Period(day, freq="M")
    endpoints = {
        lag: calendar.sessions_in_range(
            (month - lag).start_time, (month - lag).end_time.normalize()
        )[-1].date()
        for lag in (0, 1, 3, 6, 12)
    }
    if endpoints[0] != day:
        raise ValueError("signal must be the actual XNYS month-end")
    return endpoints


@beartype
def calendar_score(frame: pd.DataFrame, day: date) -> float:
    endpoints = calendar_endpoints(day)
    points: dict[int, float] = {}
    for lag, endpoint in endpoints.items():
        stamp = pd.Timestamp(endpoint)
        if stamp not in frame.index:
            raise ValueError(f"missing exact calendar endpoint: {endpoint}")
        value = float(frame.tr.loc[stamp])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"nonfinite or nonpositive calendar endpoint: {endpoint}")
        points[lag] = value
    return (
        math.fsum(
            weight * (points[0] / points[lag] - 1)
            for lag, weight in ((1, 12), (3, 4), (6, 2), (12, 1))
        )
        / 4
    )


@beartype
def decision(
    trial: Trial, features: FeatureStore, names: tuple[str, ...], groups: GroupMap, day: date
) -> TargetRecord:
    if trial.kind == "bridge":
        return original_decision(role_for(trial), features, names, groups, day)
    calendar_endpoints(day)
    if len(names) != len(set(names)) or "SHY" in names:
        raise ValueError("invalid risk pool membership")
    scores: dict[str, float] = {}
    for symbol in (*names, "SHY"):
        if symbol != "SHY":
            groups.group_for(symbol)
        frame = features.frames[symbol].loc[: str(day)]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("missing exact prior signal close")
        if len(frame) < 253:
            if symbol == "SHY":
                raise ValueError("insufficient253-observation SHY warmup")
            continue
        row = frame.iloc[-1]
        values = [float(row[k]) for k in ("raw_close", "turnover20", "positive20")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite qualification input")
        score = calendar_score(frame, day)
        if symbol == "SHY" or (values[0] >= 5 and values[1] >= 1e7 and values[2] == 20):
            scores[symbol] = score
    selected: list[str] = []
    occupied: set[str] = set()
    for symbol in sorted((s for s in scores if s != "SHY"), key=lambda s: (-scores[s], s)):
        group = groups.group_for(symbol)
        if group in occupied:
            continue
        selected.append(symbol)
        occupied.add(group)
        if len(selected) == trial.k:
            break
    q = {s: 0.9 / trial.k for s in selected if scores[s] > scores["SHY"]}
    q["SHY"] = max(0.0, 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


@beartype
def raw_oracle(
    trial: Trial, datasets: list[Dataset], names: tuple[str, ...], groups: GroupMap, day: date
) -> TargetRecord:
    """Independent raw endpoints and liquidity; never calls the feature decision."""
    if trial.kind == "bridge":
        return original_oracle(role_for(trial), datasets, names, groups, day)
    calendar = xcals.get_calendar("XNYS")
    current = pd.Timestamp(day)
    month_end = current + pd.offsets.MonthEnd(0)
    if calendar.date_to_session(month_end, direction="previous").date() != day:
        raise ValueError("raw oracle requires true month-end")
    endpoints: list[date] = []
    for months in (1, 3, 6, 12):
        shifted = current - pd.DateOffset(months=months)
        last = shifted + pd.offsets.MonthEnd(0)
        endpoints.append(calendar.date_to_session(last, direction="previous").date())
    frames = {d.symbol: d.frame.loc[: str(day)] for d in datasets}
    if len(frames) != len(datasets):
        raise ValueError("duplicate raw datasets")
    scored: list[tuple[float, str, str]] = []
    cash_score: float | None = None
    for symbol in (*names, "SHY"):
        frame = frames[symbol]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("raw exact signal day missing")
        if len(frame) < 253:
            if symbol == "SHY":
                raise ValueError("raw253-observation cash warmup missing")
            continue
        levels = []
        for stamp in (day, *endpoints):
            if pd.Timestamp(stamp) not in frame.index:
                raise ValueError("raw calendar endpoint missing")
            levels.append(float(frame.tr.loc[pd.Timestamp(stamp)]))
        if any(not math.isfinite(v) or v <= 0 for v in levels):
            raise ValueError("raw endpoint nonfinite or nonpositive")
        weighted = 0.0
        for denominator, weight in zip(levels[1:], (12.0, 4.0, 2.0, 1.0), strict=True):
            weighted += weight * (levels[0] / denominator - 1)
        score = weighted / 4.0
        if symbol == "SHY":
            cash_score = score
            continue
        group = groups.group_for(symbol)
        recent = frame.iloc[-20:]
        turnover = [float(v) for v in recent.turnover]
        volumes = [float(v) for v in recent.volume]
        close = float(frame.close.iloc[-1])
        if any(not math.isfinite(v) for v in (*turnover, *volumes, close)):
            raise ValueError("raw qualification nonfinite")
        if math.fsum(turnover) / 20 >= 1e7 and all(v > 0 for v in volumes) and close >= 5:
            scored.append((score, symbol, group))
    if cash_score is None:
        raise ValueError("raw SHY score unavailable")
    picks: list[tuple[float, str]] = []
    used: set[str] = set()
    for score, symbol, group in sorted(scored, key=lambda row: (-row[0], row[1])):
        if group in used:
            continue
        picks.append((score, symbol))
        used.add(group)
        if len(picks) == trial.k:
            break
    allocation = {symbol: 0.9 / trial.k for score, symbol in picks if score > cash_score}
    allocation["SHY"] = max(0.0, 1 - sum(allocation.values()))
    return TargetRecord(day=day, q=allocation, reason="oracle")


@contextmanager
def scoped_hooks(trial: Trial, groups: GroupMap) -> Iterator[None]:
    namespace = native.__dict__
    original = namespace["target"], namespace["oracle"]
    economic_groups = groups

    def target(
        rule: LegacyRule | str,
        features: FeatureStore,
        day: date,
        names: tuple[str, ...],
        groups: tuple[tuple[str, ...], ...],
        context: SignalContext | None = None,
    ) -> dict[str, float]:
        return decision(trial, features, names, economic_groups, day).q

    def oracle(
        rule: str, datasets: list[Dataset], risk_names: tuple[str, ...], day: date
    ) -> TargetRecord:
        return raw_oracle(trial, datasets, risk_names, economic_groups, day)

    namespace.update(target=target, oracle=oracle)
    try:
        yield
    finally:
        namespace.update(target=original[0], oracle=original[1])


def prepared_for(root: Path, trial: Trial) -> Prepared:
    pools, groups = configuration(root)
    return group_prepared(root, role_for(trial), pools, groups)


def public_root(root: Path) -> Path:
    return root.parent / "c10-etf-rotation-research"


def references(root: Path) -> list[Reference]:
    accounts = TypeAdapter(list[Reference]).validate_json(
        (public_root(root) / "standard/ACCOUNTS.json").read_text()
    )
    expected = {
        f"{r}_{cost}bp_{cohort}"
        for r in native.RULES
        for cost in (5, 20)
        for cohort in ("historical2022_2025", "reset2026", "continuous2022_2026")
    }
    if len(accounts) != 36 or {a.key for a in accounts} != expected:
        raise ValueError("exact36 public V2 reused accounts required")
    for account in accounts:
        folder = public_root(root) / "standard/reference" / account.key
        if {p.name for p in folder.iterdir()} != account.reference_files.keys():
            raise ValueError("public reference file membership changed")
        verify_hashes({str(folder / name): h for name, h in account.reference_files.items()})
    return accounts


def journal(root: Path) -> Path:
    return root / OUT / "NATIVE_ATTEMPTS.jsonl"


def native_count(root: Path) -> int:
    path = journal(root)
    return (
        sum(
            Attempt.model_validate_json(line).state == "started"
            for line in path.read_text().splitlines()
        )
        if path.exists()
        else 0
    )


def append_attempt(
    root: Path,
    key: str,
    state: Literal["started", "native_completed", "finished", "failed", "revalidated"],
    error: str = "",
) -> None:
    path = journal(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        records = [Attempt.model_validate_json(line) for line in handle.read().splitlines()]
        if state == "started" and sum(a.state == "started" for a in records) >= 28:
            raise ValueError(
                "frozen28-native-attempt budget exhausted, including failures/fixtures"
            )
        handle.write(
            Attempt(key=key, state=state, utc=datetime.now(UTC), error=error).model_dump_json()
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def prepare(root: Path) -> None:
    root = root.resolve()
    protocol = json.loads((root / RESEARCH / "PROTOCOL.json").read_text())
    verify_hashes(protocol["source_hashes"])
    pools, groups = configuration(root)
    projected = {
        g: tuple(s for s in members if s in pools.historical34)
        for g, members in groups.groups.items()
    }
    if (
        protocol["id"] != IDENTITY
        or tuple(protocol["pool"]) != pools.historical34
        or {g: tuple(v) for g, v in protocol["groups"].items()} != projected
    ):
        raise ValueError("protocol score pool/group identity changed")
    trials = catalog(root)
    accounts = references(root)
    accepted = verify_joint_acceptance(root)
    source = json.loads(
        (root / "inputs_etf_group_cap_20261109/SOURCE_PROVENANCE.json").read_text()
    )["hashes"]
    code = [
        p
        for base in (root / "src/rotbt", root / "scripts", public_root(root) / "src")
        for p in base.rglob("*.py")
    ]
    files = code + [
        root / RESEARCH / "PROTOCOL.json",
        root / RESEARCH / "CATALOG.json",
    ]
    files += list((root / "tests").glob("test_etf_13612w*.py"))
    files += [p for p in (public_root(root) / "standard").rglob("*") if p.is_file()]
    hashes = {**protocol["source_hashes"], **accepted, **source, **{str(p): sha(p) for p in files}}
    freeze(root / RESEARCH / "RUN_LOCK.json", hashes)
    output = root / OUT
    output.mkdir(parents=True, exist_ok=True)
    registry: list[dict[str, Any]] = []
    for trial in trials:
        if trial.kind == "science":
            registry.append(
                {
                    "key": trial.key,
                    "score": trial.score,
                    "rule": contract_for(trial).rule,
                    "k": trial.k,
                    "cost_bps": trial.cost_bps,
                    "cohort": trial.cohort,
                    "path": str(output / trial.key),
                    "mode": "new_native",
                    "source_hashes": {},
                }
            )
    for account in accounts:
        folder = public_root(root) / "standard/reference" / account.key
        registry.append(
            {
                "key": account.key,
                "score": "original252" if account.rule in ("C10", "C11", "C00") else "benchmark",
                "rule": account.rule,
                "k": {"C10": 1, "C11": 2, "C00": 3}.get(account.rule),
                "cost_bps": account.cost_bps,
                "cohort": account.cohort,
                "path": str(folder),
                "mode": "reuse",
                "source_hashes": {str(folder / n): h for n, h in account.reference_files.items()},
            }
        )
    if not (output / "REGISTRY.json").exists():
        write(output / "REGISTRY.json", {"roles": registry})
    elif json.loads((output / "REGISTRY.json").read_text()) != {"roles": registry}:
        raise ValueError("immutable54-role registry changed")


def verify_run_lock(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = json.loads((root / RESEARCH / "RUN_LOCK.json").read_text())["hashes"]
    verify_hashes(hashes)
    return hashes


def compare_bridge(root: Path, trial: Trial, artifact: native.Artifact, folder: Path) -> None:
    contract = contract_for(trial)
    key = f"{contract.rule}_{trial.cost_bps}bp_{trial.cohort}"
    reference = next(a for a in references(root) if a.key == key)
    source = public_root(root) / "standard/reference" / reference.key
    compare_books(source, folder)
    targets = TypeAdapter(list[TargetRecord]).validate_json((source / "targets.json").read_text())
    if targets != artifact.targets:
        raise ValueError("bridge exact target/reason mismatch")
    equal(
        float(json.loads((source / "native_total_return.json").read_text())["native_total_return"]),
        artifact.account.native_total_return,
        "bridge actual native scalar",
        1e-10,
    )
    if "native_returns.json" not in reference.reference_files:
        raise ValueError("both frozen bridges require actual original native return vectors")
    returns = TypeAdapter(list[native.NativeReturn]).validate_json(
        (source / "native_returns.json").read_text()
    )
    trades = TypeAdapter(list[native.OpenTrade]).validate_json(
        (source / "open_trades.json").read_text()
    )
    if returns != artifact.native_returns or trades != artifact.open_trades:
        raise ValueError("bridge exact native vector/open Trade mismatch")
    saved = source / ("metrics.json" if contract.mode == "bridge" else "ytd.json")
    actual = folder / ("METRICS.json" if contract.mode == "bridge" else "YTD.json")
    if json.loads(saved.read_text()) != json.loads(actual.read_text()):
        raise ValueError("bridge exact saved metrics mismatch")


def validate_saved(root: Path, trial: Trial, prepared: Prepared | None, groups: GroupMap) -> None:
    if prepared is None:
        raise ValueError("actual35-feed prepared inputs required")
    folder = root / OUT / trial.key
    artifact = native.Artifact.model_validate_json((folder / "ARTIFACT.json").read_text())
    if (
        Trial.model_validate_json((folder / "TRIAL.json").read_text()) != trial
        or native.Contract.model_validate_json((folder / "CONTRACT.json").read_text())
        != contract_for(trial)
        or artifact.contract != contract_for(trial)
    ):
        raise ValueError("saved trial identity/execution shell mismatch")
    if json.loads((folder / "NATIVE_COMPLETE.json").read_text()) != {
        "artifact_sha256": sha(folder / "ARTIFACT.json")
    }:
        raise ValueError("artifact lacks completed fault-free native evidence")
    settings = settings_for(root, role_for(trial))
    with scoped_hooks(trial, groups):
        native.audit(artifact, prepared, settings)
    with TemporaryDirectory(prefix="13612w-books-") as temporary:
        generated = Path(temporary)
        native.export_books(generated, artifact)
        for name in native.BOOKS:
            filename = f"{name}.csv.gz"
            if not (folder / filename).exists():
                shutil.copyfile(generated / filename, folder / filename)
            native.compare_book_frames(
                pd.read_csv(generated / filename, float_precision="round_trip"),
                pd.read_csv(folder / filename, float_precision="round_trip"),
            )
    values: dict[str, Any] = {
        "TARGETS.json": [r.model_dump(mode="json") for r in artifact.targets],
        "NATIVE_TOTAL_RETURN.json": {"native_total_return": artifact.account.native_total_return},
    }
    if trial.cohort == "historical2022_2025":
        values["METRICS.json"] = [
            m.model_dump(mode="json") for m in analyze(artifact.account, settings)
        ]
    else:
        values["YTD.json"] = describe(
            artifact.account, continuous=trial.cohort == "continuous2022_2026"
        ).model_dump(mode="json")
    for name, value in values.items():
        path = folder / name
        if not path.exists():
            write(path, value)
        elif json.loads(path.read_text()) != value:
            raise ValueError(f"saved native scalar/targets/metrics mismatch: {name}")
    if trial.kind == "bridge":
        compare_bridge(root, trial, artifact, folder)
    receipt = folder / "VERIFIED.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["status"] != "PASS" or saved["key"] != trial.key:
            raise ValueError("saved verification identity mismatch")
        verify_hashes(saved["hashes"])
    else:
        write(
            receipt,
            {
                "status": "PASS",
                "key": trial.key,
                "score": trial.score,
                "hashes": {
                    str(p): sha(p)
                    for p in folder.iterdir()
                    if p.is_file() and p.name != "FAILURE.txt"
                },
            },
        )


def execute(root: Path, trial: Trial, prepared: Prepared | None, groups: GroupMap) -> None:
    folder = root / OUT / trial.key
    if folder.exists():
        if (folder / "ARTIFACT.json").exists() and (folder / "NATIVE_COMPLETE.json").exists():
            verified = (folder / "VERIFIED.json").exists()
            validate_saved(root, trial, prepared, groups)
            if not verified:
                append_attempt(root, trial.key, "revalidated")
            return
        raise ValueError(f"failed/partial native evidence preserved; no automatic retry: {folder}")
    if trial not in catalog(root) or prepared is None:
        raise ValueError("registered trial and actual prepared inputs required")
    if (
        len(prepared.datasets) != 35
        or prepared.last != contract_for(trial).through
        or tuple(sorted(d.symbol for d in prepared.datasets))
        != tuple(sorted((*prepared.risk_names, "SHY")))
    ):
        raise ValueError("actual feed set or physical cohort cutoff mismatch")
    folder.mkdir(parents=True, exist_ok=False)
    model_write(folder / "TRIAL.json", trial)
    model_write(folder / "CONTRACT.json", contract_for(trial))
    settings = settings_for(root, role_for(trial))
    with scoped_hooks(trial, groups):
        append_attempt(root, trial.key, "started")
        try:
            native._native(prepared, contract_for(trial), settings, folder)  # pyright: ignore[reportPrivateUsage]
            write(
                folder / "NATIVE_COMPLETE.json", {"artifact_sha256": sha(folder / "ARTIFACT.json")}
            )
            append_attempt(root, trial.key, "native_completed")
            validate_saved(root, trial, prepared, groups)
        except BaseException as error:
            with (folder / "FAILURE.txt").open("a", encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
            append_attempt(root, trial.key, "failed", str(error))
            raise
    append_attempt(root, trial.key, "finished")
    print(f"PASS {trial.key}", flush=True)


def stage_receipt(root: Path, stage: Literal["bridge", "science"]) -> None:
    trials = catalog(root)
    selected = [t for t in trials if stage == "science" or t.kind == "bridge"]
    hashes: dict[str, str] = {}
    for trial in selected:
        path = root / OUT / trial.key / "VERIFIED.json"
        receipt = json.loads(path.read_text())
        if receipt["status"] != "PASS" or receipt["key"] != trial.key:
            raise ValueError("stage is missing a verified frozen trial")
        verify_hashes(receipt["hashes"])
        hashes.update(receipt["hashes"])
        hashes[str(path)] = sha(path)
    sources = verify_run_lock(root)
    refs = references(root)
    if stage == "science":
        for account in refs:
            hashes.update(
                {
                    str(public_root(root) / "standard/reference" / account.key / name): h
                    for name, h in account.reference_files.items()
                }
            )
        records = [
            Attempt.model_validate_json(line) for line in journal(root).read_text().splitlines()
        ]
        starts = [a.key for a in records if a.state == "started"]
        if set(starts) != {t.key for t in trials} or not 20 <= len(starts) <= 28:
            raise ValueError("final actual native attempts differ from frozen20 paths/hard28budget")
        hashes[str(journal(root))] = sha(journal(root))
        hashes[str(root / OUT / "REGISTRY.json")] = sha(root / OUT / "REGISTRY.json")
        counts = {
            state: sum(a.state == state for a in records)
            for state in ("started", "native_completed", "finished", "failed", "revalidated")
        }
    else:
        counts = {}
    hashes[str(root / RESEARCH / "RUN_LOCK.json")] = sha(root / RESEARCH / "RUN_LOCK.json")
    hashes.update(sources)
    value = {
        "status": "PASS",
        "keys": [t.key for t in selected],
        "science": 18 if stage == "science" else 0,
        "bridges": 2,
        "reuse_roles": 36,
        "report_roles": 54,
        "journal": counts,
        "hashes": hashes,
    }
    path = root / OUT / ("FULL_AUDIT.json" if stage == "science" else "BRIDGE_AUDIT.json")
    if not path.exists():
        write(path, value)
    elif json.loads(path.read_text()) != value:
        raise ValueError("immutable stage audit differs")


def run(root: Path, stage: Literal["bridge", "science"]) -> None:
    root = root.resolve()
    verify_run_lock(root)
    _, groups = configuration(root)
    if stage == "science":
        receipt = json.loads((root / OUT / "BRIDGE_AUDIT.json").read_text())
        expected = [t.key for t in catalog(root) if t.kind == "bridge"]
        if receipt["status"] != "PASS" or receipt["keys"] != expected:
            raise ValueError("both fixed bridges must pass before science")
        verify_hashes(receipt["hashes"])
    with (root / OUT / ".native.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache: dict[bool, Prepared] = {}
        for trial in catalog(root):
            if (trial.kind == "bridge") != (stage == "bridge"):
                continue
            key = trial.cohort == "historical2022_2025"
            if key not in cache:
                cache[key] = prepared_for(root, trial)
            execute(root, trial, cache[key], groups)
        stage_receipt(root, stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=("prepare", "bridge", "science"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root)
    else:
        run(args.root, args.stage)


if __name__ == "__main__":
    main()
