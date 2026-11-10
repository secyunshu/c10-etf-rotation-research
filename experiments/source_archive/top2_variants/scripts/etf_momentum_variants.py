"""Six frozen top2 mechanisms; unchanged Backtrader code owns execution and accounting."""

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
from statistics import stdev
from tempfile import TemporaryDirectory
from typing import Any, Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype
from pydantic import ConfigDict, TypeAdapter, model_validator

from rotbt.analysis import analyze
from rotbt.models import Dataset, Model
from rotbt.phase2.models import FeatureStore, SignalContext
from scripts import etf_13612w as prior
from scripts import etf_c10_2026_run as native
from scripts.etf_c10_2026_audit import TargetRecord, describe
from scripts.etf_c10_2026_inputs import Prepared
from scripts.etf_group_cap_audit import raw_oracle as base_oracle
from scripts.etf_group_cap_audit import settings_for, sha, verify_hashes
from scripts.etf_group_cap_inputs import clip_dataset, features_for
from scripts.etf_group_cap_run import (
    configuration,
    freeze,
    model_write,
    verify_joint_acceptance,
    write,
)
from scripts.etf_group_cap_run import prepared_for as group_prepared
from scripts.etf_group_cap_targets import Cohort, GroupMap, Role
from scripts.etf_group_cap_targets import decision as base_decision
from scripts.etf_large36_return_targets import Rule as LegacyRule

RESEARCH = "research/ETF_MOMENTUM_VARIANTS_20261110"
OUT = "etf_momentum_variants_outputs/20261110_v1"
IDENTITY = "ETF_V2_TOP2_SIX_MOMENTUM_VARIANTS_20261110"
Variant = Literal["E01", "E02", "E03", "E04", "E05", "E06"]
VARIANTS: tuple[Variant, ...] = ("E01", "E02", "E03", "E04", "E05", "E06")


class Trial(Model):
    model_config = ConfigDict(frozen=True)
    key: str
    variant: Literal["BASE252", "E01", "E02", "E03", "E04", "E05", "E06"]
    cost_bps: Literal[5, 20]
    cohort: Cohort
    kind: Literal["bridge", "science"]
    k: Literal[2]

    @model_validator(mode="after")
    def fixed_identity(self) -> "Trial":
        prefix = "bridge" if self.kind == "bridge" else self.variant
        if self.key != f"{prefix}_K2_{self.cost_bps}bp_{self.cohort}" or (
            self.kind == "bridge"
        ) != (self.variant == "BASE252"):
            raise ValueError("frozen mechanism/outer trial identity mismatch")
        return self


class Score(Model):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)
    rank: float
    r252: float
    gate_value: float
    above_trend: bool


class Attempt(Model):
    study: Literal["ETF_V2_TOP2_SIX_MOMENTUM_VARIANTS_20261110"] = (
        "ETF_V2_TOP2_SIX_MOMENTUM_VARIANTS_20261110"
    )
    key: str
    state: Literal["started", "native_completed", "finished", "failed", "revalidated"]
    utc: datetime
    error: str = ""


def catalog(root: Path) -> list[Trial]:
    trials = TypeAdapter(list[Trial]).validate_json((root / RESEARCH / "CATALOG.json").read_text())
    expected = {
        f"{v}_K2_{cost}bp_{cohort}"
        for v in VARIANTS
        for cost in (5, 20)
        for cohort in ("historical2022_2025", "reset2026", "continuous2022_2026")
    } | {"bridge_K2_5bp_historical2022_2025", "bridge_K2_5bp_continuous2022_2026"}
    if len(trials) != 38 or {t.key for t in trials} != expected:
        raise ValueError("exact frozen36 science and2 bridges required")
    return trials


def role_for(trial: Trial) -> Role:
    return Role(
        key=f"historical34_cap1_K2_{trial.cost_bps}bp_{trial.cohort}",
        pool="historical34",
        cap_one_per_group=True,
        top_k=2,
        cost_bps=trial.cost_bps,
        cohort=trial.cohort,
        mode="new_native",
        alias_of=None,
    )


def contract_for(trial: Trial) -> native.Contract:
    return native.Contract(
        rule="C11",
        cost=trial.cost_bps,
        mode="bridge"
        if trial.cohort == "historical2022_2025"
        else "reset2026"
        if trial.cohort == "reset2026"
        else "continuous2022",
    )


@beartype
def score(variant: Variant, frame: pd.DataFrame, day: date) -> Score:
    prefix = frame.loc[: str(day)]
    if len(prefix) < 253 or prefix.index[-1].date() != day:
        raise ValueError("253 exact signal observations required")
    row = prefix.iloc[-1]
    r252 = float(row.mom252)
    rank = r252
    gate = r252
    trend = True
    if variant == "E01":
        rank = float(prefix.tr.iloc[-22] / prefix.tr.iloc[-253] - 1)
        gate = rank
    elif variant == "E02":
        endpoints = prior.calendar_endpoints(day)
        levels = []
        for lag in (0, 1, 3, 6, 12):
            stamp = pd.Timestamp(endpoints[lag])
            if stamp not in prefix.index:
                raise ValueError("calendar score endpoint missing")
            levels.append(float(prefix.tr.loc[stamp]))
        if any(not math.isfinite(v) or v <= 0 for v in levels):
            raise ValueError("invalid calendar endpoint")
        rank = math.fsum(levels[0] / v - 1 for v in levels[1:]) / 4
        gate = rank
    elif variant == "E03":
        volatility = float(row.vol63)
        if not math.isfinite(volatility) or volatility < 0:
            raise ValueError("volatility must be finite and nonnegative before flooring")
        rank = r252 / max(volatility, 0.01)
    elif variant == "E04":
        rank = float(row.reg252)
    elif variant == "E06":
        tr, average = float(row.tr), float(row.sma200)
        if not math.isfinite(tr) or not math.isfinite(average) or tr <= 0 or average <= 0:
            raise ValueError("invalid long-trend input")
        trend = tr > average
    return Score(rank=rank, r252=r252, gate_value=gate, above_trend=trend)


@beartype
def decision(
    trial: Trial, features: FeatureStore, names: tuple[str, ...], groups: GroupMap, day: date
) -> TargetRecord:
    if trial.variant == "BASE252":
        return base_decision(role_for(trial), features, names, groups, day)
    prior.calendar_endpoints(day)
    if len(names) != len(set(names)) or "SHY" in names:
        raise ValueError("duplicate or defensive symbol in risk pool")
    scores: dict[str, Score] = {}
    for symbol in (*names, "SHY"):
        if symbol != "SHY":
            groups.group_for(symbol)
        prefix = features.frames[symbol].loc[: str(day)]
        if prefix.empty or prefix.index[-1].date() != day:
            raise ValueError("exact signal close missing")
        if len(prefix) < 253:
            if symbol == "SHY":
                raise ValueError("cash warmup insufficient")
            continue
        row = prefix.iloc[-1]
        values = [float(row[k]) for k in ("raw_close", "turnover20", "positive20")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite qualification input")
        metric = score(trial.variant, prefix, day)
        if symbol == "SHY" or (values[0] >= 5 and values[1] >= 1e7 and values[2] == 20):
            scores[symbol] = metric
    selected: list[str] = []
    used: set[str] = set()
    for symbol in sorted((s for s in scores if s != "SHY"), key=lambda s: (-scores[s].rank, s)):
        group = groups.group_for(symbol)
        if group in used:
            continue
        selected.append(symbol)
        used.add(group)
        if len(selected) == 2:
            break
    hurdle = (
        max(scores["SHY"].gate_value, 0.0) if trial.variant == "E05" else scores["SHY"].gate_value
    )
    q = {s: 0.45 for s in selected if scores[s].gate_value > hurdle and scores[s].above_trend}
    q["SHY"] = max(0.0, 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


@beartype
def raw_oracle(
    trial: Trial, datasets: list[Dataset], names: tuple[str, ...], groups: GroupMap, day: date
) -> TargetRecord:
    """Independent raw arithmetic, OLS and sample volatility, without feature scores."""
    if trial.variant == "BASE252":
        return base_oracle(role_for(trial), datasets, names, groups, day)
    calendar = xcals.get_calendar("XNYS")
    stamp = pd.Timestamp(day)
    if calendar.date_to_session(stamp + pd.offsets.MonthEnd(0), direction="previous").date() != day:
        raise ValueError("raw oracle requires actual month-end")
    frames = {d.symbol: d.frame.loc[: str(day)] for d in datasets}
    if len(frames) != len(datasets):
        raise ValueError("duplicate raw datasets")
    # Store raw rank, independently computed gate, trend, group.
    candidates: list[tuple[float, str, float, bool, str]] = []
    cash: float | None = None
    for symbol in (*names, "SHY"):
        frame = frames[symbol]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("raw exact signal day missing")
        if len(frame) < 253:
            if symbol == "SHY":
                raise ValueError("raw cash warmup insufficient")
            continue
        levels = [float(v) for v in frame.tr.iloc[-253:]]
        if any(not math.isfinite(v) or v <= 0 for v in levels):
            raise ValueError("raw TR nonfinite or nonpositive")
        r252 = levels[-1] / levels[0] - 1
        ranking, gate, trend = r252, r252, True
        if trial.variant == "E01":
            ranking = levels[-22] / levels[0] - 1
            gate = ranking
        elif trial.variant == "E02":
            returns = []
            for months in (1, 3, 6, 12):
                shifted = stamp - pd.DateOffset(months=months)
                endpoint = calendar.date_to_session(
                    shifted + pd.offsets.MonthEnd(0), direction="previous"
                )
                if endpoint not in frame.index:
                    raise ValueError("raw calendar lag endpoint missing")
                denominator = float(frame.tr.loc[endpoint])
                if not math.isfinite(denominator) or denominator <= 0:
                    raise ValueError("raw calendar endpoint invalid")
                returns.append(levels[-1] / denominator - 1)
            ranking = sum(returns) / 4
            gate = ranking
        elif trial.variant == "E03":
            daily = [
                right / left - 1 for left, right in zip(levels[-64:-1], levels[-63:], strict=True)
            ]
            volatility = stdev(daily) * math.sqrt(252)
            if not math.isfinite(volatility) or volatility < 0:
                raise ValueError("raw volatility invalid")
            ranking = r252 / max(volatility, 0.01)
        elif trial.variant == "E04":
            y = np.log(np.asarray(levels[-252:], dtype=float))
            x = np.arange(252, dtype=float)
            centered_x, centered_y = x - np.mean(x), y - np.mean(y)
            xx = math.fsum(float(v) * float(v) for v in centered_x)
            yy = math.fsum(float(v) * float(v) for v in centered_y)
            xy = math.fsum(float(a) * float(b) for a, b in zip(centered_x, centered_y, strict=True))
            ranking = 0.0 if yy == 0 else 252 * (xy / xx) * min(1.0, xy * xy / (xx * yy))
        elif trial.variant == "E06":
            trend = levels[-1] > math.fsum(levels[-200:]) / 200
        if not math.isfinite(ranking) or not math.isfinite(gate):
            raise ValueError("raw score nonfinite")
        if symbol == "SHY":
            cash = gate
            continue
        recent = frame.iloc[-20:]
        turnovers, volumes = [float(v) for v in recent.turnover], [float(v) for v in recent.volume]
        close = float(frame.close.iloc[-1])
        if any(not math.isfinite(v) for v in (*turnovers, *volumes, close)):
            raise ValueError("raw qualification nonfinite")
        if close >= 5 and math.fsum(turnovers) / 20 >= 1e7 and all(v > 0 for v in volumes):
            candidates.append((ranking, symbol, gate, trend, groups.group_for(symbol)))
    if cash is None:
        raise ValueError("raw cash score missing")
    picks: list[tuple[str, float, bool]] = []
    occupied: set[str] = set()
    for _, symbol, gate, trend, group in sorted(candidates, key=lambda row: (-row[0], row[1])):
        if group in occupied:
            continue
        picks.append((symbol, gate, trend))
        occupied.add(group)
        if len(picks) == 2:
            break
    threshold = max(cash, 0.0) if trial.variant == "E05" else cash
    q = {symbol: 0.45 for symbol, gate, trend in picks if gate > threshold and trend}
    q["SHY"] = max(0.0, 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


@contextmanager
def scoped_hooks(trial: Trial, groups: GroupMap) -> Iterator[None]:
    namespace = native.__dict__
    previous = namespace["target"], namespace["oracle"]
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
        namespace.update(target=previous[0], oracle=previous[1])


def prepared_for(root: Path, trial: Trial) -> Prepared:
    pools, groups = configuration(root)
    return group_prepared(root, role_for(trial), pools, groups)


def expected_registry(root: Path) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = [
        {
            "key": t.key,
            "variant": t.variant,
            "rule": "C11",
            "k": 2,
            "cost_bps": t.cost_bps,
            "cohort": t.cohort,
            "path": str(root / OUT / t.key),
            "mode": "new_native",
            "source_hashes": {},
        }
        for t in catalog(root)
        if t.kind == "science"
    ]
    for account in prior.references(root):
        if account.rule not in ("C11", "SPY_BH", "QQQ_BH", "SPY60_IEF40"):
            continue
        path = prior.public_root(root) / "standard/reference" / account.key
        roles.append(
            {
                "key": account.key,
                "variant": "BASE252" if account.rule == "C11" else "benchmark",
                "rule": account.rule,
                "k": 2 if account.rule == "C11" else None,
                "cost_bps": account.cost_bps,
                "cohort": account.cohort,
                "path": str(path),
                "mode": "reuse",
                "source_hashes": {str(path / n): h for n, h in account.reference_files.items()},
            }
        )
    full_path = root / prior.OUT / "FULL_AUDIT.json"
    full = json.loads(full_path.read_text())
    if full["status"] != "PASS":
        raise ValueError("completed frozen13612W controls required")
    verify_hashes(full["hashes"])
    for previous in prior.catalog(root):
        if previous.kind != "science" or previous.k != 2:
            continue
        path = root / prior.OUT / previous.key
        receipt_path = path / "VERIFIED.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["status"] != "PASS" or receipt["key"] != previous.key:
            raise ValueError("13612W reuse contract mismatch")
        verify_hashes(receipt["hashes"])
        hashes = {
            **receipt["hashes"],
            str(receipt_path): sha(receipt_path),
            str(full_path): sha(full_path),
        }
        roles.append(
            {
                "key": previous.key,
                "variant": "W13612",
                "rule": "C11",
                "k": 2,
                "cost_bps": previous.cost_bps,
                "cohort": previous.cohort,
                "path": str(path),
                "mode": "reuse",
                "source_hashes": hashes,
            }
        )
    if len(roles) != 66 or len({r["key"] for r in roles}) != 66:
        raise ValueError("exact66 reported roles required")
    return roles


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
        if state == "started" and sum(r.state == "started" for r in records) >= 48:
            raise ValueError("frozen48-native-attempt cap exhausted, including failures/fixtures")
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
        g: tuple(s for s in names if s in pools.historical34) for g, names in groups.groups.items()
    }
    if (
        protocol["id"] != IDENTITY
        or tuple(protocol["pool"]) != pools.historical34
        or {g: tuple(v) for g, v in protocol["groups"].items()} != projected
    ):
        raise ValueError("protocol pool/groups/identity changed")
    registry = expected_registry(root)
    accepted = verify_joint_acceptance(root)
    provenance = json.loads(
        (root / "inputs_etf_group_cap_20261109/SOURCE_PROVENANCE.json").read_text()
    )["hashes"]
    files = [
        p
        for base in (
            root / "src/rotbt",
            root / "scripts",
            prior.public_root(root) / "src",
            prior.public_root(root) / "standard",
        )
        for p in base.rglob("*")
        if p.is_file()
        and (p.suffix == ".py" or "standard" in p.parts)
        and "__pycache__" not in p.parts
    ]
    files += list((root / "tests").glob("test_etf_momentum_variants*.py"))
    files += [root / RESEARCH / name for name in ("PROTOCOL.json", "CATALOG.json")]
    hashes = {
        **protocol["source_hashes"],
        **accepted,
        **provenance,
        **{str(p): sha(p) for p in files},
    }
    for row in registry:
        hashes.update(row["source_hashes"])
    freeze(root / RESEARCH / "RUN_LOCK.json", hashes)
    output = root / OUT
    output.mkdir(parents=True, exist_ok=True)
    path = output / "REGISTRY.json"
    if path.exists():
        if json.loads(path.read_text()) != {"roles": registry}:
            raise ValueError("immutable66-role registry differs")
    else:
        write(path, {"roles": registry})


def verify_run_lock(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = json.loads((root / RESEARCH / "RUN_LOCK.json").read_text())["hashes"]
    verify_hashes(hashes)
    return hashes


def causality(root: Path) -> None:
    """Rebuild all used features from raw prefixes at each registered monthly close."""
    locked = verify_run_lock(root)
    _, groups = configuration(root)
    trials = catalog(root)
    fields = ("tr", "mom252", "vol63", "reg252", "sma200", "raw_close", "turnover20", "positive20")
    endpoint_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    continuous = next(
        t for t in trials if t.kind == "science" and t.cohort == "continuous2022_2026"
    )
    historical = next(
        t for t in trials if t.kind == "science" and t.cohort == "historical2022_2025"
    )
    prepared = prepared_for(root, continuous)
    historical_prepared = prepared_for(root, historical)
    days = (date(2021, 12, 31), *native.decision_days(contract_for(continuous)))
    max_feature_difference = 0.0
    for number, day in enumerate(days, 1):
        prefix = [clip_dataset(d, day) for d in prepared.datasets]
        rebuilt = features_for(prefix)
        endpoints = prior.calendar_endpoints(day)
        for symbol in prepared.features.frames:
            raw_prefix = next(d.frame for d in prefix if d.symbol == symbol)
            required = raw_prefix.index[-253:].union(pd.DatetimeIndex(list(endpoints.values())))
            candidates = [prepared.features.frames[symbol]]
            if day <= historical_prepared.last:
                candidates.append(historical_prepared.features.frames[symbol])
            for full in candidates:
                actual = full.loc[pd.Timestamp(day), list(fields)].to_numpy(dtype=float)
                expected = rebuilt.frames[symbol].iloc[-1][list(fields)].to_numpy(dtype=float)
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=False,
                )
                max_feature_difference = max(
                    max_feature_difference, float(np.max(np.abs(actual - expected)))
                )
                pd.testing.assert_series_equal(
                    full.tr.loc[required], rebuilt.frames[symbol].tr.loc[required], check_exact=True
                )
        for variant in VARIANTS:
            trial = next(
                t
                for t in trials
                if t.variant == variant and t.cost_bps == 5 and t.cohort == "continuous2022_2026"
            )
            target = decision(trial, prepared.features, prepared.risk_names, groups, day)
            recalculated = decision(trial, rebuilt, prepared.risk_names, groups, day)
            raw = raw_oracle(trial, prefix, prepared.risk_names, groups, day)
            if target.q.keys() != recalculated.q.keys() or target.q.keys() != raw.q.keys():
                raise ValueError("prefix/full/raw target asset mismatch")
            differences = [abs(w - recalculated.q[s]) for s, w in target.q.items()] + [
                abs(w - raw.q[s]) for s, w in target.q.items()
            ]
            if max(differences) > 1e-12:
                raise ValueError("prefix/full/raw target weight mismatch")
            observations.append(
                {
                    "day": day,
                    "variant": variant,
                    "full": target.q,
                    "prefix": recalculated.q,
                    "raw": raw.q,
                    "max_target_difference": max(differences),
                }
            )
        endpoint_rows.append({"day": day, "calendar_endpoints": endpoints})
        if number % 10 == 0 or number == len(days):
            print(
                f"CAUSALITY {number}/{len(days)} closes; {len(observations)} variant targets",
                flush=True,
            )
    verify_hashes(locked)
    value = {
        "status": "PASS",
        "identity": IDENTITY,
        "prefix_rebuilds": len(endpoint_rows),
        "variant_targets": len(observations),
        "used_fields": fields,
        "max_feature_difference": max_feature_difference,
        "observations": observations,
        "monthly_closes": endpoint_rows,
        "run_lock_sha256": sha(root / RESEARCH / "RUN_LOCK.json"),
        "native_runs": 0,
    }
    path = root / RESEARCH / "CAUSALITY_RECEIPT.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value, default=str)):
            raise ValueError("immutable causality receipt differs")
    else:
        write(path, value)


def validate_saved(root: Path, trial: Trial, prepared: Prepared | None, groups: GroupMap) -> None:
    if prepared is None:
        raise ValueError("actual35-feed prepared inputs required")
    folder = root / OUT / trial.key
    artifact = native.Artifact.model_validate_json((folder / "ARTIFACT.json").read_text())
    if (
        Trial.model_validate_json((folder / "TRIAL.json").read_text()) != trial
        or artifact.contract != contract_for(trial)
        or native.Contract.model_validate_json((folder / "CONTRACT.json").read_text())
        != contract_for(trial)
    ):
        raise ValueError("saved variant identity/execution shell differs")
    if json.loads((folder / "NATIVE_COMPLETE.json").read_text()) != {
        "artifact_sha256": sha(folder / "ARTIFACT.json")
    }:
        raise ValueError("saved artifact lacks completed fault-free native proof")
    settings = settings_for(root, role_for(trial))
    with scoped_hooks(trial, groups):
        native.audit(artifact, prepared, settings)
    # audit() already proves the exact signal calendar; require each ordinary fill at its next open.
    calendar = xcals.get_calendar("XNYS")
    execution_days = {calendar.next_session(pd.Timestamp(t.day)).date() for t in artifact.targets}
    for order in artifact.account.orders:
        if (
            order.reason not in {"terminal", "annual_liquidation"}
            and order.day not in execution_days
        ):
            raise ValueError("ordinary order is not the next session of a registered signal")
    with TemporaryDirectory(prefix="momentum-variant-books-") as temporary:
        generated = Path(temporary)
        native.export_books(generated, artifact)
        for book in native.BOOKS:
            name = f"{book}.csv.gz"
            if not (folder / name).exists():
                shutil.copyfile(generated / name, folder / name)
            native.compare_book_frames(
                pd.read_csv(generated / name, float_precision="round_trip"),
                pd.read_csv(folder / name, float_precision="round_trip"),
            )
    outputs: dict[str, Any] = {
        "TARGETS.json": [t.model_dump(mode="json") for t in artifact.targets],
        "NATIVE_TOTAL_RETURN.json": {"native_total_return": artifact.account.native_total_return},
    }
    if trial.cohort == "historical2022_2025":
        outputs["METRICS.json"] = [
            m.model_dump(mode="json") for m in analyze(artifact.account, settings)
        ]
    else:
        outputs["YTD.json"] = describe(
            artifact.account, continuous=trial.cohort == "continuous2022_2026"
        ).model_dump(mode="json")
    for name, value in outputs.items():
        path = folder / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"saved native scalar/targets/metrics mismatch: {name}")
        else:
            write(path, value)
    if trial.kind == "bridge":
        bridge = prior.Trial(
            key=trial.key, k=2, cost_bps=5, cohort=trial.cohort, kind="bridge", score="original252"
        )
        prior.compare_bridge(root, bridge, artifact, folder)
    path = folder / "VERIFIED.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        if (
            receipt["status"] != "PASS"
            or receipt["key"] != trial.key
            or receipt["variant"] != trial.variant
        ):
            raise ValueError("saved verification identity differs")
        verify_hashes(receipt["hashes"])
    else:
        write(
            path,
            {
                "status": "PASS",
                "key": trial.key,
                "variant": trial.variant,
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
        raise ValueError(f"partial/failed native evidence preserved; no automatic retry: {folder}")
    if trial not in catalog(root) or prepared is None:
        raise ValueError("registered variant and prepared inputs required")
    if (
        len(prepared.datasets) != 35
        or prepared.last != contract_for(trial).through
        or tuple(sorted(d.symbol for d in prepared.datasets))
        != tuple(sorted((*prepared.risk_names, "SHY")))
    ):
        raise ValueError("actual feed membership or physical cutoff differs")
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
        if (
            receipt["status"] != "PASS"
            or receipt["key"] != trial.key
            or receipt["variant"] != trial.variant
        ):
            raise ValueError("stage missing a verified frozen variant")
        verify_hashes(receipt["hashes"])
        hashes.update(receipt["hashes"])
        hashes[str(path)] = sha(path)
    registry = expected_registry(root)
    registry_path = root / OUT / "REGISTRY.json"
    if json.loads(registry_path.read_text()) != {"roles": registry}:
        raise ValueError("registry differs from exact66 frozen economic roles")
    for row in registry:
        hashes.update(row["source_hashes"])
    hashes[str(registry_path)] = sha(registry_path)
    sources = verify_run_lock(root)
    counts: dict[str, int] = {}
    if stage == "science":
        causal = root / RESEARCH / "CAUSALITY_RECEIPT.json"
        receipt = json.loads(causal.read_text())
        if (
            receipt["status"] != "PASS"
            or receipt["identity"] != IDENTITY
            or receipt["run_lock_sha256"] != sha(root / RESEARCH / "RUN_LOCK.json")
        ):
            raise ValueError("accepted prefix causal audit required")
        hashes[str(causal)] = sha(causal)
        records = [
            Attempt.model_validate_json(line) for line in journal(root).read_text().splitlines()
        ]
        starts = [r.key for r in records if r.state == "started"]
        if set(starts) != {t.key for t in trials} or not 38 <= len(starts) <= 48:
            raise ValueError("actual native journal differs from38 frozen paths/hard48budget")
        hashes[str(journal(root))] = sha(journal(root))
        counts = {
            s: sum(r.state == s for r in records)
            for s in ("started", "native_completed", "finished", "failed", "revalidated")
        }
    hashes.update(sources)
    hashes[str(root / RESEARCH / "RUN_LOCK.json")] = sha(root / RESEARCH / "RUN_LOCK.json")
    value = {
        "status": "PASS",
        "keys": [t.key for t in selected],
        "science": 36 if stage == "science" else 0,
        "bridges": 2,
        "reuse_roles": 30,
        "report_roles": 66,
        "journal": counts,
        "hashes": hashes,
    }
    path = root / OUT / ("FULL_AUDIT.json" if stage == "science" else "BRIDGE_AUDIT.json")
    if not path.exists():
        write(path, value)
    elif json.loads(path.read_text()) != value:
        raise ValueError("immutable stage acceptance differs")


def run(root: Path, stage: Literal["bridge", "science"]) -> None:
    root = root.resolve()
    verify_run_lock(root)
    _, groups = configuration(root)
    if stage == "science":
        receipt = json.loads((root / OUT / "BRIDGE_AUDIT.json").read_text())
        if receipt["status"] != "PASS" or receipt["keys"] != [
            t.key for t in catalog(root) if t.kind == "bridge"
        ]:
            raise ValueError("both fixed bridges must pass before science")
        verify_hashes(receipt["hashes"])
        causal = json.loads((root / RESEARCH / "CAUSALITY_RECEIPT.json").read_text())
        if (
            causal["status"] != "PASS"
            or causal["identity"] != IDENTITY
            or causal["run_lock_sha256"] != sha(root / RESEARCH / "RUN_LOCK.json")
        ):
            raise ValueError("raw-prefix feature causality acceptance required")
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
    parser.add_argument(
        "--stage", choices=("prepare", "causality", "bridge", "science"), required=True
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root)
    elif args.stage == "causality":
        causality(args.root.resolve())
    else:
        run(args.root, args.stage)


if __name__ == "__main__":
    main()
