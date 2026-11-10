"""Frozen group-cap adapter: reuse native Backtrader accounts, meter every actual run."""

import argparse
import fcntl
import json
import shutil
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import pandas as pd

from rotbt.analysis import analyze
from rotbt.models import (
    ClosedTrade,
    DailyRecord,
    Dataset,
    Distribution,
    HoldingRecord,
    Model,
    OrderRecord,
)
from rotbt.phase2.models import FeatureStore, SignalContext
from scripts import etf_c10_2026_run as native
from scripts.etf_c10_2026_audit import TargetRecord, describe
from scripts.etf_c10_2026_inputs import Prepared
from scripts.etf_c10_2026_inputs import load as load_2026
from scripts.etf_c10_institutional_inputs import load as load_history
from scripts.etf_group_cap_audit import (
    audit_historical_reuse,
    bind_reference,
    compare_books,
    raw_oracle,
    reference_path,
    settings_for,
    sha,
    verify_hashes,
)
from scripts.etf_group_cap_inputs import clip_dataset, features_for
from scripts.etf_group_cap_inputs import load as load_joint
from scripts.etf_group_cap_targets import (
    RESEARCH,
    GroupMap,
    Pool,
    Pools,
    Role,
    decision,
    load_roles,
)
from scripts.etf_large36_return_targets import Rule as LegacyRule

OUT = "etf_group_cap_outputs/20261109_v1"


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def model_write(path: Path, value: Model) -> None:
    write(path, value.model_dump(mode="json"))


def configuration(root: Path) -> tuple[Pools, GroupMap]:
    folder = root / RESEARCH
    pools = Pools.model_validate_json((folder / "POOLS.json").read_text())
    groups = GroupMap.model_validate_json((folder / "GROUPS.json").read_text())
    if len(groups.groups) != 21 or len(pools.legacy36) != 36 or len(pools.historical34) != 34:
        raise ValueError("frozen pool/group cardinalities changed")
    for symbol in set((*pools.legacy36, *pools.historical34)):
        groups.group_for(symbol)
    return pools, groups


def pool_feeds(pools: Pools, pool: Pool) -> tuple[str, ...]:
    names = pools.names(pool)
    if len(names) != len(set(names)) or "SHY" in names:
        raise ValueError("duplicate risk symbol or SHY in risk pool")
    return tuple(sorted((*names, "SHY")))


def contract_for(role: Role) -> native.Contract:
    mode: Literal["bridge", "reset2026", "continuous2022"] = (
        "bridge"
        if role.cohort == "historical2022_2025"
        else "reset2026"
        if role.cohort == "reset2026"
        else "continuous2022"
    )
    return native.Contract(rule=role.rule, cost=role.cost_bps, mode=mode)


def prepared_for(root: Path, role: Role, pools: Pools, groups: GroupMap) -> Prepared:
    cutoff: Literal["2025", "2026"] = "2025" if role.through.year == 2025 else "2026"
    if role.pool == "historical34":
        verify_joint_acceptance(root)
        source = load_joint(root, cutoff).datasets
    elif cutoff == "2025":
        source = load_history(root, "modern38").datasets
    else:
        source = load_2026(root).datasets
    names = pool_feeds(pools, role.pool)
    mapping = {d.symbol: d for d in source}
    if len(mapping) != len(source) or not set(names) <= mapping.keys():
        raise ValueError("missing or duplicate actual pool feed")
    selected = [clip_dataset(mapping[s], role.through) for s in names]
    return Prepared(
        datasets=selected,
        features=features_for(selected),
        risk_names=pools.names(role.pool),
        groups=tuple(tuple(s for s in members if s in names) for members in groups.groups.values()),
        first=min(d.frame.index[0].date() for d in selected),
        last=role.through,
    )


@contextmanager
def scoped_hooks(role: Role, groups: GroupMap) -> Iterator[None]:
    """The old native shell receives this role's targets; all hooks restore on failure."""
    namespace = native.__dict__
    original = namespace["target"], namespace["oracle"], native.feed_names
    economic_groups = groups

    def target(
        rule: LegacyRule | str,
        features: FeatureStore,
        day: date,
        names: tuple[str, ...],
        groups: tuple[tuple[str, ...], ...],
        context: SignalContext | None = None,
    ) -> dict[str, float]:
        return decision(role, features, names, economic_groups, day).q

    def oracle(
        rule: str, datasets: list[Dataset], names: tuple[str, ...], day: date
    ) -> TargetRecord:
        return raw_oracle(role, datasets, names, groups, day)

    def feeds(contract: native.Contract, prepared: Prepared) -> tuple[str, ...]:
        if contract != contract_for(role):
            raise ValueError("execution shell differs from true economic ROLE")
        return tuple(sorted((*prepared.risk_names, "SHY")))

    namespace.update(target=target, oracle=oracle)
    native.feed_names = feeds
    try:
        yield
    finally:
        namespace.update(target=original[0], oracle=original[1])
        native.feed_names = original[2]


def journal(root: Path) -> Path:
    return root / OUT / "NATIVE_ATTEMPTS.jsonl"


def native_count(root: Path) -> int:
    path = journal(root)
    return (
        sum(json.loads(line)["state"] == "started" for line in path.read_text().splitlines())
        if path.exists()
        else 0
    )


def require_budget(root: Path) -> None:
    if native_count(root) >= 64:
        raise ValueError("frozen 64 native-attempt cap exhausted, including failed attempts")


def append_attempt(root: Path, key: str, state: str, error: str = "") -> None:
    with journal(root).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": key,
                    "state": state,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "error": error,
                }
            )
            + "\n"
        )
        handle.flush()


def verify_joint_acceptance(root: Path) -> dict[str, str]:
    output = root / "inputs_etf_group_cap_20261109"
    required = {
        str(output / n)
        for n in ("SOURCE_PROVENANCE.json", "2025/MANIFEST.json", "2026/MANIFEST.json")
    }
    provenance = json.loads((output / "SOURCE_PROVENANCE.json").read_text())["hashes"]
    valid: list[Path] = []
    for path in (root / RESEARCH / "data_qa").glob("*/PASS_CANONICAL_BINDING.json"):
        binding = json.loads(path.read_text())
        passed = path.parent / "PASS.json"
        receipt = json.loads(passed.read_text())
        hashes = binding.get("canonical_hashes", {})
        if (
            binding.get("status") == "PASS"
            and binding.get("source_pass_path") == str(passed)
            and binding.get("source_pass_sha256") == sha(passed)
            and binding.get("source_hashes") == provenance
            and receipt.get("status") == "PASS"
            and receipt.get("old38_exact") is True
            and receipt.get("source_hashes") == provenance
            and set(hashes) == required
        ):
            verify_hashes(hashes)
            valid.append(path)
    if len(valid) != 1:
        raise ValueError("DATA_PASS requires exactly one accepted source/canonical hash binding")
    verify_hashes(provenance)
    files = [valid[0], valid[0].parent / "PASS.json", *(Path(p) for p in required)]
    for cutoff in ("2025", "2026"):
        folder = output / cutoff
        manifest = json.loads((folder / "MANIFEST.json").read_text())
        hashes = {str(folder / n): h for n, h in manifest["output_hashes"].items()}
        verify_hashes(hashes)
        files.extend(Path(p) for p in hashes)
    return {str(p): sha(p) for p in files}


def freeze(path: Path, hashes: dict[str, str]) -> None:
    if path.exists():
        old = json.loads(path.read_text())["hashes"]
        verify_hashes(old)
        if old != hashes:
            raise ValueError("immutable run lock cannot be replaced")
    else:
        write(path, {"hashes": hashes})


def prepare(root: Path, joint: bool = False) -> None:
    root = root.resolve()
    protocol = json.loads((root / RESEARCH / "PROTOCOL.json").read_text())
    verify_hashes(protocol["hashes"])
    configuration(root)
    roles = load_roles(root)
    references = {r.key: bind_reference(root, r) for r in roles if r.mode == "reuse"}
    sources = {p: h for hashes in references.values() for p, h in hashes.items()}
    files = [
        root / RESEARCH / name
        for name in ("PROTOCOL.json", "GROUPS.json", "POOLS.json", "CATALOG_ROLES.json")
    ]
    files += list((root / "src/rotbt").rglob("*.py"))
    files += [
        root / "scripts" / name
        for name in (
            "etf_group_cap_targets.py",
            "etf_group_cap_run.py",
            "etf_group_cap_audit.py",
            "etf_group_cap_report.py",
            "etf_group_cap_inputs.py",
            "etf_c10_2026_run.py",
            "etf_c10_2026_audit.py",
            "etf_c10_2026_inputs.py",
            "etf_c10_institutional_inputs.py",
            "etf_large36_safe_inputs.py",
            "etf_large36_return_targets.py",
            "etf_rotation_legacy.py",
        )
    ]
    for base in ("inputs_c10_institutional_20261108/modern38", "inputs_c10_2026_20261109/modern38"):
        files += [p for p in (root / base).iterdir() if p.is_file()]
    files += list((root / "tests").glob("test_etf_group_cap_*.py"))
    sources.update({str(p): sha(p) for p in files})
    sources.update(protocol["hashes"])
    folder = root / OUT
    folder.mkdir(parents=True, exist_ok=True)
    freeze(folder / "RUN_LOCK.json", sources)
    registry = []
    by_key = {r.key: r for r in roles}
    for role in roles:
        original = by_key[role.alias_of] if role.alias_of else role
        path = reference_path(root, original) if original.mode == "reuse" else folder / original.key
        registry.append(
            {
                "role": role.model_dump(mode="json"),
                "path": str(path),
                "mode": role.mode,
                "alias_of": role.alias_of,
                "source_hashes": references.get(original.key, {}),
            }
        )
    if not (folder / "REGISTRY.json").exists():
        write(folder / "REGISTRY.json", {"roles": registry})
    elif json.loads((folder / "REGISTRY.json").read_text()) != {"roles": registry}:
        raise ValueError("immutable role registry changed")
    if joint:
        freeze(folder / "JOINT_LOCK.json", verify_joint_acceptance(root))


def validate_saved(
    root: Path, role: Role, prepared: Prepared | None, groups: GroupMap, folder: Path, bridge: bool
) -> None:
    if prepared is None:
        raise ValueError("prepared actual pool required")
    artifact = native.Artifact.model_validate_json((folder / "ARTIFACT.json").read_text())
    complete = json.loads((folder / "NATIVE_COMPLETE.json").read_text())
    if complete != {"artifact_sha256": sha(folder / "ARTIFACT.json")}:
        raise ValueError("saved artifact lacks proof of completed native execution")
    if (
        artifact.contract != contract_for(role)
        or Role.model_validate_json((folder / "ROLE.json").read_text()) != role
    ):
        raise ValueError("saved native economic identity mismatch")
    settings = settings_for(root, role)
    with scoped_hooks(role, groups):
        native.audit(artifact, prepared, settings)
    existing = [(folder / f"{n}.csv.gz").exists() for n in native.BOOKS]
    if not any(existing):
        native.export_books(folder, artifact)
    elif not all(existing):
        with TemporaryDirectory(prefix="groupcap-export-") as temporary:
            source = Path(temporary)
            native.export_books(source, artifact)
            for name, exists in zip(native.BOOKS, existing, strict=True):
                if not exists:
                    shutil.copyfile(source / f"{name}.csv.gz", folder / f"{name}.csv.gz")
    rows_by_book: tuple[list[Model], ...] = (
        list(artifact.account.daily),
        list(artifact.account.orders),
        list(artifact.account.holdings),
        list(artifact.account.distributions),
        list(artifact.account.trades),
    )
    models: tuple[type[Model], ...] = (
        DailyRecord,
        OrderRecord,
        HoldingRecord,
        Distribution,
        ClosedTrade,
    )
    for name, rows, model in zip(native.BOOKS, rows_by_book, models, strict=True):
        expected = pd.DataFrame(
            [r.model_dump(mode="json") for r in rows], columns=list(model.model_fields)
        )
        native.compare_book_frames(
            expected, pd.read_csv(folder / f"{name}.csv.gz", float_precision="round_trip")
        )
    if bridge:
        bind_reference(root, role)
        compare_books(reference_path(root, role), folder)
        if role.cohort == "historical2022_2025":
            native.compare_reference(root, folder, artifact)
        else:
            old = native.Artifact.model_validate_json(
                (reference_path(root, role) / "ARTIFACT.json").read_text()
            )
            if (
                artifact.targets != old.targets
                or artifact.native_returns != old.native_returns
                or artifact.open_trades != old.open_trades
            ):
                raise ValueError("continuous bridge native target/return/trade mismatch")
    if role.cohort == "historical2022_2025":
        path = folder / "METRICS.json"
        result: Any = [m.model_dump(mode="json") for m in analyze(artifact.account, settings)]
    else:
        path = folder / "YTD.json"
        result = describe(
            artifact.account, continuous=role.cohort == "continuous2022_2026"
        ).model_dump(mode="json")
    if not path.exists():
        write(path, result)
    elif json.loads(path.read_text()) != result:
        raise ValueError("saved metrics disagree with audited native account")
    if not (folder / "VERIFIED.json").exists():
        write(
            folder / "VERIFIED.json",
            {
                "status": "PASS",
                "role": role.key,
                "bridge": bridge,
                "hashes": {
                    str(p): sha(p)
                    for p in folder.iterdir()
                    if p.is_file() and p.name != "FAILURE.txt"
                },
            },
        )
    else:
        verify_hashes(json.loads((folder / "VERIFIED.json").read_text())["hashes"])


def execute(
    root: Path, role: Role, prepared: Prepared | None, groups: GroupMap, *, bridge: bool
) -> None:
    folder = root / OUT / (("bridge_" if bridge else "") + role.key)
    if folder.exists():
        if (folder / "ARTIFACT.json").exists():
            previously_verified = (folder / "VERIFIED.json").exists()
            validate_saved(root, role, prepared, groups, folder, bridge)
            if not previously_verified:
                append_attempt(root, folder.name, "revalidated")
            return
        raise ValueError(f"partial failed native evidence preserved; no automatic rerun: {folder}")
    if prepared is None:
        raise ValueError("prepared actual pool required")
    require_budget(root)
    folder.mkdir(parents=True, exist_ok=False)
    model_write(folder / "ROLE.json", role)
    model_write(folder / "CONTRACT.json", contract_for(role))
    settings = settings_for(root, role)
    try:
        with scoped_hooks(role, groups):
            append_attempt(root, folder.name, "started")
            native._native(prepared, contract_for(role), settings, folder)  # pyright: ignore[reportPrivateUsage]
        write(folder / "NATIVE_COMPLETE.json", {"artifact_sha256": sha(folder / "ARTIFACT.json")})
        append_attempt(root, folder.name, "native_completed")
        validate_saved(root, role, prepared, groups, folder, bridge)
    except BaseException as error:
        with (folder / "FAILURE.txt").open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        append_attempt(root, folder.name, "failed", str(error))
        raise
    append_attempt(root, folder.name, "finished")
    print(f"PASS {folder.name}", flush=True)


def bridge_roles(root: Path) -> list[Role]:
    return [
        r
        for r in load_roles(root)
        if r.mode == "reuse"
        and r.cost_bps == 5
        and (
            (r.top_k == 1 and r.cohort == "historical2022_2025")
            or (r.top_k == 2 and r.cohort == "continuous2022_2026")
        )
    ]


def finalize(root: Path, pools: Pools, groups: GroupMap) -> bool:
    """Audit reuse and aliases after all 42 science paths exist; never invokes native."""
    folder = root / OUT
    roles = load_roles(root)
    if any(
        not (folder / r.key / "VERIFIED.json").exists() for r in roles if r.mode == "new_native"
    ):
        return False
    registry_path = folder / "REGISTRY.json"
    registry = json.loads(registry_path.read_text())["roles"]
    if len(registry) != 72 or [entry["role"] for entry in registry] != [
        r.model_dump(mode="json") for r in roles
    ]:
        raise ValueError("final registry differs from frozen 72 economic roles")
    hashes: dict[str, str] = {}
    cache: dict[bool, Prepared] = {}
    for role, entry in zip(roles, registry, strict=True):
        path = Path(entry["path"])
        if role.mode == "reuse":
            expected = bind_reference(root, role)
            if entry["source_hashes"] != expected or path != reference_path(root, role):
                raise ValueError("reused economic account source binding changed")
            key = role.through.year == 2025
            if key not in cache:
                cache[key] = prepared_for(root, role, pools, groups)
            prepared = cache[key]
            if key:
                audit_historical_reuse(root, role, prepared.datasets, prepared.risk_names, groups)
            else:
                artifact = native.Artifact.model_validate_json((path / "ARTIFACT.json").read_text())
                with scoped_hooks(role, groups):
                    native.audit(artifact, prepared, settings_for(root, role))
            hashes.update(expected)
        elif role.mode == "new_native":
            if path != folder / role.key:
                raise ValueError("new economic path registry mismatch")
            receipt = json.loads((path / "VERIFIED.json").read_text())
            if receipt.get("status") != "PASS" or receipt.get("role") != role.key:
                raise ValueError("new account verification identity mismatch")
            verify_hashes(receipt["hashes"])
            hashes.update(receipt["hashes"])
            hashes[str(path / "VERIFIED.json")] = sha(path / "VERIFIED.json")
        else:
            original = next(e for e in registry if e["role"]["key"] == role.alias_of)
            if (
                path != Path(original["path"])
                or entry["source_hashes"] != original["source_hashes"]
            ):
                raise ValueError("top-one alias must point to its exact original economic path")
    for role in bridge_roles(root):
        path = folder / ("bridge_" + role.key)
        receipt = json.loads((path / "VERIFIED.json").read_text())
        verify_hashes(receipt["hashes"])
        hashes.update(receipt["hashes"])
        hashes[str(path / "VERIFIED.json")] = sha(path / "VERIFIED.json")
    for name in ("REGISTRY.json", "RUN_LOCK.json", "JOINT_LOCK.json", "NATIVE_ATTEMPTS.jsonl"):
        hashes[str(folder / name)] = sha(folder / name)
    for name in ("RUN_LOCK.json", "JOINT_LOCK.json"):
        verify_hashes(json.loads((folder / name).read_text())["hashes"])
    attempts = [json.loads(line) for line in journal(root).read_text().splitlines()]
    planned = {r.key for r in roles if r.mode == "new_native"} | {
        "bridge_" + r.key for r in bridge_roles(root)
    }
    summary = attempt_summary(attempts, planned)
    receipt = {
        "status": "PASS",
        "roles": 72,
        "economic_paths": 60,
        "reuse": 18,
        "aliases": 12,
        "new_science_native": 42,
        "bridge_native": 2,
        **summary,
        "hashes": hashes,
    }
    target = folder / "FULL_AUDIT.json"
    if target.exists():
        if json.loads(target.read_text()) != receipt:
            raise ValueError("immutable FULL_AUDIT differs")
    else:
        write(target, receipt)
    return True


def attempt_summary(attempts: list[dict[str, Any]], planned: set[str]) -> dict[str, int]:
    started = [a["key"] for a in attempts if a["state"] == "started"]
    if set(started) != planned or not 44 <= len(started) <= 64:
        raise ValueError("final journal exceeds frozen native budget or omits/adds economic paths")
    return {
        "native_started": len(started),
        **{
            f"native_{state}": sum(a["state"] == state for a in attempts)
            for state in ("finished", "failed", "revalidated")
        },
    }


def run(
    root: Path,
    stage: Literal["bridge", "science"],
    pool: Literal["legacy36", "historical34", "all"] = "all",
) -> None:
    root = root.resolve()
    folder = root / OUT
    verify_hashes(json.loads((folder / "RUN_LOCK.json").read_text())["hashes"])
    pools, groups = configuration(root)
    roles = bridge_roles(root)
    if len(roles) != 2:
        raise ValueError("exactly two frozen bridges required")
    if stage == "science":
        for role in roles:
            receipt = folder / ("bridge_" + role.key) / "VERIFIED.json"
            verify_hashes(json.loads(receipt.read_text())["hashes"])
        if pool != "legacy36":
            verify_hashes(json.loads((folder / "JOINT_LOCK.json").read_text())["hashes"])
            verify_joint_acceptance(root)
        roles = [
            r
            for r in load_roles(root)
            if r.mode == "new_native" and (pool == "all" or r.pool == pool)
        ]
    with (folder / ".native.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache: dict[tuple[Pool, bool], Prepared] = {}
        for role in roles:
            key = role.pool, role.through.year == 2025
            if key not in cache:
                cache[key] = prepared_for(root, role, pools, groups)
            execute(root, role, cache[key], groups, bridge=stage == "bridge")
        verify_hashes(json.loads((folder / "RUN_LOCK.json").read_text())["hashes"])
        if stage == "science":
            finalize(root, pools, groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=("prepare", "bridge", "science"), required=True)
    parser.add_argument("--pool", choices=("legacy36", "historical34", "all"), default="all")
    parser.add_argument("--joint", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root, args.joint)
    else:
        run(args.root, args.stage, args.pool)


if __name__ == "__main__":
    main()
