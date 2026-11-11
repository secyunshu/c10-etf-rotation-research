"""Source locks for the bounded 2023-start study, without reading2026 payloads."""

from pathlib import Path
from typing import Literal

from beartype import beartype

from c10standard.io import ROOT
from c10standard.native import write
from c10standard.run import sha
from rotbt.models import Model

STUDY = Path("experiments/v2-2023-start")


class SourceLock(Model):
    experiment: Literal["V2_REPOOL_2023_V1"] = "V2_REPOOL_2023_V1"
    hashes: dict[str, str]


@beartype
def snapshot(root: Path) -> SourceLock:
    paths = {
        root / name
        for name in (
            "pyproject.toml",
            "standard/STRATEGY.json",
            "standard/ACCOUNTS.json",
            str(STUDY / "PLAN.md"),
            str(STUDY / "PLAN_REVIEW.json"),
        )
    }
    for directory in ("src", "tests", "typings", str(STUDY / "tests")):
        paths.update(p for p in (root / directory).rglob("*") if p.suffix in (".py", ".pyi"))
    paths.update((root / "scripts").glob("v2_2023_*.py"))
    for folder in (
        root / STUDY / "data",
        root / STUDY / "universe",
        root / "standard/reference/C11_5bp_historical2022_2025",
        root / "standard/reference/C11_20bp_historical2022_2025",
    ):
        paths.update(p for p in folder.rglob("*") if p.is_file())
    for name in ("NEW_POOL.json", "OLD_POOL.json", "DATA_ACCEPTANCE.json"):
        path = root / STUDY / name
        if path.exists():
            paths.add(path)
    return SourceLock(hashes={p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)})


@beartype
def verify_lock(root: Path = ROOT) -> str:
    path = root / STUDY / "SOURCE_LOCK.json"
    if SourceLock.model_validate_json(path.read_bytes()) != snapshot(root):
        raise ValueError("source/input/pool lock changed")
    return sha(path)


@beartype
def freeze(root: Path = ROOT) -> str:
    path = root / STUDY / "SOURCE_LOCK.json"
    if path.exists():
        return verify_lock(root)
    journal = root / STUDY / "evidence/attempts.jsonl"
    if journal.exists() and journal.stat().st_size:
        raise ValueError("cannot freeze source after native attempts")
    write(path, snapshot(root))
    return sha(path)
