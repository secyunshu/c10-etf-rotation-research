"""Immutable source/data/schedule lock for one bounded annual-pool experiment."""

from pathlib import Path
from typing import Literal

from beartype import beartype

from c10standard.io import ROOT as ROOT
from c10standard.native import write
from c10standard.run import sha
from rotbt.models import Model

STUDY = Path("experiments/v2-annual-repool")


class SourceLock(Model):
    experiment: Literal["V2_ANNUAL_REPOOL_V1"] = "V2_ANNUAL_REPOOL_V1"
    hashes: dict[str, str]


@beartype
def snapshot(root: Path) -> SourceLock:
    paths = {
        root / n
        for n in (
            "pyproject.toml",
            "standard/STRATEGY.json",
            "standard/ACCOUNTS.json",
            str(STUDY / "PLAN.md"),
            str(STUDY / "PLAN_REVIEW.json"),
        )
    }
    for name in ("src", "tests", "typings", str(STUDY / "tests")):
        paths.update(p for p in (root / name).rglob("*") if p.suffix in (".py", ".pyi"))
    paths.update((root / "scripts").glob("annual_*.py"))
    paths.update((root / "scripts").glob("v2_2023_*.py"))
    for directory in (root / STUDY / "data", root / STUDY / "universe"):
        paths.update(p for p in directory.rglob("*") if p.is_file())
    for rule in ("C11", "SPY_BH", "QQQ_BH"):
        for cost in (5, 20):
            folder = root / "standard/reference" / f"{rule}_{cost}bp_historical2022_2025"
            paths.update(p for p in folder.rglob("*") if p.is_file())
    for name in ("FIXED.json", "ANNUAL.json", "DATA_ACCEPTANCE.json"):
        p = root / STUDY / name
        if p.exists():
            paths.add(p)
    return SourceLock(hashes={p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)})


@beartype
def verify_lock(root: Path = ROOT) -> str:
    path = root / STUDY / "SOURCE_LOCK.json"
    if SourceLock.model_validate_json(path.read_bytes()) != snapshot(root):
        raise ValueError("annual source/data/schedule lock changed")
    return sha(path)


@beartype
def freeze(root: Path = ROOT) -> str:
    path = root / STUDY / "SOURCE_LOCK.json"
    if path.exists():
        return verify_lock(root)
    journal = root / STUDY / "evidence/attempts.jsonl"
    if journal.exists() and journal.stat().st_size:
        raise ValueError("cannot freeze after native attempts")
    write(path, snapshot(root))
    return sha(path)
