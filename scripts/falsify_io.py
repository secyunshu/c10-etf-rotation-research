"""Independent immutable source lock for finite mechanism counterfactuals."""

from pathlib import Path
from typing import Literal

from beartype import beartype

from c10standard.io import ROOT as ROOT
from c10standard.native import write
from c10standard.run import sha
from rotbt.models import Model

STUDY = Path("experiments/annual-pool-falsification")


class SourceLock(Model):
    experiment: Literal["ANNUAL_POOL_FALSIFY_V1"] = "ANNUAL_POOL_FALSIFY_V1"
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
    for prefix in ("falsify_", "annual_", "v2_2023_"):
        paths.update((root / "scripts").glob(f"{prefix}*.py"))
    for data_directory in (
        root / "experiments/v2-annual-repool/data",
        root / "experiments/v2-annual-repool/reference",
    ):
        paths.update(p for p in data_directory.rglob("*") if p.is_file())
    for rule in ("C11", "SPY_BH", "QQQ_BH"):
        for cost in (5, 20):
            folder = root / "standard/reference" / f"{rule}_{cost}bp_historical2022_2025"
            paths.update(p for p in folder.rglob("*") if p.is_file())
    for name in ("FIXED.json", "ANNUAL.json"):
        file = root / "experiments/v2-annual-repool" / name
        if file.exists():
            paths.add(file)
    case_file = root / STUDY / "CASES.json"
    if case_file.exists():
        paths.add(case_file)
    for name in ("SOURCE_REVIEW.json", "SOURCE_REVIEW.md"):
        evidence = root / STUDY / name
        if evidence.exists():
            paths.add(evidence)
    return SourceLock(hashes={p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)})


@beartype
def verify_lock(root: Path = ROOT) -> str:
    path = root / STUDY / "SOURCE_LOCK.json"
    if SourceLock.model_validate_json(path.read_bytes()) != snapshot(root):
        raise ValueError("falsification source/data/identity lock changed")
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
