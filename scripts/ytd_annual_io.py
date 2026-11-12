"""Independent source locks for fixed annual2026 continuation, never live data."""

from pathlib import Path
from typing import Literal

from beartype import beartype
from pydantic import ConfigDict

from c10standard.io import ROOT as ROOT
from c10standard.native import write
from c10standard.run import sha
from rotbt.models import Model

STUDY = Path("experiments/annual-2026")


class SourceLock(Model):
    experiment: Literal["ANNUAL_2026_CONTINUOUS_V1"] = "ANNUAL_2026_CONTINUOUS_V1"
    hashes: dict[str, str]


class Acceptance(Model):
    model_config = ConfigDict(extra="ignore")
    status: Literal["PASS_CONDITIONAL_SOURCE_AND_MARKET"]
    schedule_sha256: str
    manifest_sha256: str
    through: Literal["2026-11-10"]


def require_acceptance(root: Path) -> None:
    path = root / STUDY / "DATA_ACCEPTANCE.json"
    if not path.exists():
        raise ValueError("data acceptance required before source freeze or execution")
    acceptance = Acceptance.model_validate_json(path.read_bytes())
    if (acceptance.schedule_sha256 != sha(root / STUDY / "SCHEDULE.json")
            or acceptance.manifest_sha256 != sha(root / STUDY / "data/science/MANIFEST.json")):
        raise ValueError("data acceptance is not bound to current schedule/manifest")


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
    for directory in ("src", "tests", "typings", str(STUDY / "tests")):
        paths.update(p for p in (root / directory).rglob("*") if p.suffix in (".py", ".pyi"))
    for prefix in ("ytd_annual_", "annual_", "v2_2023_"):
        paths.update((root / "scripts").glob(f"{prefix}*.py"))
    for data_directory in (
        root / STUDY / "data",
        root / STUDY / "universe",
        root / "experiments/v2-annual-repool/data",
        root / "experiments/v2-annual-repool/reference",
    ):
        paths.update(p for p in data_directory.rglob("*") if p.is_file())
    for rule in ("SPY_BH", "QQQ_BH"):
        for cost in (5, 20):
            ref_directory = root / "standard/reference" / f"{rule}_{cost}bp_historical2022_2025"
            paths.update(p for p in ref_directory.rglob("*") if p.is_file())
    for n in ("SCHEDULE.json", "DATA_ACCEPTANCE.json"):
        path = root / STUDY / n
        if path.exists():
            paths.add(path)
    return SourceLock(hashes={p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)})


@beartype
def verify_lock(root: Path = ROOT) -> str:
    require_acceptance(root)
    path = root / STUDY / "SOURCE_LOCK.json"
    if SourceLock.model_validate_json(path.read_bytes()) != snapshot(root):
        raise ValueError("annual2026 source/input lock changed")
    return sha(path)


@beartype
def freeze(root: Path = ROOT) -> str:
    require_acceptance(root)
    path = root / STUDY / "SOURCE_LOCK.json"
    if path.exists():
        return verify_lock(root)
    journal = root / STUDY / "evidence/attempts.jsonl"
    if journal.exists() and journal.stat().st_size:
        raise ValueError("cannot freeze after native attempts")
    write(path, snapshot(root))
    return sha(path)
