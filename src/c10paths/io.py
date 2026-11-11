"""Fixed trial identities and portable, immutable source/input bindings."""

from pathlib import Path
from typing import Final, Literal

from pydantic import ConfigDict, model_validator

from c10standard.models import COHORTS, Contract
from c10standard.native import write
from c10standard.run import sha
from rotbt.models import Model

from .models import VARIANTS

EXPERIMENT = Path("experiments/wechat-momentum-paths")
IDENTITY: Final = "WECHAT_MOMENTUM_PATHS_V1"


class Trial(Model):
    model_config = ConfigDict(frozen=True)
    variant: Literal["BASE", "P01", "P02", "P03"]
    contract: Contract

    @model_validator(mode="after")
    def only_c11(self) -> "Trial":
        if self.contract.rule != "C11":
            raise ValueError("path trials require the unchanged C11 account contract")
        return self

    @property
    def key(self) -> str:
        return f"{self.variant}_{self.contract.cost}bp_{self.contract.cohort}"


class SourceLock(Model):
    experiment: Literal["WECHAT_MOMENTUM_PATHS_V1"] = IDENTITY
    hashes: dict[str, str]


class Binding(Model):
    experiment: Literal["WECHAT_MOMENTUM_PATHS_V1"] = IDENTITY
    trial: Trial
    source_lock_sha256: str


def catalog() -> tuple[Trial, ...]:
    return (
        Trial(variant="BASE", contract=Contract()),
        Trial(variant="BASE", contract=Contract(cost_bps=20, cohort="continuous2022_2026")),
        *(Trial(variant=v, contract=Contract(cost_bps=cost, cohort=cohort))
          for v in VARIANTS for cost in (5, 20) for cohort in COHORTS),
    )


def folder(output: Path, trial: Trial) -> Path:
    return output / trial.variant / trial.contract.key


def journal_path(root: Path) -> Path:
    return root / EXPERIMENT / "evidence/attempts.jsonl"


def source_snapshot(root: Path) -> SourceLock:
    fixed = [
        "pyproject.toml", "standard/STRATEGY.json", "standard/ACCOUNTS.json",
        str(EXPERIMENT / "PLAN.md"), str(EXPERIMENT / "PLAN_REVIEW.json"),
    ]
    paths = {root / name for name in fixed}
    for directory in ("src", "tests", "typings"):
        paths.update(p for p in (root / directory).rglob("*") if p.suffix in (".py", ".pyi"))
    for directory in ("standard/data", "standard/reference"):
        paths.update(p for p in (root / directory).rglob("*") if p.is_file())
    for name in ("ARTICLE_REVIEW.md", "ARTICLE_EVIDENCE.json"):
        article = root / EXPERIMENT / name
        if article.exists():
            paths.add(article)
    return SourceLock(hashes={str(p.relative_to(root)): sha(p) for p in sorted(paths)})


def freeze(root: Path) -> str:
    """Called explicitly by the coordinator only after all source and tests settle."""
    path = root / EXPERIMENT / "SOURCE_LOCK.json"
    if path.exists():
        return verify_lock(root)
    if journal_path(root).exists() and journal_path(root).stat().st_size:
        raise ValueError("cannot freeze a source lock after native attempts")
    write(path, source_snapshot(root))
    return sha(path)


def verify_lock(root: Path) -> str:
    path = root / EXPERIMENT / "SOURCE_LOCK.json"
    if (
        not path.exists()
        or SourceLock.model_validate_json(path.read_text()) != source_snapshot(root)
    ):
        raise ValueError("source lock differs from current source, plan, inputs or references")
    return sha(path)


def write_binding(trial: Trial, place: Path, root: Path) -> None:
    write(place / "BINDING.json", Binding(trial=trial, source_lock_sha256=verify_lock(root)))


def check_binding(trial: Trial, place: Path, root: Path) -> None:
    path = place / "BINDING.json"
    expected = Binding(trial=trial, source_lock_sha256=verify_lock(root))
    if not path.exists() or Binding.model_validate_json(path.read_text()) != expected:
        raise ValueError("saved trial binding differs from requested variant or source lock")
