"""Read-only catalogs, bounded prefix features and the shared new-release budget."""

import fcntl
import hashlib
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from c10share.io import verify_hashes
from c10standard.io import ROOT
from c10standard.io import load_market as load_standard_market
from c10standard.models import Prepared
from rotbt.models import Dataset, Model
from rotbt.phase2.features import build_features

from .models import Attempt, Experiment


class CatalogEntry(Model):
    trial: dict[str, Any]
    path: str
    hashes: dict[str, str]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_experiments(root: Path = ROOT) -> list[Experiment]:
    experiments = []
    for family in ("calendar13612w", "top2-momentum-variants"):
        entries = TypeAdapter(list[CatalogEntry]).validate_json(
            (root / "experiments" / family / "ACCOUNTS.json").read_text()
        )
        for entry in entries:
            trial = entry.trial
            variant = "W13612" if family == "calendar13612w" else trial.get("variant")
            experiment = Experiment.model_validate(
                {
                    "key": trial["key"],
                    "variant": variant,
                    "k": trial["k"],
                    "cost_bps": trial["cost_bps"],
                    "cohort": trial["cohort"],
                    "reference_path": entry.path,
                    "reference_hashes": entry.hashes,
                }
            )
            if trial != experiment.original_trial:
                raise ValueError("published trial differs from exact frozen experiment")
            reference_folder(experiment, root)
            experiments.append(experiment)
    cohorts = ("historical2022_2025", "reset2026", "continuous2022_2026")
    expected = {
        f"W13612_K{k}_{cost}bp_{cohort}"
        for k in (1, 2, 3)
        for cost in (5, 20)
        for cohort in cohorts
    } | {
        f"{variant}_K2_{cost}bp_{cohort}"
        for variant in ("E01", "E02", "E03", "E04", "E05", "E06")
        for cost in (5, 20)
        for cohort in cohorts
    }
    if len(experiments) != 54 or {e.key for e in experiments} != expected:
        raise ValueError("exactly54 frozen experiment contracts required")
    return experiments


def reference_folder(experiment: Experiment, root: Path = ROOT) -> Path:
    folder = root / experiment.reference_path
    verify_hashes(folder, experiment.reference_hashes)
    if {p.name for p in folder.iterdir()} != set(experiment.reference_hashes):
        raise ValueError("missing or extra experiment reference files")
    if json.loads((folder / "TRIAL.json").read_text()) != experiment.original_trial:
        raise ValueError("saved reference TRIAL differs from experiment identity")
    return folder


def load_market(root: Path = ROOT, cutoff: str = "2025") -> Prepared:
    return load_standard_market(root, cutoff)


def prefix_market(prepared: Prepared, day: date) -> Prepared:
    datasets = [
        Dataset(
            symbol=d.symbol,
            frame=d.frame.loc[: str(day)].copy(),
            dividends=[e for e in d.dividends if e.ex_date <= day],
            splits=[e for e in d.splits if e.day <= day],
        )
        for d in prepared.datasets
    ]
    features = build_features(datasets)
    for data in datasets:
        frame = features.frames[data.symbol]
        frame["raw_close"] = data.frame.close
        frame["turnover20"] = data.frame.turnover.rolling(20).mean()
        frame["positive20"] = data.frame.volume.gt(0).rolling(20).sum()
    return Prepared(
        datasets=datasets,
        features=features,
        strategy=prepared.strategy,
        risk_names=prepared.risk_names,
        first=prepared.first,
        last=day,
    )


def journal_event(
    journal: Path,
    event: Literal["started", "native_completed", "finished", "failed", "revalidated"],
    account: str,
    attempt_id: str,
    output: Path,
    error: str = "",
) -> None:
    record = Attempt(
        release_id="TOP2_DEFAULT_AND_EXPERIMENTS_20261111",
        event=event,
        account=account,
        attempt_id=attempt_id,
        utc=datetime.now(UTC),
        output=str(output),
        error=error,
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        records = [Attempt.model_validate_json(line) for line in handle.read().splitlines()]
        if event == "started" and sum(r.event == "started" for r in records) >= 72:
            raise ValueError("new release72-native-attempt budget exhausted, including failures")
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_hashes(experiment: Experiment, root: Path = ROOT) -> dict[str, str]:
    reference_folder(experiment, root)
    files = [p for p in (root / "src").rglob("*.py")]
    files += [
        root / "experiments" / family / "ACCOUNTS.json"
        for family in ("calendar13612w", "top2-momentum-variants")
    ]
    files += [root / "standard/STRATEGY.json", root / "pyproject.toml"]
    for cutoff in ("2025", "2026"):
        files += [p for p in (root / "standard/data" / cutoff).iterdir() if p.is_file()]
    files += [root / experiment.reference_path / name for name in experiment.reference_hashes]
    return {p.relative_to(root).as_posix(): sha(p) for p in files}


def verify_source_hashes(root: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha(root / path) != expected:
            raise ValueError(f"experiment source binding changed: {name}")
