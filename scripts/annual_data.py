"""Load the union needed for accounting while retaining separate active-year pools."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from beartype import beartype
from pydantic import TypeAdapter

from c10share.io import verify_hashes
from c10standard.io import ROOT, load_strategy, validate_dataset
from rotbt.models import CashEvent, Dataset, SplitEvent
from scripts.annual_io import STUDY
from scripts.annual_models import Mode, Prepared, Schedule
from scripts.annual_signals import features_for


@beartype
def load_prepared(mode: Mode, root: Path = ROOT) -> Prepared:
    schedule = Schedule.model_validate_json((root / STUDY / f"{mode}.json").read_bytes())
    if schedule.mode != mode:
        raise ValueError("schedule file differs from requested mode")
    folder = root / STUDY / "data"
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    if (manifest["first"], manifest["last"]) != ("2020-12-31", "2025-12-31"):
        raise ValueError("physical market cutoff changed")
    verify_hashes(folder, manifest["hashes"])
    cash = TypeAdapter(dict[str, list[CashEvent]]).validate_json(
        (folder / "CASH.json").read_bytes()
    )
    splits = TypeAdapter(dict[str, list[SplitEvent]]).validate_json(
        (folder / "SPLITS.json").read_bytes()
    )
    names = sorted({s for pool in schedule.entries for s in pool.risk_names} | {"SHY"})
    datasets = []
    for symbol in names:
        frame = pd.read_csv(
            folder / f"{symbol}.csv.gz",
            index_col="date",
            parse_dates=True,
            float_precision="round_trip",
        )
        data = Dataset(symbol=symbol, frame=frame, dividends=cash[symbol], splits=splits[symbol])
        validate_dataset(data, date(2025, 12, 31))
        datasets.append(data)
    return Prepared(
        schedule=schedule,
        datasets=datasets,
        features=features_for(datasets),
        fees=load_strategy(root).fees,
    )
