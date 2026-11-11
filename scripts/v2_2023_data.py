"""Immutable canonical daily inputs and frozen pools; no provider/network calls."""

import json
from pathlib import Path

import pandas as pd
from beartype import beartype
from pydantic import TypeAdapter

from c10share.io import verify_hashes
from c10standard.io import ROOT, load_strategy
from c10standard.io import validate_dataset as validate_standard_dataset
from rotbt.models import CashEvent, Dataset, SplitEvent
from scripts.v2_2023_models import LAST, Pool, Prepared, Role
from scripts.v2_2023_signals import features_for

STUDY = Path("experiments/v2-2023-start")


@beartype
def validate_dataset(data: Dataset) -> None:
    validate_standard_dataset(data, LAST)


@beartype
def load_prepared(role: Role, root: Path = ROOT) -> Prepared:
    name = "NEW_POOL" if role == "NEW_POOL" else "OLD_POOL"
    pool = Pool.model_validate_json((root / STUDY / f"{name}.json").read_bytes())
    if pool.role != name:
        raise ValueError("pool filename and bound identity differ")
    folder = root / STUDY / "data"
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    if manifest["first"] != "2020-12-31" or manifest["last"] != "2025-12-31":
        raise ValueError("input physical cutoff changed")
    verify_hashes(folder, manifest["hashes"])
    cash = TypeAdapter(dict[str, list[CashEvent]]).validate_json(
        (folder / "CASH.json").read_bytes()
    )
    splits = TypeAdapter(dict[str, list[SplitEvent]]).validate_json(
        (folder / "SPLITS.json").read_bytes()
    )
    names = sorted((*pool.risk_names, "SHY"))
    datasets = []
    for symbol in names:
        frame = pd.read_csv(
            folder / f"{symbol}.csv.gz",
            index_col="date",
            parse_dates=True,
            float_precision="round_trip",
        )
        data = Dataset(symbol=symbol, frame=frame, dividends=cash[symbol], splits=splits[symbol])
        validate_dataset(data)
        datasets.append(data)
    return Prepared(
        pool=pool, datasets=datasets, features=features_for(datasets), fees=load_strategy(root).fees
    )
