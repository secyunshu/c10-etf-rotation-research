"""Load frozen canonical inputs for annual continuation; never query a provider."""

import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype
from pydantic import TypeAdapter

from c10share.io import verify_hashes
from c10standard.io import ROOT, load_strategy
from rotbt.models import CashEvent, Dataset, SplitEvent
from scripts.ytd_annual_io import STUDY
from scripts.ytd_annual_models import (
    FIRST,
    HISTORICAL_LAST,
    LAST,
    Prepared,
    Role,
    Schedule,
    Stage,
)
from scripts.ytd_annual_signals import features_for


@beartype
def validate(data: Dataset, stage: Stage) -> None:
    last = HISTORICAL_LAST if stage == "bridge" else LAST
    expected = xcals.get_calendar("XNYS").sessions_in_range(FIRST, last).tz_localize(None)
    f = data.frame
    if not f.index.equals(expected):
        raise ValueError(f"{data.symbol}: physical fullcalendar cutoff mismatch")
    core = f[["open", "high", "low", "close", "volume", "turnover", "tr"]]
    if not np.isfinite(core.to_numpy(dtype=float)).all() or (core <= 0).any().any():
        raise ValueError("nonpositive/nonfinite price/volume/turnover/TR")
    if (f.high < f[["open", "close", "low"]].max(axis=1)).any() or (
        f.low > f[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("invalid OHLC")
    if len({e.ex_date for e in data.dividends}) != len(data.dividends) or any(
        e.symbol != data.symbol
        or e.pay_date is None
        or e.pay_date < e.ex_date
        or not FIRST < e.ex_date <= last
        for e in data.dividends
    ):
        raise ValueError("cash identity/payment/cutoff invalid")
    if len({s.day for s in data.splits}) != len(data.splits) or any(
        s.symbol != data.symbol or not FIRST < s.day <= last or s.shares_after_per_before <= 0
        for s in data.splits
    ):
        raise ValueError("split identity/ratio/cutoff invalid")
    amounts = pd.Series({pd.Timestamp(e.ex_date): e.amount for e in data.dividends}, dtype=float)
    ratios = pd.Series(
        {pd.Timestamp(s.day): s.shares_after_per_before for s in data.splits}, dtype=float
    )
    recurrence = (
        (f.close + amounts.reindex(f.index, fill_value=0))
        * ratios.reindex(f.index, fill_value=1)
        / f.close.shift()
    )
    if not np.allclose((f.tr / f.tr.shift()).iloc[1:], recurrence.iloc[1:], rtol=1e-10, atol=1e-10):
        raise ValueError("cash/split totalreturn recurrence mismatch")


@beartype
def load_prepared(role: Role, stage: Stage, root: Path = ROOT) -> Prepared:
    if stage == "bridge":
        old = json.loads((root / "experiments/v2-annual-repool/ANNUAL.json").read_text())
        from scripts.ytd_annual_models import Pool, schedule_sha256

        pools = tuple(Pool.model_validate(x) for x in old["entries"])
        schedule = Schedule(stage=stage, entries=pools, sha256=schedule_sha256(stage, pools))
        folder = root / "experiments/v2-annual-repool/data"
    else:
        schedule = Schedule.model_validate_json((root / STUDY / "SCHEDULE.json").read_bytes())
        folder = root / STUDY / "data/science"
    if schedule.stage != stage:
        raise ValueError("wrong schedule stage")
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    last = HISTORICAL_LAST if stage == "bridge" else LAST
    if (manifest["first"], manifest["last"]) != (str(FIRST), str(last)):
        raise ValueError("wrong manifest physical cutoff")
    verify_hashes(folder, manifest["hashes"])
    cash = TypeAdapter(dict[str, list[CashEvent]]).validate_json(
        (folder / "CASH.json").read_bytes()
    )
    splits = TypeAdapter(dict[str, list[SplitEvent]]).validate_json(
        (folder / "SPLITS.json").read_bytes()
    )
    names = (
        sorted({s for p in schedule.entries for s in p.risk_names} | {"SHY"})
        if role == "ANNUAL"
        else sorted((role.removesuffix("_BH"), "SHY"))
    )
    data = []
    for name in names:
        frame = pd.read_csv(
            folder / f"{name}.csv.gz",
            index_col="date",
            parse_dates=True,
            float_precision="round_trip",
        )
        item = Dataset(symbol=name, frame=frame, dividends=cash[name], splits=splits[name])
        validate(item, stage)
        data.append(item)
    return Prepared(
        role=role,
        stage=stage,
        schedule=schedule,
        datasets=data,
        features=features_for(data),
        fees=load_strategy(root).fees,
        last=last,
    )
