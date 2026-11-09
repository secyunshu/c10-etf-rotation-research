"""Portable V2 inputs and source-precision reference binding, entirely offline."""

from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from c10share.io import verify_hashes
from rotbt.models import CashEvent, Dataset, ResearchPeriod, Settings, SplitEvent
from rotbt.phase2.features import build_features

from .models import (
    FIRST,
    HISTORICAL_LAST,
    LAST,
    NAMES,
    Account,
    Contract,
    MarketManifest,
    Prepared,
    Strategy,
)

ROOT = Path(__file__).resolve().parents[2]


def load_strategy(root: Path = ROOT) -> Strategy:
    return Strategy.model_validate_json((root / "standard/STRATEGY.json").read_text())


def load_accounts(root: Path = ROOT) -> list[Account]:
    accounts = TypeAdapter(list[Account]).validate_json(
        (root / "standard/ACCOUNTS.json").read_text()
    )
    from .run import catalog

    if len(accounts) != 36 or {a.key for a in accounts} != {c.key for c in catalog()}:
        raise ValueError("exactly36 fixed V2 reference accounts required")
    return accounts


def reference_folder(account: Account, root: Path = ROOT) -> Path:
    folder = root / "standard/reference" / account.key
    verify_hashes(folder, account.reference_files)
    if {p.name for p in folder.iterdir()} != set(account.reference_files):
        raise ValueError("missing or extra V2 reference files")
    return folder


def settings_for(strategy: Strategy, contract: Contract) -> Settings:
    return Settings(
        fees=strategy.fees,
        initial_cash=50000,
        invested=0.95,
        slip_bps=contract.cost_bps,
        tax=0,
        research_period=ResearchPeriod(first_year=2022, last_year=2025, split_year=2023),
    )


def validate_dataset(data: Dataset, last: date) -> None:
    if last not in (HISTORICAL_LAST, LAST):
        raise ValueError("unregistered market cutoff")
    frame = data.frame
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise ValueError("missing date-indexed market")
    if frame.index.min() < pd.Timestamp(FIRST) or frame.index.max() > pd.Timestamp(last):
        raise ValueError("physical cutoff violated")
    expected = xcals.get_calendar("XNYS").sessions_in_range(FIRST, last).tz_localize(None)
    if not frame.index.equals(expected):
        raise ValueError("incomplete or duplicated exchange calendar")
    values = frame[["open", "high", "low", "close", "volume", "turnover", "tr"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= 0).any().any():
        raise ValueError("nonfinite or nonpositive market input")
    if (
        (frame.high < frame[["open", "low", "close"]].max(axis=1))
        | (frame.low > frame[["open", "high", "close"]].min(axis=1))
    ).any():
        raise ValueError("invalid OHLC")
    if len({e.ex_date for e in data.dividends}) != len(data.dividends) or any(
        e.symbol != data.symbol
        or not FIRST < e.ex_date <= last
        or e.pay_date is None
        or e.pay_date < e.ex_date
        or pd.Timestamp(e.ex_date) not in frame.index
        for e in data.dividends
    ):
        raise ValueError("invalid cash identity, cutoff or payment evidence")
    if len({e.day for e in data.splits}) != len(data.splits) or any(
        e.symbol != data.symbol
        or not FIRST < e.day <= last
        or pd.Timestamp(e.day) not in frame.index
        for e in data.splits
    ):
        raise ValueError("invalid split identity or cutoff")
    amounts = pd.Series({pd.Timestamp(e.ex_date): e.amount for e in data.dividends}, dtype=float)
    ratios = pd.Series(
        {pd.Timestamp(e.day): e.shares_after_per_before for e in data.splits}, dtype=float
    )
    recurrence = (
        (frame.close + amounts.reindex(frame.index, fill_value=0))
        * ratios.reindex(frame.index, fill_value=1)
        / frame.close.shift()
    )
    expected_tr = recurrence.fillna(1).cumprod() * 100
    if not np.allclose(frame.tr.to_numpy(), expected_tr.to_numpy(), rtol=1e-13, atol=1e-10):
        raise ValueError(f"canonical total return recurrence changed: {data.symbol}")


def load_market(root: Path = ROOT, cutoff: str = "2025") -> Prepared:
    if cutoff not in ("2025", "2026"):
        raise ValueError("unregistered market cohort")
    strategy = load_strategy(root)
    folder = root / "standard/data" / cutoff
    manifest = MarketManifest.model_validate_json((folder / "MANIFEST.json").read_text())
    if manifest.cohort != cutoff:
        raise ValueError("market cohort identity mismatch")
    verify_hashes(folder, manifest.output_hashes)
    if {p.name for p in folder.iterdir()} != set(manifest.output_hashes) | {"MANIFEST.json"}:
        raise ValueError("missing or extra V2 market files")
    cash = TypeAdapter(dict[str, list[CashEvent]]).validate_json((folder / "CASH.json").read_text())
    splits = TypeAdapter(dict[str, list[SplitEvent]]).validate_json(
        (folder / "SPLITS.json").read_text()
    )
    if set(cash) != set(NAMES) or set(splits) != set(NAMES):
        raise ValueError("cash/split symbols changed")
    datasets = []
    for symbol in NAMES:
        data = Dataset(
            symbol=symbol,
            frame=pd.read_csv(
                folder / f"{symbol}.csv.gz",
                index_col=0,
                parse_dates=True,
                float_precision="round_trip",
            ),
            dividends=cash[symbol],
            splits=splits[symbol],
        )
        validate_dataset(data, manifest.last)
        datasets.append(data)
    features = build_features(datasets)
    for data in datasets:
        frame = features.frames[data.symbol]
        frame["raw_close"] = data.frame.close
        frame["turnover20"] = data.frame.turnover.rolling(20).mean()
        frame["positive20"] = data.frame.volume.gt(0).rolling(20).sum()
    return Prepared(
        datasets=datasets,
        features=features,
        strategy=strategy,
        risk_names=strategy.risk_names,
        first=manifest.first,
        last=manifest.last,
    )
