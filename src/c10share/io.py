"""Load frozen local files, verify identity, and preserve stored TR precision."""

import hashlib
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from rotbt.models import CashEvent, Dataset, SplitEvent
from rotbt.phase2.features import build_features
from rotbt.phase2.models import P2Spec

from .models import FIRST, LAST, RISK, RULES, START, Account, MarketManifest, Prepared, Task
from .targets import target

ROOT = Path(__file__).resolve().parents[2]


def verify_hashes(folder: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = folder / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"missing or unsafe file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"hash mismatch: {name}")


def load_accounts(root: Path = ROOT) -> list[Account]:
    accounts = TypeAdapter(list[Account]).validate_json((root / "data/ACCOUNTS.json").read_text())
    expected = {f"{rule}_{cost}bp" for rule in RULES for cost in (5, 20)}
    if len(accounts) != 12 or {a.name for a in accounts} != expected:
        raise ValueError("expected exactly twelve fixed accounts")
    return accounts


def load_task(account: Account, root: Path = ROOT) -> Task:
    folder = root / "data/reference" / account.name
    verify_hashes(folder, account.reference_files)
    if {p.name for p in folder.iterdir()} != set(account.reference_files):
        raise ValueError(f"missing or extra reference files: {account.name}")
    task = Task.model_validate_json((folder / "task.json").read_text())
    hold = account.rule.endswith("_BH")
    expected_spec = P2Spec(
        label=account.rule,
        universe="expanded",
        windows=(252,),
        top=3,
        frequency="hold" if hold else "monthly",
        fixed=(account.rule.removesuffix("_BH"),) if hold else (),
    )
    settings = task.settings
    if (
        task.key != account.name
        or task.spec != expected_spec
        or (task.start, task.end) != (START, LAST)
        or (settings.initial_cash, settings.invested, settings.tax, settings.slip_bps)
        != (50000, 0.95, 0, account.cost_bps)
        or settings.effective_research_period.model_dump()
        != {"first_year": 2022, "last_year": 2025, "split_year": 2023}
        or tuple(settings.fees.model_dump().values())
        != (0.0049, 0.99, 0.005, 1.0, 0.003, 0.0000206, 0.01, 0.000003)
    ):
        raise ValueError(f"frozen task changed: {account.name}")
    return task


def validate_dataset(data: Dataset) -> None:
    frame = data.frame
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise ValueError("missing date-indexed market data")
    if frame.index.max() > pd.Timestamp(LAST) or frame.index.min() < pd.Timestamp(FIRST):
        raise ValueError("physical scope cutoff violated")
    expected = xcals.get_calendar("XNYS").sessions_in_range(FIRST, LAST).tz_localize(None)
    if not frame.index.equals(expected):
        raise ValueError("incomplete or duplicated exchange calendar")
    values = frame[["open", "high", "low", "close", "volume", "turnover", "tr"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= 0).any().any():
        raise ValueError("invalid price, turnover, volume or TR")
    if (
        (frame.high < frame[["open", "low", "close"]].max(axis=1))
        | (frame.low > frame[["open", "high", "close"]].min(axis=1))
    ).any():
        raise ValueError("invalid OHLC")
    if len({e.ex_date for e in data.dividends}) != len(data.dividends) or any(
        e.symbol != data.symbol
        or not FIRST < e.ex_date <= LAST
        or e.pay_date is None
        or e.pay_date < e.ex_date
        or pd.Timestamp(e.ex_date) not in frame.index
        for e in data.dividends
    ):
        raise ValueError("invalid cash event identity, cutoff or payment evidence")
    if any(
        not FIRST < e.day <= LAST or pd.Timestamp(e.day) not in frame.index for e in data.splits
    ):
        raise ValueError("invalid split event cutoff or calendar")
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


def load_market(root: Path = ROOT) -> Prepared:
    folder = root / "data/market"
    manifest = MarketManifest.model_validate_json((folder / "MANIFEST.json").read_text())
    verify_hashes(folder, manifest.output_hashes)
    if {p.name for p in folder.iterdir()} != set(manifest.output_hashes) | {"MANIFEST.json"}:
        raise ValueError("missing or extra market files")
    cash = TypeAdapter(dict[str, list[CashEvent]]).validate_json((folder / "CASH.json").read_text())
    splits = TypeAdapter(dict[str, list[SplitEvent]]).validate_json(
        (folder / "SPLITS.json").read_text()
    )
    names = tuple(sorted((*RISK, "SHY", "SPY")))
    if set(cash) != set(names) or set(splits) != set(names):
        raise ValueError("cash/split symbols changed")
    datasets = []
    for symbol in names:
        data = Dataset(
            symbol=symbol,
            frame=pd.read_csv(folder / f"{symbol}.csv.gz", index_col=0, parse_dates=True),
            dividends=cash[symbol],
            splits=splits[symbol],
        )
        validate_dataset(data)
        datasets.append(data)
    features = build_features(datasets)
    for data in datasets:
        frame = features.frames[data.symbol]
        frame["momskip21"] = frame.tr.shift(21) / frame.tr.shift(252) - 1
        frame["raw_close"] = data.frame.close
        frame["turnover20"] = data.frame.turnover.rolling(20).mean()
        frame["volume20"] = data.frame.volume.rolling(20).mean()
        frame["positive20"] = data.frame.volume.gt(0).rolling(20).sum()
        for field in ("open", "high", "low", "close"):
            frame[f"signal_{field}"] = data.frame[field] * frame.tr / data.frame.close
    return Prepared(datasets=datasets, features=features, manifest=manifest)


def feed_names(rule: str) -> tuple[str, ...]:
    if rule not in RULES:
        raise ValueError("unregistered public account")
    if rule.endswith("_BH"):
        return tuple(sorted((rule.removesuffix("_BH"), "SHY")))
    return ("IEF", "SHY", "SPY") if rule == "SPY60_IEF40" else tuple(sorted((*RISK, "SHY")))


def weights_for(rule: str, prepared: Prepared, day: date) -> dict[str, float]:
    if rule not in RULES:
        raise ValueError("unregistered public account")
    if rule.endswith("_BH"):
        return {rule.removesuffix("_BH"): 1.0}
    if rule == "SPY60_IEF40":
        return {"SPY": 0.6, "IEF": 0.4}
    return target(
        rule, prepared.features, day, prepared.manifest.risk_names, prepared.manifest.groups
    )
