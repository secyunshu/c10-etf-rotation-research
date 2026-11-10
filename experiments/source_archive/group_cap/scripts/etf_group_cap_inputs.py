"""Immutable joint input adapter; no network, strategy, broker or account execution."""

import json
import math
import re
import traceback
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype
from pydantic import ConfigDict, TypeAdapter

from rotbt.models import CashEvent, Dataset, Model, SplitEvent
from rotbt.phase2.features import build_features
from rotbt.phase2.models import FeatureStore
from scripts.etf_c10_2026_inputs import (
    FIELDS,
    FIRST,
    LAST,
    OLD_LAST,
    sha,
    validate_dataset,
    verify_hashes,
)
from scripts.etf_c10_2026_inputs import (
    NAMES as OLD_NAMES,
)
from scripts.etf_c10_2026_inputs import (
    load as load_old,
)
from scripts.etf_c10_forward_data import (
    CASH_MATCH_TOLERANCE,
    UNSUPPORTED,
    _date,  # pyright: ignore[reportPrivateUsage]
    _number,  # pyright: ignore[reportPrivateUsage]
    _ratio,  # pyright: ignore[reportPrivateUsage]
)
from scripts.etf_rotation_futu_inputs import load_dataset as load_legacy

RESEARCH = "research/ETF_GROUP_CAP_20261109"
OUTPUT = "inputs_etf_group_cap_20261109"
NEW_NAMES = ("AGG", "BIL", "HYG", "QUAL", "SOXX", "TIP", "VIG")
LEGACY_NAMES = ("HYG", "QUAL", "SOXX", "VIG")
NAMES = tuple(sorted((*OLD_NAMES, *NEW_NAMES)))


class Prepared(Model):
    datasets: list[Dataset]
    features: FeatureStore
    first: date
    last: date


class Actions(Model):
    cash: list[CashEvent]
    splits: list[SplitEvent]
    matches: list[dict[str, Any]]


class Protocol(Model):
    model_config = ConfigDict(extra="ignore")
    id: Literal["ETF_ONE_PER_ECONOMIC_RISK_GROUP_20261109_V1"]
    end2026: date
    hashes: dict[str, str]


class Manifest(Model):
    cohort: Literal["2025", "2026"]
    first: date
    last: date
    names: tuple[str, ...]
    index_names: dict[str, str | None]
    provenance_sha256: str
    output_hashes: dict[str, str]


def blocked(symbol: str, reason: str) -> ValueError:
    return ValueError(f"DATA_BLOCKED {symbol}: {reason}")


@beartype
def parse_actions(
    symbol: str,
    rehab: list[dict[str, Any]],
    details: list[dict[str, Any]],
    *,
    first: date = FIRST,
    last: date = LAST,
    retained: list[CashEvent] | None = None,
    approved_revisions: dict[date, float] | None = None,
    ignored_details: dict[date, tuple[float, date]] | None = None,
) -> Actions:
    """Retain historical-share cash; corroborate detail using actual later splits."""
    old = {e.ex_date: e for e in retained or [] if first < e.ex_date <= last}
    actions: dict[date, tuple[float, float, float]] = {}
    for row in rehab:
        ex = _date(row.get("ex_div_date"), symbol, "rehab ex date")
        if not first < ex <= last:
            continue
        if ex in actions:
            raise blocked(symbol, f"duplicate rehab date {ex}")
        if any(_number(row.get(field), symbol, field) > 0 for field in UNSUPPORTED):
            raise blocked(symbol, f"unmodelled stock/spinoff action {ex}")
        if _number(row.get("special_dividend"), symbol, "special_dividend") > 0:
            raise blocked(symbol, f"unresolved special distribution {ex}")
        amount = raw_amount = _number(row.get("per_cash_div"), symbol, "per_cash_div")
        if ex in old:
            if (
                abs(amount - old[ex].amount) > CASH_MATCH_TOLERANCE
                and (approved_revisions or {}).get(ex) != old[ex].amount
            ):
                raise blocked(symbol, f"unapproved historical cash revision {ex}")
            amount = old[ex].amount
        actions[ex] = amount, _ratio(row, symbol), raw_amount
    grouped: dict[date, tuple[float, date, set[str]]] = {}
    ignored_seen: set[date] = set()
    for row in details:
        ex = _date(row.get("ex_date"), symbol, "detail ex date")
        if not first < ex <= last:
            continue
        statement = str(row.get("statement", ""))
        match = re.fullmatch(r"1股(?:派息|收益)(\d+(?:\.\d+)?)USD", statement)
        if match is None:
            raise blocked(symbol, f"unknown USD cash component {ex}: {statement}")
        amount = float(match[1])
        pay = _date(row.get("dividend_payable_date"), symbol, "real payment date")
        if amount <= 0 or pay < ex:
            raise blocked(symbol, f"invalid cash/payment date {ex}")
        if ex in (ignored_details or {}):
            if (
                (amount, pay) != (ignored_details or {})[ex]
                or actions.get(ex, (0.0, 1.0, 0.0))[0] != 0
                or ex in ignored_seen
            ):
                raise blocked(symbol, f"approved zero detail scope changed {ex}")
            ignored_seen.add(ex)
            continue
        before, old_pay, statements = grouped.get(ex, (0.0, pay, set()))
        if statement in statements:
            raise blocked(symbol, f"duplicate identical cash detail {ex}")
        if old_pay != pay:
            raise blocked(symbol, f"conflicting component payment dates {ex}")
        grouped[ex] = before + amount, pay, statements | {statement}
    if ignored_seen != set(ignored_details or {}):
        raise blocked(symbol, "approved zero detail missing")
    if {ex for ex, (amount, _, _) in actions.items() if amount > 0} != set(grouped):
        raise blocked(symbol, "cash date coverage mismatch between rehab and detailed payment")
    cash, splits, matches = [], [], []
    for ex, (amount, ratio, raw_amount) in sorted(actions.items()):
        if ratio != 1:
            splits.append(
                SplitEvent(
                    symbol=symbol,
                    day=ex,
                    shares_after_per_before=ratio,
                    source="Futu actual split/join numerator and denominator",
                )
            )
        if not amount:
            continue
        stated, pay, _ = grouped[ex]
        factor = math.prod(r for day, (_, r, _) in actions.items() if ex < day <= last)
        difference = min(abs(stated - amount), abs(stated * factor - amount))
        if difference > CASH_MATCH_TOLERANCE + 1e-12:
            raise blocked(
                symbol, f"cash amount/split unit conflict {ex}: {amount}/{stated}/{factor}"
            )
        if ex in old and old[ex].pay_date != pay:
            raise blocked(symbol, f"historical actual payment revision {ex}")
        cash.append(
            old.get(ex)
            or CashEvent(
                symbol=symbol,
                ex_date=ex,
                pay_date=pay,
                amount=amount,
                source="new Futu historical-share rehab + actual detail/payment",
            )
        )
        matches.append(
            dict(
                symbol=symbol,
                ex_date=str(ex),
                pay_date=str(pay),
                raw_rehab_amount=raw_amount,
                retained_approved_amount=amount,
                detailed_amount=stated,
                later_share_factor=factor,
                difference=difference,
            )
        )
    if not set(old) <= {e.ex_date for e in cash}:
        raise blocked(symbol, "approved cash event missing from new evidence")
    return Actions(cash=cash, splits=splits, matches=matches)


@beartype
def validate_frame(symbol: str, frame: pd.DataFrame, first: date, last: date) -> None:
    expected = xcals.get_calendar("XNYS").sessions_in_range(first, last).tz_localize(None)
    if not frame.index.equals(expected) or set(frame.columns) != set(FIELDS):
        raise blocked(symbol, "raw calendar/field mismatch")
    values = frame[list(FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise blocked(symbol, "nonfinite/nonpositive raw OHLCV or actual turnover")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (
        frame.low > frame[["open", "close", "high"]].min(axis=1)
    ).any():
        raise blocked(symbol, "inconsistent raw OHLC")


@beartype
def with_total_return(
    symbol: str, frame: pd.DataFrame, cash: list[CashEvent], splits: list[SplitEvent]
) -> Dataset:
    out = frame.copy()
    amounts = pd.Series({pd.Timestamp(e.ex_date): e.amount for e in cash}, dtype=float)
    ratios = pd.Series(
        {pd.Timestamp(e.day): e.shares_after_per_before for e in splits}, dtype=float
    )
    out["tr"] = (
        (out.close + amounts.reindex(out.index, fill_value=0))
        * ratios.reindex(out.index, fill_value=1)
        / out.close.shift()
    ).fillna(1).cumprod() * 100
    return Dataset(symbol=symbol, frame=out, dividends=cash, splits=splits)


@beartype
def clip_dataset(data: Dataset, last: date) -> Dataset:
    return Dataset(
        symbol=data.symbol,
        frame=data.frame.loc[: str(last)].copy(),
        dividends=[e for e in data.dividends if e.ex_date <= last],
        splits=[e for e in data.splits if e.day <= last],
    )


@beartype
def features_for(datasets: list[Dataset]) -> FeatureStore:
    store = build_features(datasets)
    for data in datasets:
        frame = store.frames[data.symbol]
        frame["raw_close"] = data.frame.close
        frame["turnover20"] = data.frame.turnover.rolling(20).mean()
        frame["positive20"] = data.frame.volume.gt(0).rolling(20).sum()
    return store


@beartype
def retain_history(old: Dataset, new: Dataset) -> Dataset:
    before = old.frame.loc[str(FIRST) : str(OLD_LAST), list(FIELDS)]
    after = new.frame.loc[before.index, list(FIELDS)]
    if not np.array_equal(before.to_numpy(dtype=float), after.to_numpy(dtype=float)):
        raise blocked(old.symbol, "raw historical overlap revision")
    cash = [e for e in old.dividends if FIRST < e.ex_date <= OLD_LAST]
    observed = {e.ex_date: e for e in new.dividends if e.ex_date <= OLD_LAST}
    if set(observed) != {e.ex_date for e in cash} or any(
        observed[e.ex_date].pay_date != e.pay_date
        or abs(observed[e.ex_date].amount - e.amount) > CASH_MATCH_TOLERANCE
        for e in cash
    ):
        raise blocked(old.symbol, "historical cash overlap revision")
    splits = [e for e in old.splits if FIRST < e.day <= OLD_LAST]
    if [(e.day, e.shares_after_per_before) for e in splits] != [
        (e.day, e.shares_after_per_before) for e in new.splits if e.day <= OLD_LAST
    ]:
        raise blocked(old.symbol, "historical split overlap revision")
    return with_total_return(
        old.symbol,
        new.frame[list(FIELDS)],
        cash + [e for e in new.dividends if e.ex_date > OLD_LAST],
        splits + [e for e in new.splits if e.day > OLD_LAST],
    )


@beartype
def load_legacy_evidence(root: Path, symbol: str) -> Dataset:
    folder = root / "inputs_etf_legacy_resolved_20261107"
    data = load_legacy(folder, symbol)
    manifest = json.loads((folder / f"{symbol}_manifest.json").read_text())
    rows = [
        row
        for i in range(manifest["pages"])
        for row in json.loads((folder / f"{symbol}_raw_page{i}.json").read_text())
    ]
    source = pd.DataFrame(rows)
    source.index = pd.DatetimeIndex(pd.to_datetime(source.pop("time_key")))
    if not source.index.equals(data.frame.index):
        raise blocked(symbol, "legacy actual-turnover calendar differs from trusted input")
    # Decode all original JSON scalars consistently. pd.read_json in the trusted
    # cash loader can shift a decimal quote by one binary ULP; no quote is revised.
    data.frame[list(FIELDS)] = source[list(FIELDS)].astype(float)
    return data


def _read_wrapper(folder: Path, symbol: str, kind: str) -> Any:
    obj = json.loads((folder / f"{symbol}_{kind}.json").read_text())
    if obj.get("ret") != 0 or obj.get("page_key") is not None or obj.get("code") != f"US.{symbol}":
        raise blocked(symbol, f"failed or incomplete {kind} wrapper")
    stamp = datetime.fromisoformat(obj["received_utc"])
    if stamp.tzinfo is None or stamp > datetime.now(UTC):
        raise blocked(symbol, "invalid receipt time")
    return obj["data"]


def _normalize(root: Path, symbol: str, old: Dataset | None) -> tuple[Dataset, Actions]:
    folder = root / RESEARCH / "raw_new7"
    prices = _read_wrapper(folder, symbol, "prices")
    if not isinstance(prices, list) or any(row.get("code") != f"US.{symbol}" for row in prices):
        raise blocked(symbol, "raw quote identity")
    frame = pd.DataFrame(prices)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("time_key")))
    frame = frame[list(FIELDS)].astype(float)
    validate_frame(symbol, frame, FIRST, LAST)
    rehab = _read_wrapper(folder, symbol, "rehab")
    detail = _read_wrapper(folder, symbol, "dividends")
    if not isinstance(rehab, list) or not isinstance(detail, dict) or "dividend_list" not in detail:
        raise blocked(symbol, "malformed action evidence")
    revisions = approved_cash_revisions(root, symbol)
    actions = parse_actions(
        symbol,
        rehab,
        detail["dividend_list"],
        retained=old.dividends if old else None,
        approved_revisions=revisions,
        ignored_details=approved_zero_details(root, symbol),
    )
    data = with_total_return(symbol, frame, actions.cash, actions.splits)
    return retain_history(old, data) if old else data, actions


def approved_cash_revisions(root: Path, symbol: str) -> dict[date, float]:
    if symbol != "QUAL":
        return {}
    path = root / RESEARCH / "data_qa/QUAL_ISSUER_ADJUDICATION.json"
    proof = json.loads(path.read_text())
    verify_hashes(proof["hashes"])
    row = proof["issuer_row"]
    if (
        proof["status"] != "ROOT_ADJUDICATED_RETAIN_PRIOR_APPROVED_CASH"
        or proof["retained_amount"] != 0.09652
        or proof["actual_later_split_factor"] != 4
        or row["exDate"] != 20231228
        or row["payableDate"] != 20240104
        or row["totalDistribution"] != "0.024128"
        or any(
            float(row[k]) != 0
            for k in ("shortTermCapitalGain", "longTermCapitalGain", "returnOnCapital")
        )
    ):
        raise blocked(symbol, "issuer adjudication identity or financial classification")
    return {date(2023, 12, 28): 0.09652}


def approved_zero_details(root: Path, symbol: str) -> dict[date, tuple[float, date]]:
    if symbol != "BIL":
        return {}
    path = root / RESEARCH / "data_qa/BIL_ZERO_ADJUDICATION.json"
    proof = json.loads(path.read_text())
    verify_hashes(proof["hashes"])
    row = proof["issuer_row"]
    if (
        proof["status"] != "USER_APPROVED_BIL_ZERO_DISTRIBUTION"
        or proof["retained_amount"] != 0
        or proof["ignored_detail_amount"] != 0.0215
        or proof["ex_date"] != "2022-03-01"
        or proof["pay_date"] != "2022-03-07"
        or row["B"] != "BIL"
        or row["C"] != "78468R663"
        or row["D"] != "03/01/2022"
        or row["F"] != "03/07/2022"
        or float(row["G"]) != 0
    ):
        raise blocked(symbol, "approved issuer zero event identity")
    return {date(2022, 3, 1): (0.0215, date(2022, 3, 7))}


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def _index_name(data: Dataset) -> str | None:
    name = data.frame.index.name
    if name is not None and not isinstance(name, str):
        raise blocked(data.symbol, "unsupported source index name")
    return name


def _sources(root: Path) -> dict[str, str]:
    protocol_path = root / RESEARCH / "PROTOCOL.json"
    protocol = Protocol.model_validate_json(protocol_path.read_text())
    if protocol.end2026 != LAST:
        raise blocked("joint45", "protocol cutoff")
    verify_hashes(protocol.hashes)
    raw = root / RESEARCH / "raw_new7"
    harvest = json.loads((raw / "HARVEST_RECEIPT.json").read_text())
    if (
        harvest["status"] != "RAW_FETCH_COMPLETE_QA_PENDING"
        or tuple(harvest["symbols"]) != NEW_NAMES
        or harvest["first"] != str(FIRST)
        or harvest["asof"] != str(LAST)
        or harvest["protocol_sha256"] != sha(protocol_path)
    ):
        raise blocked("joint45", "harvest scope or protocol binding")
    paths = [p for p in raw.iterdir() if p.is_file()]
    paths += [protocol_path, Path(__file__).resolve(), root / "tests/test_etf_group_cap_inputs.py"]
    adjudication = root / RESEARCH / "data_qa/QUAL_ISSUER_ADJUDICATION.json"
    approved_cash_revisions(root, "QUAL")
    paths += [adjudication, root / RESEARCH / "data_qa/ISSUER_QUAL_BROWSER_RAW.json"]
    approved_zero_details(root, "BIL")
    paths += [
        root / RESEARCH / "data_qa/BIL_ZERO_ADJUDICATION.json",
        root / RESEARCH / "data_qa/ISSUER_BIL_XLSX_BROWSER_RAW_V3.json",
    ]
    legacy = root / "inputs_etf_legacy_resolved_20261107"
    receipt_path = legacy / "RESOLUTION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "USER_APPROVED_EXPLICIT_FIVE_RESOLUTIONS":
        raise blocked("QUAL", "approved historical resolution missing")
    paths.append(receipt_path)
    for symbol in LEGACY_NAMES:
        path = legacy / f"{symbol}_manifest.json"
        manifest = json.loads(path.read_text())
        for name, expected in manifest["sha256"].items():
            item = legacy / name
            if sha(item) != expected:
                allowed = next(
                    (
                        e
                        for e in receipt["changed_events"]
                        if e["symbol"] == symbol and e["derived_sha256"] == sha(item)
                    ),
                    None,
                )
                if allowed is None:
                    raise blocked(symbol, f"unapproved historical source change: {name}")
            paths.append(item)
        paths.append(path)
    return dict(protocol.hashes) | {str(p): sha(p) for p in paths}


@beartype
def build(root: Path) -> Prepared:
    root = root.resolve()
    qa = (
        root
        / RESEARCH
        / "data_qa"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    )
    qa.mkdir(parents=True, exist_ok=False)
    try:
        sources = _sources(root)
        _write(qa / "SOURCE_PROVENANCE.json", {"frozen_utc": datetime.now(UTC), "hashes": sources})
        old = load_old(root)
        datasets = list(old.datasets)
        reports = []
        for symbol in NEW_NAMES:
            historical = load_legacy_evidence(root, symbol) if symbol in LEGACY_NAMES else None
            data, actions = _normalize(root, symbol, historical)
            validate_dataset(data)
            datasets.append(data)
            reports.append(
                {
                    "symbol": symbol,
                    "rows": len(data.frame),
                    "cash_matches": actions.matches,
                    "splits": [e.model_dump(mode="json") for e in data.splits],
                    "old_overlap_retained": historical is not None,
                }
            )
        if tuple(sorted(d.symbol for d in datasets)) != NAMES or len(datasets) != 45:
            raise blocked("joint45", "missing/duplicate complete symbol set")
        datasets.sort(key=lambda d: d.symbol)
        output = root / OUTPUT
        output.mkdir(exist_ok=False)
        provenance_path = output / "SOURCE_PROVENANCE.json"
        _write(provenance_path, {"hashes": sources, "frozen_utc": datetime.now(UTC)})
        cohorts: tuple[tuple[Literal["2025", "2026"], date], ...] = (
            ("2025", OLD_LAST),
            ("2026", LAST),
        )
        for cutoff, last in cohorts:
            folder = output / cutoff
            folder.mkdir()
            cohort = [clip_dataset(data, last) for data in datasets]
            for data in cohort:
                validate_dataset(data, last)
                data.frame.to_csv(
                    folder / f"{data.symbol}.csv.gz", index_label="date", compression="gzip"
                )
            for name, mapping in (
                (
                    "CASH.json",
                    {d.symbol: [e.model_dump(mode="json") for e in d.dividends] for d in cohort},
                ),
                (
                    "SPLITS.json",
                    {d.symbol: [e.model_dump(mode="json") for e in d.splits] for d in cohort},
                ),
            ):
                _write(folder / name, mapping)
            manifest = Manifest(
                cohort=cutoff,
                first=FIRST,
                last=last,
                names=NAMES,
                index_names={data.symbol: _index_name(data) for data in cohort},
                provenance_sha256=sha(provenance_path),
                output_hashes={p.name: sha(p) for p in folder.iterdir()},
            )
            _write(folder / "MANIFEST.json", manifest.model_dump(mode="json"))
        verify_hashes(sources)
        for data in old.datasets:
            candidate = next(d for d in datasets if d.symbol == data.symbol)
            pd.testing.assert_frame_equal(candidate.frame, data.frame, check_exact=True)
            if candidate.dividends != data.dividends or candidate.splits != data.splits:
                raise blocked(data.symbol, "immutable old38 financial event changed")
        prepared = load(root)
        for data in old.datasets:
            persisted = next(d for d in prepared.datasets if d.symbol == data.symbol)
            pd.testing.assert_frame_equal(persisted.frame, data.frame, check_exact=True)
        _write(
            qa / "PASS.json",
            {
                "status": "PASS",
                "symbols": len(datasets),
                "old38_exact": True,
                "source_hashes": sources,
                "new7": reports,
                "cohorts": {"2025": 1256, "2026": 1470},
                "native_runs": 0,
            },
        )
        return prepared
    except Exception:
        _write(
            qa / "FAILURE.json",
            {"status": "DATA_BLOCKED", "traceback": traceback.format_exc(), "native_runs": 0},
        )
        raise


@beartype
def load(root: Path, cutoff: Literal["2025", "2026"] = "2026") -> Prepared:
    root = root.resolve()
    output = root / OUTPUT
    provenance = output / "SOURCE_PROVENANCE.json"
    verify_hashes(json.loads(provenance.read_text())["hashes"])
    folder = output / cutoff
    manifest = Manifest.model_validate_json((folder / "MANIFEST.json").read_text())
    last = OLD_LAST if cutoff == "2025" else LAST
    expected = {f"{s}.csv.gz" for s in NAMES} | {"CASH.json", "SPLITS.json"}
    if (
        manifest.cohort != cutoff
        or (manifest.first, manifest.last) != (FIRST, last)
        or manifest.names != NAMES
        or set(manifest.index_names) != set(NAMES)
        or manifest.provenance_sha256 != sha(provenance)
        or set(manifest.output_hashes) != expected
        or {p.name for p in folder.iterdir()} != expected | {"MANIFEST.json"}
    ):
        raise blocked("joint45", "canonical physical cohort identity or file membership")
    verify_hashes({str(folder / name): value for name, value in manifest.output_hashes.items()})
    cash = TypeAdapter(dict[str, list[CashEvent]]).validate_json((folder / "CASH.json").read_text())
    splits = TypeAdapter(dict[str, list[SplitEvent]]).validate_json(
        (folder / "SPLITS.json").read_text()
    )
    if set(cash) != set(NAMES) or set(splits) != set(NAMES):
        raise blocked("joint45", "cash/split symbol identity")
    datasets = [
        Dataset(
            symbol=s,
            frame=pd.read_csv(
                folder / f"{s}.csv.gz",
                index_col="date",
                parse_dates=True,
                float_precision="round_trip",
            ),
            dividends=cash[s],
            splits=splits[s],
        )
        for s in NAMES
    ]
    for data in datasets:
        data.frame.index.name = manifest.index_names[data.symbol]
        validate_dataset(data, last)
    return Prepared(datasets=datasets, features=features_for(datasets), first=FIRST, last=last)
