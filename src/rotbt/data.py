import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .models import CashEvent, Dataset


def _futu(path: Path) -> Any:
    # Preserve logs verbatim; extract exactly one structured CLI response.
    parsed = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.startswith("{") and '"data"' in line
    ]
    if len(parsed) != 1:
        raise ValueError(f"expected exactly one API response: {path}")
    return parsed[0]["data"]


def cash_events(root: Path, symbol: str) -> list[CashEvent]:
    rehab = _futu(root / f"{symbol}_rehab_futu.log")
    detail = _futu(root / f"{symbol}_dividends_futu.log") or []
    if symbol == "GLD":
        if rehab or detail:
            raise ValueError("unexpected GLD corporate action")
        return []
    records: list[CashEvent] = []
    rh = {
        row["ex_div_date"]: row
        for row in rehab
        if "2019-01-01" <= row["ex_div_date"] <= "2025-12-31"
    }
    # All selected instruments must have cash-only events in this window.
    for row in rh.values():
        for field in ("split_ratio", "spin_off_ratio", "allotment_ratio", "stk_spo_ratio"):
            if row.get(field) not in (None, 0, 1):
                raise ValueError(f"unmodelled action {symbol}: {row}")
    if symbol == "SPY":
        for row in detail:
            ex = date.fromisoformat(row["ex_date"].replace("/", "-"))
            if not date(2020, 1, 1) <= ex <= date(2025, 12, 31):
                continue
            match = re.fullmatch(r"1股(?:派息|收益)([\d.]+)USD", row["statement"])
            if match is None:
                raise ValueError(f"unknown cash statement: {row}")
            amount = float(match[1])
            if (
                ex.isoformat() not in rh
                or abs(amount - float(rh[ex.isoformat()]["per_cash_div"])) > 0.0001
            ):
                raise ValueError(f"SPY cash sources disagree {ex}")
            records.append(
                CashEvent(
                    symbol=symbol,
                    ex_date=ex,
                    pay_date=date.fromisoformat(row["dividend_payable_date"].replace("/", "-")),
                    amount=amount,
                    source="Futu dividend_list; rehab cross-check",
                )
            )
        if len(records) != 24:
            raise ValueError("SPY must have 24 main-window cash events")
        for key, row in rh.items():
            if key < "2020":
                records.append(
                    CashEvent(
                        symbol=symbol,
                        ex_date=date.fromisoformat(key),
                        pay_date=None,
                        amount=float(row["per_cash_div"]),
                        source="Futu rehab warmup",
                    )
                )
    else:
        response = json.loads((root / f"{symbol}_dividends_nasdaq.json").read_text())
        rows = response["data"]["dividends"]["rows"]
        grouped: dict[date, list[Any]] = {}
        for row in detail:
            ex = date.fromisoformat(row["ex_date"].replace("/", "-"))
            if date(2020, 1, 1) <= ex <= date(2025, 12, 31):
                grouped.setdefault(ex, []).append(row)
        for row in rows:
            ex = pd.to_datetime(row["exOrEffDate"], format="%m/%d/%Y").date()
            if not date(2019, 1, 1) <= ex <= date(2025, 12, 31):
                continue
            if row["type"] != "Cash":
                raise ValueError(f"unmodelled distribution {row}")
            amount = float(row["amount"].replace("$", ""))
            pay = pd.to_datetime(row["paymentDate"], format="%m/%d/%Y").date()
            if ex.year >= 2020:
                if ex not in grouped:
                    raise ValueError(f"missing secondary cash date {symbol} {ex}")
                parts = []
                for item in grouped[ex]:
                    match = re.fullmatch(r"1股(?:派息|收益)([\d.]+)USD", item["statement"])
                    if match is None:
                        raise ValueError(f"unknown cash statement: {item}")
                    parts.append(float(match[1]))
                    if date.fromisoformat(item["dividend_payable_date"].replace("/", "-")) != pay:
                        raise ValueError(f"pay dates disagree {symbol} {ex}")
                if abs(sum(parts) - amount) > 0.0001:
                    raise ValueError(f"independent cash amounts disagree {symbol} {ex}")
            records.append(
                CashEvent(
                    symbol=symbol,
                    ex_date=ex,
                    pay_date=pay,
                    amount=amount,
                    source="Nasdaq Cash; Futu aggregated",
                )
            )
        main = [event for event in records if event.ex_date.year >= 2020]
        if len(main) != 72 or len(grouped) != 72:
            raise ValueError(f"{symbol}: incomplete cash history")
    if len(set(event.ex_date for event in records)) != len(records):
        raise ValueError("duplicate canonical cash event")
    if any(event.pay_date is not None and event.pay_date < event.ex_date for event in records):
        raise ValueError("payment before ex date")
    return sorted(records, key=lambda event: event.ex_date)


def prepare_frame(
    frame: pd.DataFrame, events: list[CashEvent], weekly: bool = True
) -> pd.DataFrame:
    out = frame.copy().sort_index()
    index = pd.DatetimeIndex(out.index)
    if len(index) == 0 or index.has_duplicates:
        raise ValueError("empty/duplicate price input")
    values = out[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (out.volume <= 0).any():
        raise ValueError("invalid prices or volume")
    if (
        (out.high < out[["open", "close", "low"]].max(axis=1))
        | (out.low > out[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("invalid OHLC")
    cash = pd.Series({pd.Timestamp(event.ex_date): event.amount for event in events}, dtype=float)
    amounts = cash.reindex(index, fill_value=0)
    out["tr"] = ((out.close + amounts) / out.close.shift()).fillna(1).cumprod() * 100
    for period in (21, 63, 126):
        out[f"mom{period}"] = out.tr.pct_change(periods=period, fill_method=None)
    out["week_end"] = 0.0
    if weekly:
        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(index[0], index[-1] + pd.Timedelta(days=10))
        expected = pd.DatetimeIndex(sessions).tz_localize(None)
        if not index.equals(expected[expected <= index[-1]]):
            raise ValueError("incomplete exchange calendar")
        weeks = [(day.isocalendar().year, day.isocalendar().week) for day in expected]
        ends = {expected[i] for i in range(len(expected) - 1) if weeks[i] != weeks[i + 1]}
        out["week_end"] = index.isin(ends).astype(float)
    return out


def load_dataset(root: Path, symbol: str, end: date = date(2025, 12, 31)) -> Dataset:
    if end > date(2025, 12, 31):
        raise ValueError("2026 is outside the registered range")
    source = json.loads((root / f"{symbol}_prices_source.json").read_text())
    selected = []
    for row in source:
        day = pd.to_datetime(row["date"], format="%m/%d/%Y").date()
        if date(2019, 1, 1) <= day <= end:
            selected.append(
                (
                    pd.Timestamp(day),
                    *(
                        float(row[key].replace(",", ""))
                        for key in ("open", "high", "low", "close", "volume")
                    ),
                )
            )
    frame = pd.DataFrame(
        selected, columns=["date", "open", "high", "low", "close", "volume"]
    ).set_index("date")
    events = cash_events(root, symbol)
    return Dataset(symbol=symbol, frame=prepare_frame(frame, events), dividends=events)
