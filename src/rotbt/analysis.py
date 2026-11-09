"""Validate broker exports, then describe the frozen account periods.

No fills or account values are generated here. Returns use broker closing equity,
including the first day's change from initial cash. Sharpe uses sample daily
volatility (rf=0); singleton/zero-volatility samples report zero. Drawdown includes
the period's opening capital. Money reconciliation allows $0.00001 roundoff.
"""

from collections import defaultdict
from datetime import date
from itertools import pairwise
from math import fsum, isclose, isfinite, sqrt
from statistics import mean, stdev

from .models import Distribution, HoldingRecord, Metrics, OrderRecord, Result, Settings

_TOL = 0.00001


def _equal(
    actual: float,
    expected: float,
    label: str,
    day: date | None = None,
    *,
    abs_tol: float = _TOL,
) -> None:
    if (
        not isfinite(actual)
        or not isfinite(expected)
        or not isclose(
            actual,
            expected,
            rel_tol=1e-10,
            abs_tol=abs_tol,
        )
    ):
        raise ValueError(f"{label} reconciliation {day}: {actual} != {expected}")


def _reconcile(result: Result, settings: Settings) -> None:
    period = settings.effective_research_period
    split_keys = [(event.symbol, event.day) for event in result.splits]
    if len(split_keys) != len(set(split_keys)):
        raise ValueError("duplicate split event in exported account")
    if not isfinite(settings.initial_cash) or settings.initial_cash <= 0:
        raise ValueError("initial cash must be finite and positive")
    days = [row.day for row in result.daily]
    if not days:
        raise ValueError("empty daily ledger")
    if any(not period.first_year <= day.year <= period.last_year for day in days):
        raise ValueError(
            f"daily dates outside research period {period.first_year}-{period.last_year}"
        )
    if any(left >= right for left, right in pairwise(days)):
        raise ValueError("daily dates must be unique and increasing")
    calendar = set(days)
    orders: dict[date, list[OrderRecord]] = defaultdict(list)
    holdings: dict[date, dict[str, HoldingRecord]] = defaultdict(dict)
    dividends: dict[date, list[Distribution]] = defaultdict(list)
    releases: dict[date, list[float]] = defaultdict(list)
    for order in result.orders:
        if order.day not in calendar:
            raise ValueError(f"order outside daily ledger: {order.day}")
        if (
            order.size == 0
            or not isfinite(order.price)
            or order.price <= 0
            or not isfinite(order.fee)
            or order.fee < 0
        ):
            raise ValueError(f"invalid order: {order.day} {order.symbol}")
        orders[order.day].append(order)
    for holding in result.holdings:
        if holding.day not in calendar:
            raise ValueError(f"holding outside daily ledger: {holding.day}")
        if holding.symbol in holdings[holding.day]:
            raise ValueError(f"duplicate holding: {holding.day} {holding.symbol}")
        if holding.size < 0 or not isfinite(holding.close) or holding.close <= 0:
            raise ValueError(f"invalid holding: {holding.day} {holding.symbol}")
        holdings[holding.day][holding.symbol] = holding
    for dividend in result.distributions:
        if dividend.ex_date not in calendar:
            raise ValueError(f"dividend outside daily ledger: {dividend.ex_date}")
        if (
            dividend.qty <= 0
            or not isfinite(dividend.per_share)
            or dividend.per_share <= 0
            or dividend.pay_date < dividend.ex_date
            or dividend.release_date <= dividend.pay_date
        ):
            raise ValueError(f"invalid dividend or release date: {dividend.ex_date}")
        # Only observed sessions are checked; future release needs no future prices.
        release = next((day for day in days if day > dividend.pay_date), None)
        if release is not None and dividend.release_date != release:
            raise ValueError(f"release must be first session after payment: {dividend.ex_date}")
        _equal(
            dividend.gross_amount,
            dividend.qty * dividend.per_share,
            "gross dividend",
            dividend.ex_date,
        )
        _equal(
            dividend.net_amount,
            dividend.gross_amount * (1 - settings.tax),
            "net dividend",
            dividend.ex_date,
        )
        dividends[dividend.ex_date].append(dividend)
        releases[dividend.release_date].append(dividend.net_amount)

    previous_cash = settings.initial_cash
    locked = 0.0
    quantities: dict[str, int] = {}
    for row in result.daily:
        day = row.day
        for split in result.splits:
            if split.day == day:
                new_qty = quantities.get(split.symbol, 0) * split.shares_after_per_before
                if not isfinite(new_qty) or abs(new_qty - round(new_qty)) > 1e-8:
                    raise ValueError(f"fractional split needs actual cash-in-lieu data: {day}")
                quantities[split.symbol] = round(new_qty)
        if (
            not all(
                isfinite(value)
                for value in (
                    row.equity,
                    row.cash,
                    row.locked,
                    row.available,
                    row.gross_exposure,
                )
            )
            or row.equity <= 0
            or row.locked < -_TOL
            or row.available < -_TOL
        ):
            raise ValueError(f"invalid daily equity/cash/locked/available: {day}")
        locked -= fsum(releases[day])
        for dividend in dividends[day]:
            if dividend.qty != quantities.get(dividend.symbol, 0):
                raise ValueError(f"dividend entitlement disagrees with prior holdings: {day}")
        # Check actual exported fills against prior spendable cash. Today's queued
        # dividend enters broker cash only after matching and cannot fund a buy.
        spendable = previous_cash - locked
        for order in orders[day]:
            quantities[order.symbol] = quantities.get(order.symbol, 0) + order.size
            if quantities[order.symbol] < 0:
                raise ValueError(f"order creates short position: {day} {order.symbol}")
            spendable -= order.size * order.price + order.fee
            if spendable < -_TOL:
                raise ValueError(f"order spends locked cash: {day} {order.symbol}")
        income = fsum(dividend.net_amount for dividend in dividends[day])
        cash_flow = fsum(-order.size * order.price - order.fee for order in orders[day])
        _equal(row.cash, previous_cash + cash_flow + income, "cash", day)
        locked += income
        _equal(row.locked, locked, "locked", day)
        _equal(row.available, row.cash - row.locked, "available", day)
        actual = {symbol: holding.size for symbol, holding in holdings[day].items() if holding.size}
        expected = {symbol: qty for symbol, qty in quantities.items() if qty}
        if actual != expected:
            raise ValueError(f"holdings disagree with executed orders: {day}")
        if row.positions != len(actual):
            raise ValueError(f"positions count disagrees with holdings: {day}")
        market = fsum(holding.size * holding.close for holding in holdings[day].values())
        _equal(row.equity, row.cash + market, "equity", day)
        _equal(row.gross_exposure, market / row.equity, "gross exposure", day, abs_tol=1e-10)
        previous_cash = row.cash
    if any(quantities.values()) or result.daily[-1].positions != 0:
        raise ValueError("terminal positions must be zero")
    _equal(
        result.native_total_return,
        result.daily[-1].equity / settings.initial_cash - 1,
        "native total return",
        abs_tol=1e-10,
    )


def analyze(result: Result, settings: Settings) -> list[Metrics]:
    """Return early/late, available calendar years, and full-account metrics.

    Empty periods are omitted for short test runs. Every nonempty period starts
    at the preceding broker close, or initial_cash at the start of the sample.
    Annual capital is never reset. Turnover is absolute executed order notional
    divided by mean closing equity; dividends are net ex-date income.
    """
    _reconcile(result, settings)
    period = settings.effective_research_period
    daily = result.daily
    previous = [settings.initial_cash] + [row.equity for row in daily[:-1]]
    returns = [row.equity / before - 1 for row, before in zip(daily, previous, strict=True)]
    years = sorted({row.day.year for row in daily})
    periods = [
        ("early", period.first_year, period.split_year),
        ("late", period.split_year + 1, period.last_year),
    ]
    periods.extend((str(year), year, year) for year in years)
    periods.append(("full", period.first_year, period.last_year))
    metrics = []
    for name, first_year, last_year in periods:
        indices = [i for i, row in enumerate(daily) if first_year <= row.day.year <= last_year]
        if not indices:
            continue
        first, last = indices[0], indices[-1]
        opening = previous[first]
        growth = daily[last].equity / opening
        sample = returns[first : last + 1]
        volatility = stdev(sample) if len(sample) > 1 else 0.0
        peak = opening
        max_dd = 0.0
        for row in daily[first : last + 1]:
            peak = max(peak, row.equity)
            max_dd = max(max_dd, 1 - row.equity / peak)
        positive_years = 0
        for year in years:
            annual = [i for i in indices if daily[i].day.year == year]
            if annual and daily[annual[-1]].equity > previous[annual[0]]:
                positive_years += 1
        orders = [
            order for order in result.orders if daily[first].day <= order.day <= daily[last].day
        ]
        dividends = [
            dividend
            for dividend in result.distributions
            if daily[first].day <= dividend.ex_date <= daily[last].day
        ]
        metrics.append(
            Metrics(
                period=name,
                days=len(sample),
                cagr=growth ** (252 / len(sample)) - 1,
                total_return=growth - 1,
                sharpe=sqrt(252) * mean(sample) / volatility if volatility else 0,
                max_dd=max_dd,
                fees=fsum(order.fee for order in orders),
                dividends=fsum(dividend.net_amount for dividend in dividends),
                orders=len(orders),
                turnover=fsum(abs(order.size * order.price) for order in orders)
                / mean(row.equity for row in daily[first : last + 1]),
                positive_years=positive_years,
            )
        )
    return metrics
