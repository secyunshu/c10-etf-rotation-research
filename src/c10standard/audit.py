"""Independent checks of saved native books; never creates fills or account NAV."""

from collections import defaultdict
from datetime import date
from math import fsum, isclose, isfinite, sqrt
from statistics import mean, stdev
from typing import cast

import exchange_calendars as xcals
import pandas as pd
from beartype import beartype

from rotbt.broker import commission
from rotbt.models import Dataset, Distribution, HoldingRecord, Model, OrderRecord, Result, Settings

from .models import Artifact, OpenTrade, Prepared
from .native import decision_days, feed_names
from .targets import oracle

START, THROUGH = date(2026, 1, 2), date(2026, 11, 6)


class FinancialScope(Model):
    start: date
    through: date
    settings: Settings
    terminal: bool


class Description(Model):
    days: int
    opening_equity: float
    closing_equity: float
    total_return: float
    max_drawdown: float
    sharpe_rf0: float
    annualized_volatility: float
    fees: float
    orders: int
    dividends: float
    end_cash: float
    end_locked: float
    end_available: float
    monthly_returns: dict[str, float]
    end_positions: dict[str, int]
    asset_profit: dict[str, float]


def equal(actual: float, expected: float, label: str, tolerance: float = 1e-5) -> None:
    if (
        not isfinite(actual)
        or not isfinite(expected)
        or not isclose(actual, expected, rel_tol=1e-10, abs_tol=tolerance)
    ):
        raise ValueError(f"{label}: {actual} != {expected}")


@beartype
def reconcile(result: Result, datasets: list[Dataset], scope: FinancialScope) -> None:
    """Reconcile every source cash event, split, fill and all holdings, including zero rows."""
    calendar = xcals.get_calendar("XNYS")
    days = [t.date() for t in calendar.sessions_in_range(scope.start, scope.through)]
    if [r.day for r in result.daily] != days:
        raise ValueError("native daily calendar differs from exact finite account window")
    data = {d.symbol: d for d in datasets}
    if len(data) != len(datasets) or not data:
        raise ValueError("duplicate or empty native universe")
    orders: dict[date, list[OrderRecord]] = defaultdict(list)
    holdings: dict[date, dict[str, HoldingRecord]] = defaultdict(dict)
    distributions: dict[date, dict[str, Distribution]] = defaultdict(dict)
    for order in result.orders:
        if order.day not in days or order.symbol not in data or order.size == 0:
            raise ValueError("invalid native order identity")
        if not scope.terminal and order.reason in {"terminal", "annual_liquidation"}:
            raise ValueError("artificial liquidation in continuing account")
        orders[order.day].append(order)
    for holding in result.holdings:
        if holding.day not in days or holding.symbol in holdings[holding.day]:
            raise ValueError("duplicate or out-of-window holding")
        holdings[holding.day][holding.symbol] = holding
    for dividend in result.distributions:
        if dividend.ex_date not in days or dividend.symbol in distributions[dividend.ex_date]:
            raise ValueError("duplicate or out-of-window entitlement")
        distributions[dividend.ex_date][dividend.symbol] = dividend
    expected_splits = [
        s for d in datasets for s in d.splits if scope.start <= s.day <= scope.through
    ]
    if result.splits != expected_splits:
        raise ValueError("native split export differs from source events")
    cash = scope.settings.initial_cash
    quantities = dict.fromkeys(data, 0)
    for row in result.daily:
        day = row.day
        for split in expected_splits:
            if split.day == day:
                value = quantities[split.symbol] * split.shares_after_per_before
                equal(value, round(value), "fractional split needs actual cash-in-lieu", 1e-8)
                quantities[split.symbol] = round(value)
        events = [
            e
            for d in datasets
            for e in d.dividends
            if e.ex_date == day and quantities[e.symbol] > 0
        ]
        if {e.symbol for e in events} != set(distributions[day]):
            raise ValueError("source dividend entitlement set mismatch")
        for event in events:
            record = distributions[day][event.symbol]
            if event.pay_date is None or record.pay_date != event.pay_date:
                raise ValueError("cash event payment evidence mismatch")
            release = calendar.date_to_session(pd.Timestamp(event.pay_date), direction="next")
            if release.date() <= event.pay_date:
                release = calendar.next_session(release)
            if record.release_date != release.date() or record.qty != quantities[event.symbol]:
                raise ValueError("entitlement units or pay-date cash release mismatch")
            equal(record.per_share, event.amount, "cash amount")
            gross = quantities[event.symbol] * event.amount
            equal(record.gross_amount, gross, "gross entitlement")
            equal(record.net_amount, gross * (1 - scope.settings.tax), "net entitlement")
            cash += record.net_amount
        locked = fsum(
            d.net_amount for d in result.distributions if d.ex_date <= day < d.release_date
        )
        for order in orders[day]:
            raw_open = float(cast(float, data[order.symbol].frame.at[pd.Timestamp(day), "open"]))
            expected = raw_open * (
                1 + (1 if order.size > 0 else -1) * scope.settings.slip_bps / 10000
            )
            equal(order.price, expected, "adverse open fill", 1e-8)
            equal(order.fee, commission(order.size, order.price, scope.settings.fees), "fee", 1e-7)
            cash -= order.size * order.price + order.fee
            if cash - locked < -1e-5:
                raise ValueError("order spends locked or unavailable cash")
            quantities[order.symbol] += order.size
            if quantities[order.symbol] < 0:
                raise ValueError("negative independently reconstructed position")
        if set(holdings[day]) != set(data):
            raise ValueError("holding set incomplete or unexpected symbol")
        for symbol, holding in holdings[day].items():
            if holding.size != quantities[symbol]:
                raise ValueError("native share reconciliation")
            equal(
                holding.close,
                float(cast(float, data[symbol].frame.at[pd.Timestamp(day), "close"])),
                "raw close",
                1e-8,
            )
        market = fsum(h.size * h.close for h in holdings[day].values())
        for actual, expected, label in (
            (row.cash, cash, "cash"),
            (row.equity, cash + market, "NAV"),
            (row.locked, locked, "receivable"),
            (row.available, cash - locked, "available"),
            (row.gross_exposure, market / row.equity, "gross exposure"),
        ):
            equal(actual, expected, label)
        if row.available < -1e-5 or row.positions != sum(q > 0 for q in quantities.values()):
            raise ValueError("unavailable cash spent or position count mismatch")
    if scope.terminal and any(quantities.values()):
        raise ValueError("bridge terminal account not flat")
    equal(
        result.native_total_return,
        result.daily[-1].equity / scope.settings.initial_cash - 1,
        "native TimeReturn",
        1e-9,
    )


@beartype
def describe(result: Result, *, continuous: bool) -> Description:
    days = [r for r in result.daily if START <= r.day <= THROUGH]
    expected = [t.date() for t in xcals.get_calendar("XNYS").sessions_in_range(START, THROUGH)]
    if [r.day for r in days] != expected or len(days) != 214:
        raise ValueError("description requires exact frozen 214-session YTD window")
    opening = 50000.0
    if continuous:
        prior = [r for r in result.daily if r.day == date(2025, 12, 31)]
        if len(prior) != 1:
            raise ValueError("continuous account requires its own December 31 closing NAV")
        opening = prior[0].equity
    previous, peak, drawdown = opening, opening, 0.0
    returns: list[float] = []
    month_gross: dict[str, float] = {}
    for row in days:
        change = row.equity / previous - 1
        returns.append(change)
        key = row.day.strftime("%Y-%m")
        month_gross[key] = month_gross.get(key, 1.0) * (1 + change)
        peak = max(peak, row.equity)
        drawdown = max(drawdown, 1 - row.equity / peak)
        previous = row.equity
    sigma = stdev(returns)
    orders = [o for o in result.orders if START <= o.day <= THROUGH]
    dividends = [d for d in result.distributions if START <= d.ex_date <= THROUGH]
    profits: dict[str, float] = defaultdict(float)
    for h in result.holdings:
        if h.day == THROUGH:
            profits[h.symbol] += h.size * h.close
        elif continuous and h.day == date(2025, 12, 31):
            profits[h.symbol] -= h.size * h.close
    for o in orders:
        profits[o.symbol] -= o.size * o.price + o.fee
    for d in dividends:
        profits[d.symbol] += d.net_amount
    equal(fsum(profits.values()), days[-1].equity - opening, "asset profit sum")
    return Description(
        days=len(days),
        opening_equity=opening,
        closing_equity=days[-1].equity,
        total_return=days[-1].equity / opening - 1,
        max_drawdown=drawdown,
        sharpe_rf0=mean(returns) / sigma * sqrt(252) if sigma else 0,
        annualized_volatility=sigma * sqrt(252),
        fees=fsum(o.fee for o in orders),
        orders=len(orders),
        dividends=fsum(d.net_amount for d in dividends),
        end_cash=days[-1].cash,
        end_locked=days[-1].locked,
        end_available=days[-1].available,
        monthly_returns={k: v - 1 for k, v in month_gross.items()},
        end_positions={h.symbol: h.size for h in result.holdings if h.day == THROUGH and h.size},
        asset_profit={k: v for k, v in profits.items() if v},
    )


def audit(artifact: Artifact, prepared: Prepared, settings: Settings) -> None:
    contract, account = artifact.contract, artifact.account
    names = feed_names(contract, prepared)
    selected = [next(d for d in prepared.datasets if d.symbol == name) for name in names]
    reconcile(
        account,
        selected,
        FinancialScope(
            start=contract.start,
            through=contract.through,
            settings=settings,
            terminal=contract.mode == "bridge",
        ),
    )
    prior = xcals.get_calendar("XNYS").previous_session(pd.Timestamp(contract.start)).date()
    expected_days = [prior, *decision_days(contract)]
    if [r.day for r in artifact.targets] != expected_days:
        raise ValueError("signal calendar mismatch")
    for record in artifact.targets:
        expected = oracle(contract, prepared, record.day)
        if set(record.q) != set(expected.q):
            raise ValueError("independent target asset mismatch")
        for symbol, weight in record.q.items():
            equal(weight, expected.q[symbol], "independent target weight", 1e-12)
    if [r.day for r in artifact.native_returns] != [r.day for r in account.daily]:
        raise ValueError("native TimeReturn daily calendar")
    previous = 50000.0
    for row, native in zip(account.daily, artifact.native_returns, strict=True):
        equal(native.value, row.equity / previous - 1, "daily native TimeReturn", 1e-12)
        previous = row.equity
    live = {h.symbol: h.size for h in account.holdings if h.day == contract.through and h.size}
    native_sizes: dict[str, int] = {}
    for trade in artifact.open_trades:
        native_sizes[trade.symbol] = native_sizes.get(trade.symbol, 0) + trade.size
    if live != native_sizes:
        raise ValueError("native Position/Trade split units disagree")
    equal(
        sum(t.commission for t in artifact.open_trades) + sum(t.commission for t in account.trades),
        sum(o.fee for o in account.orders),
        "native closed/open trade commissions",
    )
    reconcile_trade_books(account, artifact.open_trades)


def reconcile_trade_books(account: Result, trades: list[OpenTrade]) -> None:
    """Native realized plus unrealized trade P&L must explain independently audited NAV."""
    closes = {h.symbol: h.close for h in account.holdings if h.day == account.daily[-1].day}
    for closed in account.trades:
        equal(
            closed.net_pnl_ex_dividends,
            closed.gross_pnl - closed.commission,
            "native closed trade net profit",
        )
    gross = fsum(t.gross_pnl for t in account.trades) + fsum(
        t.pnl + t.size * (closes[t.symbol] - t.price) for t in trades
    )
    expected = (
        gross
        + fsum(d.net_amount for d in account.distributions)
        - fsum(o.fee for o in account.orders)
    )
    equal(expected, account.daily[-1].equity - 50000, "native trade profit")
