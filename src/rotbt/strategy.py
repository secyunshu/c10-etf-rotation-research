"""Signals and cash-event adapter; no custom broker or matching implementation."""

import math
from datetime import date
from typing import Any, override

import backtrader as bt

from .broker import commission
from .models import (
    RISK,
    CashEvent,
    ClosedTrade,
    DailyRecord,
    Distribution,
    Episode,
    HoldingRecord,
    OrderRecord,
    Settings,
    Spec,
)


class Rotation(bt.Strategy):
    params = (
        ("spec", None),
        ("settings", None),
        ("start", None),
        ("end", None),
        ("events", ()),
        ("forced_targets", ()),
        ("calendar_days", ()),
    )

    def __init__(self) -> None:
        self.spec: Spec = self.p.spec
        self.settings: Settings = self.p.settings
        self.start_day: date = self.p.start
        self.end_day: date = self.p.end
        self.events: tuple[CashEvent, ...] = self.p.events
        self.forced: tuple[tuple[date, tuple[str, ...]], ...] = self.p.forced_targets
        self.orders: list[OrderRecord] = []
        self.daily: list[DailyRecord] = []
        self.holdings: list[HoldingRecord] = []
        self.distributions: list[Distribution] = []
        self.closed_trades: list[ClosedTrade] = []
        self.pending: list[Any] = []
        self.episodes: dict[str, Episode] = {}
        self.targets: dict[str, float] | None = None
        self.blocked: set[str] = set()
        self.exit_only: set[str] = set()
        self.last_entitlement: date | None = None
        self.new_events: list[Distribution] = []
        self.receivables: list[Distribution] = []
        self.locked = 0.0
        self.initialized = False
        self.reason = ""
        self.sold_value = 0.0
        self.available_at_open = 0.0

    def _targets(self, names: tuple[str, ...]) -> None:
        self.targets = {}
        for name in names:
            self.targets[name] = self.targets.get(name, 0) + 1 / len(names)

    def _selection(self) -> tuple[str, ...]:
        if self.spec.layer in ("hold", "equal"):
            return self.spec.fixed
        by_name = {data._name: data for data in self.datas}
        scores = {
            name: float(getattr(by_name[name], f"mom{self.spec.lookback}")[0])
            for name in (*RISK, "SHY")
        }
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError("signal is not warmed up")
        ranked = sorted(RISK, key=lambda name: (-scores[name], RISK.index(name)))[: self.spec.top]
        selected = []
        for name in ranked:
            failed = self.spec.layer != "M" and scores[name] <= scores["SHY"]
            selected.append("SHY" if failed or name in self.blocked else name)
        return tuple(selected)

    def _cash_events(self, day: date) -> None:
        if self.last_entitlement == day:
            raise RuntimeError("duplicate open callback would double credit dividends")
        self.last_entitlement = day
        self.receivables = [record for record in self.receivables if record.release_date > day]
        self.locked = sum(record.net_amount for record in self.receivables)
        self.new_events = []
        if not self.start_day <= day <= self.end_day:
            return
        feeds = {data._name: data for data in self.datas}
        for event in self.events:
            if event.ex_date != day:
                continue
            qty = int(self.getposition(feeds[event.symbol]).size)
            if qty <= 0:
                continue
            if event.pay_date is None:
                raise RuntimeError("main-window event has no pay date")
            # Conservative: funds become usable only AFTER the payment date.
            release = next(
                (date_value for date_value in self._calendar() if date_value > event.pay_date),
                date.max,
            )
            gross = qty * event.amount
            record = Distribution(
                symbol=event.symbol,
                ex_date=day,
                pay_date=event.pay_date,
                release_date=release,
                qty=qty,
                per_share=event.amount,
                gross_amount=gross,
                net_amount=gross * (1 - self.settings.tax),
            )
            self.new_events.append(record)
            self.distributions.append(record)
            self.broker.add_cash(record.net_amount)

    def _calendar(self) -> tuple[date, ...]:
        # Calendar metadata is predetermined, with no future prices consulted.
        return self.p.calendar_days  # type: ignore[no-any-return]

    @override
    def prenext(self) -> None:
        self.next()

    @override
    def prenext_open(self) -> None:
        self.next_open()

    @override
    def next_open(self) -> None:
        day = self.datas[0].datetime.date(0)
        if self.pending:
            raise RuntimeError("uncompleted order carried to next open")
        self._cash_events(day)
        if not self.start_day <= day <= self.end_day:
            return
        if not self.initialized:
            if self.targets is None:
                # Synthetic tests without warmup may request a fixed allocation.
                if self.spec.layer not in ("equal", "hold"):
                    raise RuntimeError("missing prior-close first signal")
                self._targets(self.spec.fixed)
            self.initialized = True
        if day == self.end_day:
            self.targets = {}
            self.exit_only.clear()
            self.reason = "terminal"
        if self.targets is None:
            return
        slip = self.settings.slip_bps / 10000
        before_cash = float(self.broker.getcash())
        available = before_cash - self.locked
        if available < -0.00001:
            raise RuntimeError("receivable was spent before payment")
        positions = {data._name: int(self.getposition(data).size) for data in self.datas}
        opens = {data._name: float(data.open[0]) for data in self.datas}
        investable = available + sum(positions[name] * opens[name] for name in positions)
        target_qty: dict[str, int] = {}
        for name in positions:
            weight = self.targets.get(name, 0)
            target_qty[name] = math.floor(
                investable * self.settings.invested * weight / (opens[name] * (1 + slip))
            )
        if self.exit_only:
            # Nonweekly risk exits leave all unaffected share counts untouched.
            target_qty = positions.copy()
            rotation_budget = 0.0
            for name in self.exit_only:
                size = positions[name]
                price = opens[name] * (1 - slip)
                rotation_budget += size * price - commission(-size, price, self.settings.fees)
                target_qty[name] = 0
            shy_price = opens["SHY"] * (1 + slip)
            add = max(0, math.floor(rotation_budget / shy_price))
            while (
                add > 0
                and add * shy_price + commission(add, shy_price, self.settings.fees)
                > rotation_budget
            ):
                add -= 1
            target_qty["SHY"] += add
        # Submit sells before buys; expected proceeds never change broker cash.
        for data in self.datas:
            name = data._name
            delta = target_qty[name] - positions[name]
            if delta < 0:
                price = opens[name] * (1 - slip)
                fee = commission(delta, price, self.settings.fees)
                available += -delta * price - fee
                order = self.sell(data=data, size=-delta)
                order.addinfo(reason=self.reason or "rebalance")
                self.pending.append(order)
        for data in self.datas:
            name = data._name
            delta = target_qty[name] - positions[name]
            if delta <= 0:
                continue
            price = opens[name] * (1 + slip)
            qty = min(delta, math.floor(max(0, available) / price))
            while qty > 0 and qty * price + commission(qty, price, self.settings.fees) > available:
                qty -= 1
            if qty <= 0:
                continue
            available -= qty * price + commission(qty, price, self.settings.fees)
            order = self.buy(data=data, size=qty)
            order.addinfo(reason=self.reason or "rebalance")
            self.pending.append(order)
        self.available_at_open = before_cash - self.locked
        self.targets = None
        self.exit_only.clear()
        self.reason = ""

    @override
    def next(self) -> None:
        day = self.datas[0].datetime.date(0)
        if self.pending:
            raise RuntimeError("market orders did not fully complete at open")
        # broker.next() has now applied this open's queued receivables.
        self.receivables.extend(self.new_events)
        self.new_events = []
        self.locked = sum(record.net_amount for record in self.receivables)
        if day < self.start_day:
            future = next((item for item in self._calendar() if item > day), date.max)
            if future >= self.start_day:
                self._targets(self._selection())
            return
        if day > self.end_day:
            return
        exited: set[str] = set()
        for data in self.datas:
            name = data._name
            qty = int(self.getposition(data).size)
            if qty < 0:
                raise RuntimeError("unexpected short")
            if qty and name in self.episodes:
                episode = self.episodes[name]
                tr = float(data.tr[0])
                if episode.index == 0:
                    episode.index = tr * episode.entry_price / float(data.close[0])
                wealth = tr / episode.index
                episode.peak = max(episode.peak, wealth)
                if self.spec.layer in ("S", "T") and name != "SHY":
                    if wealth <= 0.90 * episode.peak:
                        exited.add(name)
                    elif self.spec.layer == "T" and wealth >= 1.20:
                        exited.add(name)
            self.holdings.append(
                HoldingRecord(day=day, symbol=name, size=qty, close=float(data.close[0]))
            )
        cash = float(self.broker.getcash())
        if cash - self.locked < -0.00001:
            raise RuntimeError("negative actual spendable cash")
        equity = float(self.broker.getvalue())
        market = sum(self.getposition(data).size * float(data.close[0]) for data in self.datas)
        self.daily.append(
            DailyRecord(
                day=day,
                equity=equity,
                cash=cash,
                locked=self.locked,
                available=cash - self.locked,
                positions=sum(bool(self.getposition(data).size) for data in self.datas),
                gross_exposure=market / equity if equity else 0,
            )
        )
        if day == self.end_day:
            if market != 0:
                raise RuntimeError("terminal liquidation incomplete")
            return
        forced = next((names for when, names in self.forced if when == day), None)
        weekly = bool(self.datas[0].week_end[0])
        if forced is not None:
            self._targets(forced)
        elif weekly and self.spec.layer != "hold":
            # A previously stopped instrument may re-enter only at this future
            # weekly close. Instruments stopped today remain blocked today.
            self.blocked = set(exited)
            self._targets(self._selection())
        if exited:
            self.blocked.update(exited)
            if self.targets is None:
                self.targets = {}
                self.exit_only = set(exited)
            for name in exited:
                self.targets["SHY"] = self.targets.get("SHY", 0) + self.targets.pop(name, 0)
            self.reason = "risk_exit"

    @override
    def notify_order(self, order: Any) -> None:
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status != order.Completed:
            raise RuntimeError(f"order not completed: {order.data._name} {order.getstatusname()}")
        name = order.data._name
        qty = int(order.executed.size)
        price = float(order.executed.price)
        self.orders.append(
            OrderRecord(
                symbol=name,
                day=order.data.datetime.date(0),
                size=qty,
                price=price,
                fee=float(order.executed.comm),
                reason=str(order.info.reason),
            )
        )
        if qty > 0:
            self.available_at_open -= qty * price + float(order.executed.comm)
            if name not in self.episodes:
                self.episodes[name] = Episode(
                    entry_day=order.data.datetime.date(0), entry_price=price
                )
        else:
            self.available_at_open += -qty * price - float(order.executed.comm)
            if not self.getposition(order.data).size:
                self.episodes.pop(name, None)
        if self.available_at_open < -0.00001:
            raise RuntimeError("fill spent locked receivable or exceeded cash")
        self.pending = [pending for pending in self.pending if pending.ref != order.ref]

    @override
    def notify_trade(self, trade: Any) -> None:
        if trade.isclosed:
            self.closed_trades.append(
                ClosedTrade(
                    symbol=trade.data._name,
                    opened=bt.num2date(trade.dtopen).date(),
                    closed=bt.num2date(trade.dtclose).date(),
                    gross_pnl=float(trade.pnl),
                    net_pnl_ex_dividends=float(trade.pnlcomm),
                    commission=float(trade.commission),
                )
            )
