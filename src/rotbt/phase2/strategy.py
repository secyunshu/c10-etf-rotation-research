"""Phase-two signals/sizing on the unchanged native Backtrader account."""

import math
from bisect import bisect_right
from datetime import date
from typing import Any, override

from ..broker import commission
from ..models import Distribution
from ..strategy import Rotation
from .models import (
    AnnualChoice,
    Fault,
    FeatureStore,
    HeldAsset,
    P2Spec,
    SignalContext,
    TargetSignal,
)


def signal(spec: P2Spec, features: FeatureStore, context: SignalContext) -> TargetSignal:
    # The catalog/signal implementation is written independently of execution.
    from .signals import signal as calculate

    return calculate(spec, features, context)


class Phase2Rotation(Rotation):
    params: Any = (("phase2_spec", None), ("features", None), ("choices", ()), ("splits", ()))

    @override
    def __init__(self) -> None:
        super().__init__()
        self.p2: P2Spec | None = self.p.phase2_spec
        self.features: FeatureStore = self.p.features
        self.choices: tuple[AnnualChoice, ...] = self.p.choices
        self.by_year = {choice.year: choice for choice in self.choices}
        self.faults: list[Fault] = []
        self.failed: set[str] = set()
        self.active_year: int | None = None
        self._days = self._calendar()
        self._ordinal = {day: i for i, day in enumerate(self._days)}
        self._next_day = dict(zip(self._days[:-1], self._days[1:], strict=True))
        self._year_ends = {
            day for day, following in self._next_day.items() if day.year != following.year
        }
        self._events_by_day = {
            day: tuple(event for event in self.events if event.ex_date == day)
            for day in {event.ex_date for event in self.events}
        }
        self._release: dict[date, date] = {}
        for event in self.events:
            if event.pay_date is not None:
                i = bisect_right(self._days, event.pay_date)
                self._release[event.pay_date] = self._days[i] if i < len(self._days) else date.max
        self._execute_on: date | None = None
        self._force = False

    @override
    def _cash_events(self, day: date) -> None:
        """The phase-one dividend adapter with indexed calendar/event lookups."""
        if self.last_entitlement == day:
            raise RuntimeError("duplicate open callback would double credit dividends")
        self.last_entitlement = day
        self.receivables = [record for record in self.receivables if record.release_date > day]
        self.locked = sum(record.net_amount for record in self.receivables)
        self.new_events = []
        if not self.start_day <= day <= self.end_day:
            return
        feeds = {data._name: data for data in self.datas}
        for event in self._events_by_day.get(day, ()):
            qty = int(self.getposition(feeds[event.symbol]).size)
            if qty <= 0:
                continue
            if event.pay_date is None:
                raise ValueError("main-window event has no pay date")
            record = Distribution(
                symbol=event.symbol,
                ex_date=day,
                pay_date=event.pay_date,
                release_date=self._release[event.pay_date],
                qty=qty,
                per_share=event.amount,
                gross_amount=qty * event.amount,
                net_amount=qty * event.amount * (1 - self.settings.tax),
            )
            self.new_events.append(record)
            self.distributions.append(record)
            self.broker.add_cash(record.net_amount)

    def _activate(self, year: int) -> None:
        if year == self.active_year:
            return
        self.active_year = year
        if self.choices:
            self.p2 = self.by_year[year].spec
            self.blocked.clear()

    def _normal_close(self, day: date) -> bool:
        if self.p2 is None or self.p2.sid in self.failed or self.p2.frequency == "hold":
            return False
        following = self._next_day.get(day)
        if following is None:
            return False
        if self.p2.phase_shift:
            if self.p2.frequency == "weekly":
                return day.weekday() == 1
            month_days = [d for d in self._days if (d.year, d.month) == (day.year, day.month)]
            return len(month_days) >= 5 and day == month_days[4]
        if self.p2.frequency == "weekly":
            return day.isocalendar()[:2] != following.isocalendar()[:2]
        return (day.year, day.month) != (following.year, following.month)

    def _stage(
        self, weights: dict[str, float], day: date, reason: str, *, force: bool = False
    ) -> None:
        self.targets = weights
        self.reason = reason
        self._force = force
        # force controls the trade band only. Delay is a property of the signal,
        # including risk exits and annual rebuilds; liquidation is handled at open.
        delay = self.p2.delay_days if self.p2 is not None else 0
        ordinal = self._ordinal[day] + 1 + delay
        self._execute_on = self._days[ordinal] if ordinal < len(self._days) else date.max

    def _decide(self, day: date, reason: str, *, force: bool = False) -> None:
        if self.p2 is None or self.p2.sid in self.failed:
            self._stage({"SHY": self.settings.invested}, day, reason, force=True)
            return
        context = SignalContext(
            day=day,
            holdings=tuple(
                HeldAsset(
                    symbol=data._name,
                    shares=int(self.getposition(data).size),
                    age=self._ordinal[day] - self._ordinal[self.episodes[data._name].entry_day],
                )
                for data in self.datas
                if self.getposition(data).size and data._name in self.episodes
            ),
            blocked=tuple(data._name for data in self.datas if data._name in self.blocked),
        )
        result = signal(self.p2, self.features, context)
        if result.numerical_failure:
            self.failed.add(self.p2.sid)
            self.faults.append(
                Fault(
                    day=day, spec_id=self.p2.sid, reason=result.error or "numerical solver failure"
                )
            )
            self._stage({"SHY": self.settings.invested}, day, "solver_fault", force=True)
            return
        weights = {item.symbol: item.weight for item in result.allocations}
        names = {data._name for data in self.datas}
        if (
            len(weights) != len(result.allocations)
            or not weights.keys() <= names
            or any(not math.isfinite(w) or w < 0 for w in weights.values())
            or sum(weights.values()) > self.settings.invested + 1e-8
        ):
            raise ValueError("invalid signal account weights (already include deployment)")
        # Risk exits cannot be undone by a signal at the same close.
        for name in self.blocked:
            weights["SHY"] = weights.get("SHY", 0) + weights.pop(name, 0)
        self._stage(weights, day, reason, force=force)

    def _risk_exits(self) -> set[str]:
        exited: set[str] = set()
        rule = self.p2
        if rule is None or rule.sid in self.failed:
            return exited
        for data in self.datas:
            name = data._name
            if name == "SHY" or not self.getposition(data).size:
                continue
            episode = self.episodes[name]
            wealth = float(data.tr[0]) / episode.index
            threshold = rule.stop_value
            if rule.stop == "volatility":
                day = data.datetime.date(0)
                series = self.features.frames[name]["vol20"]
                sigma = float(series.iloc[self._ordinal[day]])
                if not math.isfinite(sigma) or sigma < 0:
                    raise ValueError("invalid causal vol20 for trailing exit")
                threshold = min(0.20, max(0.05, rule.stop_value * sigma * math.sqrt(21 / 252)))
            if (rule.stop != "none" and wealth <= (1 - threshold) * episode.peak) or (
                rule.take_profit > 0 and wealth >= 1 + rule.take_profit
            ):
                exited.add(name)
        return exited

    @override
    def next(self) -> None:
        # The dummy phase-one spec is hold: this only updates its proven ledger,
        # gross-TR episode anchor/peak, and receivable lifecycle, never signals.
        day: date = self.datas[0].datetime.date(0)
        following = self._next_day.get(day, date.max)
        if day < self.start_day:
            self.targets = None
            if following == self.start_day:
                self._activate(following.year)
                self._decide(day, "initial")
            return
        super().next()
        if day >= self.end_day:
            return
        if self.choices and following.year != day.year and following <= self.end_day:
            self.targets = None
            self.exit_only.clear()
            self._activate(following.year)
            self._decide(day, "annual_rebuild", force=True)
            return
        exited = self._risk_exits()
        normal = self._normal_close(day)
        if normal:
            self.blocked = set(exited)
        else:
            self.blocked.update(exited)
        waiting = self.targets is not None
        if self.reason == "risk_exit" and waiting:
            # The same still-held position can trigger on successive closes.
            # Preserve the first trigger's due date rather than postpone forever.
            self.exit_only.update(exited)
            return
        if normal and not waiting and not (self.choices and following in self._year_ends):
            self._decide(day, "risk_exit" if exited else "rebalance")
            self.exit_only.clear()
        elif exited and not exited <= self.exit_only:
            self.exit_only.update(exited)
            self._stage({}, day, "risk_exit", force=True)

    @override
    def next_open(self) -> None:
        day: date = self.datas[0].datetime.date(0)
        if self.pending:
            raise RuntimeError("uncompleted order carried to next open")
        self._cash_events(day)
        if not self.start_day <= day <= self.end_day:
            return
        if not self.initialized:
            if self.targets is None:
                if (
                    self.p2 is None
                    or self.p2.frequency != "hold"
                    or not self.p2.fixed
                    or self.p2.vol_target
                    or self.p2.allocation != "equal"
                ):
                    raise RuntimeError("missing prior-close first signal")
                self.targets = {}
                for name in self.p2.fixed:
                    self.targets[name] = self.targets.get(name, 0) + self.settings.invested / len(
                        self.p2.fixed
                    )
                self.reason = "initial"
            self.initialized = True
        if day == self.end_day or (self.choices and day in self._year_ends):
            self.targets = {}
            self.exit_only.clear()
            self.reason = "terminal" if day == self.end_day else "annual_liquidation"
            self._execute_on = day
            self._force = True
        if self.targets is None or (self._execute_on is not None and day < self._execute_on):
            return
        slip = self.settings.slip_bps / 10000
        before_cash = float(self.broker.getcash())
        available = before_cash - self.locked
        if available < -0.00001:
            raise RuntimeError("receivable was spent before payment")
        positions = {data._name: int(self.getposition(data).size) for data in self.datas}
        opens = {data._name: float(data.open[0]) for data in self.datas}
        investable = available + sum(positions[name] * opens[name] for name in positions)
        # Signal allocations are ACCOUNT weights, already multiplied by 95%.
        target_qty = {
            name: math.floor(investable * self.targets.get(name, 0) / (opens[name] * (1 + slip)))
            for name in positions
        }
        if self.exit_only:
            target_qty = positions.copy()
            proceeds = 0.0
            for name in self.exit_only:
                size = positions[name]
                if size:
                    price = opens[name] * (1 - slip)
                    proceeds += size * price - commission(-size, price, self.settings.fees)
                target_qty[name] = 0
            price = opens["SHY"] * (1 + slip)
            qty = max(0, math.floor(proceeds / price))
            while qty and qty * price + commission(qty, price, self.settings.fees) > proceeds:
                qty -= 1
            target_qty["SHY"] += qty
        elif not self._force and self.p2 is not None and self.p2.band > 0:
            for name in positions:
                if positions[name] and self.targets.get(name, 0) > 0:
                    difference = abs(
                        investable * self.targets[name] - positions[name] * opens[name]
                    )
                    if difference / investable < self.p2.band:
                        target_qty[name] = positions[name]
        retained = sum(min(positions[name], target_qty[name]) * opens[name] for name in positions)
        deployment = max(0.0, self.settings.invested * investable - retained)
        # Expected proceeds constrain sizing only; native broker settles fills.
        for data in self.datas:
            name = data._name
            delta = target_qty[name] - positions[name]
            if delta < 0:
                price = opens[name] * (1 - slip)
                available += -delta * price - commission(delta, price, self.settings.fees)
                order = self.sell(data=data, size=-delta)
                order.addinfo(reason=self.reason)
                self.pending.append(order)
        for data in self.datas:
            name = data._name
            delta = target_qty[name] - positions[name]
            if delta <= 0:
                continue
            price = opens[name] * (1 + slip)
            qty = min(
                delta, math.floor(max(0.0, available) / price), math.floor(deployment / price)
            )
            while qty and qty * price + commission(qty, price, self.settings.fees) > available:
                qty -= 1
            if qty:
                available -= qty * price + commission(qty, price, self.settings.fees)
                deployment = max(0.0, deployment - qty * price)
                order = self.buy(data=data, size=qty)
                order.addinfo(reason=self.reason)
                self.pending.append(order)
        self.available_at_open = before_cash - self.locked
        self.targets = None
        self.exit_only.clear()
        self.reason = ""
        self._execute_on = None
        self._force = False
