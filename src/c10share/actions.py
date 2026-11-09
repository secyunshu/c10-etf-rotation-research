"""Original split and calendar methods; no matching or cash engine."""

from datetime import date
from typing import Any, override

import exchange_calendars as xcals
import pandas as pd

from rotbt.phase2.strategy import Phase2Rotation


class HistoricalRotation(Phase2Rotation):
    @override
    def _cash_events(self, day: date) -> None:
        for event in self.p.splits:
            if event.day != day:
                continue
            data = next(data for data in self.datas if data._name == event.symbol)
            position = self.getposition(data)
            quantity = position.size * event.shares_after_per_before
            if abs(quantity - round(quantity)) > 1e-8:
                raise ValueError("fractional split requires actual cash-in-lieu evidence")
            if position.size:
                position.fix(round(quantity), position.price / event.shares_after_per_before)
                # Backtrader's trade bookkeeping must use the same share units
                # before a later native sell. No cash, P&L or commissions added.
                native_strategy: Any = self
                native_trades = native_strategy._trades
                for trades in native_trades[data].values():
                    for trade in trades:
                        if trade.isopen:
                            trade.size *= event.shares_after_per_before
                            trade.price /= event.shares_after_per_before
        super()._cash_events(day)

    @staticmethod
    def scheduled(day: date, frequency: str, months: tuple[int, ...]) -> bool:
        calendar = xcals.get_calendar("XNYS")
        stamp = pd.Timestamp(day)
        if not calendar.is_session(stamp):
            return False
        following = calendar.next_session(stamp).date()
        if frequency == "M":
            return (not months or day.month in months) and day.month != following.month
        weekly = day.isocalendar()[:2] != following.isocalendar()[:2]
        return weekly and (frequency != "2W" or (day - date(2012, 1, 2)).days // 7 % 2 == 0)
