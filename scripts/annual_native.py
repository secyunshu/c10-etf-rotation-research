"""Annual signal configuration; original Backtrader broker owns the continuous account."""

import math
from datetime import date
from pathlib import Path
from typing import Any, override

import backtrader as bt
import exchange_calendars as xcals
import pandas as pd

from c10share.actions import HistoricalRotation
from c10standard.audit import FinancialScope, equal, reconcile, reconcile_trade_books
from c10standard.models import BOOKS, NativeReturn, OpenTrade, TargetRecord
from c10standard.native import write
from rotbt.broker import BrokerFees, DailyFeed
from rotbt.models import (
    ClosedTrade,
    DailyRecord,
    Distribution,
    HoldingRecord,
    Model,
    OrderRecord,
    ResearchPeriod,
    Result,
    Settings,
    Spec,
)
from rotbt.phase2.models import P2Spec
from rotbt.phase2.strategy import Phase2Rotation
from scripts.annual_models import Artifact, Contract, Prepared
from scripts.annual_signals import decision_timing, feed_names, oracle, target


def signal_days(contract: Contract) -> tuple[date, ...]:
    calendar = xcals.get_calendar("XNYS")
    initial = calendar.previous_session(pd.Timestamp(contract.start)).date()
    return (initial, *decision_days(contract))


def decision_days(contract: Contract) -> tuple[date, ...]:
    calendar = xcals.get_calendar("XNYS")
    return tuple(t.date() for t in calendar.sessions_in_range(contract.start, contract.through)
                 if t.date() < contract.through and t.month != calendar.next_session(t).month)


def settings_for(prepared: Prepared, contract: Contract) -> Settings:
    return Settings(fees=prepared.fees, initial_cash=50000, invested=.95,
                    slip_bps=contract.cost_bps, tax=0,
                    research_period=ResearchPeriod(first_year=contract.start.year,
                                                   last_year=2025, split_year=2023))


def simulate(prepared: Prepared, contract: Contract, settings: Settings, folder: Path) -> Artifact:
    if settings != settings_for(prepared, contract):
        raise ValueError("native settings differ from the registered account")
    trace: list[TargetRecord] = []
    instances: list[Any] = []
    names = feed_names(contract, prepared)
    selected = [next(d for d in prepared.datasets if d.symbol == name) for name in names]
    scheduled = set(decision_days(contract))

    class AnnualRotation(HistoricalRotation):
        @override
        def __init__(self) -> None:
            Phase2Rotation.__init__(self)
            instances.append(self)

        @override
        def _normal_close(self, day: date) -> bool:
            return day in scheduled

        @override
        def _decide(self, day: date, reason: str, *, force: bool = False) -> None:
            q = target(contract, prepared, day).q
            trace.append(TargetRecord(day=day, q=q, reason=reason))
            self._stage({s: w * self.settings.invested for s, w in q.items()},
                        day, reason, force=force)

    # Keep the published inert spec identifier so bridge books remain byte-equivalent.
    spec = P2Spec(label="C11", universe="expanded", windows=(252,),
                  top=3, frequency="monthly")
    dummy = Spec(name=spec.sid, layer="hold", lookback=252, top=3)
    finite_days = tuple(t.date() for t in xcals.get_calendar("XNYS").sessions_in_range(
        prepared.first, pd.Timestamp(contract.through) + pd.Timedelta(days=80)))
    cerebro: Any = bt.Cerebro(stdstats=False, cheat_on_open=True)
    for data in selected:
        frame = data.frame.loc[:str(contract.through)].copy()
        for field in ("mom21", "mom63", "mom126", "week_end"):
            frame[field] = 0.
        cerebro.adddata(DailyFeed(dataname=frame, timeframe=bt.TimeFrame.Days), name=data.symbol)
    cerebro.broker.setcash(50000)
    cerebro.broker.set_shortcash(False)
    cerebro.broker.set_coc(False)
    cerebro.broker.set_coo(True)
    cerebro.broker.set_checksubmit(True)
    cerebro.broker.set_fundmode(False)
    cerebro.broker.addcommissioninfo(BrokerFees(schedule=settings.fees))
    cerebro.broker.set_slippage_perc(contract.cost_bps / 10000, slip_open=True,
                                    slip_match=True, slip_out=True)
    cerebro.addstrategy(AnnualRotation, spec=dummy, phase2_spec=spec, features=prepared.features,
                        choices=(), splits=tuple(s for d in selected for s in d.splits),
                        settings=settings, start=contract.start, end=contract.through,
                        calendar_days=finite_days,
                        events=tuple(e for d in selected for e in d.dividends))
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="native", timeframe=bt.TimeFrame.Days,
                        fund=False)

    def bundle(strategy: Any, returns: list[NativeReturn], trades: list[OpenTrade]) -> Artifact:
        account = Result(spec=dummy, orders=strategy.orders, daily=strategy.daily,
                         holdings=strategy.holdings, distributions=strategy.distributions,
                         trades=strategy.closed_trades,
                         native_total_return=math.prod(1 + r.value for r in returns) - 1,
                         splits=[s for d in selected for s in d.splits
                                 if contract.start <= s.day <= contract.through])
        return Artifact(contract=contract, targets=trace, native_returns=returns,
                        open_trades=trades, account=account,
                        decision_times=[decision_timing(contract, prepared, r.day) for r in trace])

    try:
        strategy: Any = cerebro.run(runonce=True, preload=True)[0]
        returns = [NativeReturn(day=stamp.date(), value=value)
                   for stamp, value in strategy.analyzers.native.get_analysis().items()
                   if contract.start <= stamp.date() <= contract.through]
        trades = [OpenTrade(symbol=feed._name, size=int(t.size), price=t.price, pnl=t.pnl,
                            commission=t.commission) for feed, by_id in strategy._trades.items()
                  for group in by_id.values() for t in group if t.isopen]
        artifact = bundle(strategy, returns, trades)
        write(folder / "ARTIFACT.json", artifact)
        if strategy.pending or strategy.faults:
            raise ValueError("native pending order or execution fault")
        return artifact
    except BaseException:
        if instances and not (folder / "ARTIFACT.json").exists():
            write(folder / "PARTIAL_NATIVE.json", bundle(instances[0], [], []))
        raise


def export_books(folder: Path, artifact: Artifact) -> None:
    account = artifact.account
    rows: tuple[list[Model], ...] = (list(account.daily), list(account.orders),
                                    list(account.holdings), list(account.distributions),
                                    list(account.trades))
    models: tuple[type[Model], ...] = (DailyRecord, OrderRecord, HoldingRecord,
                                      Distribution, ClosedTrade)
    for name, records, model in zip(BOOKS, rows, models, strict=True):
        pd.DataFrame([r.model_dump(mode="json") for r in records], columns=list(model.model_fields)
                     ).to_csv(folder / f"{name}.csv.gz", index=False, mode="x")


def audit(artifact: Artifact, prepared: Prepared) -> None:
    contract, account = artifact.contract, artifact.account
    names = feed_names(contract, prepared)
    selected = [next(d for d in prepared.datasets if d.symbol == name) for name in names]
    scope = FinancialScope(start=contract.start, through=contract.through,
                           settings=settings_for(prepared, contract), terminal=True)
    reconcile(account, selected, scope)
    if tuple(r.day for r in artifact.targets) != signal_days(contract):
        raise ValueError("signal calendar mismatch")
    if artifact.decision_times != [decision_timing(contract, prepared, r.day)
                                    for r in artifact.targets]:
        raise ValueError("price/selection/source/execution timing proof mismatch")
    calendar = xcals.get_calendar("XNYS")
    due = {calendar.next_session(pd.Timestamp(r.day)).date():
           "initial" if i == 0 else "rebalance" for i, r in enumerate(artifact.targets)}
    due[contract.through] = "terminal"
    if any(o.day not in due or o.reason != due[o.day] for o in account.orders):
        raise ValueError("order violates registered next-open/terminal schedule")
    if any(r.reason != ("initial" if i == 0 else "rebalance")
           for i, r in enumerate(artifact.targets)):
        raise ValueError("target reason differs from fixed signal calendar")
    for record in artifact.targets:
        expected = oracle(contract, prepared, record.day)
        if set(record.q) != set(expected.q):
            raise ValueError("independent target asset mismatch")
        for symbol, weight in record.q.items():
            equal(weight, expected.q[symbol], "independent target weight", 1e-12)
    if [r.day for r in artifact.native_returns] != [r.day for r in account.daily]:
        raise ValueError("native TimeReturn calendar mismatch")
    previous = 50000.
    for row, native in zip(account.daily, artifact.native_returns, strict=True):
        equal(native.value, row.equity / previous - 1, "daily native TimeReturn", 1e-12)
        previous = row.equity
    if artifact.open_trades or any(h.size for h in account.holdings if h.day == contract.through):
        raise ValueError("terminal liquidation must close native Position and Trade")
    if not account.orders or min(o.day for o in account.orders) != contract.start:
        raise ValueError("fresh account must execute first order on its registered start")
    equal(sum(t.commission for t in account.trades), sum(o.fee for o in account.orders),
          "native closed trade commissions")
    reconcile_trade_books(account, artifact.open_trades)
