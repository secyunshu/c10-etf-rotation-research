"""Frozen2026 orchestration port; Backtrader owns broker accounting."""

import math
from datetime import date
from pathlib import Path
from typing import Any, override

import backtrader as bt
import exchange_calendars as xcals
import pandas as pd

from c10share.actions import HistoricalRotation
from rotbt.broker import BrokerFees, DailyFeed
from rotbt.models import (
    ClosedTrade,
    DailyRecord,
    Distribution,
    HoldingRecord,
    Model,
    OrderRecord,
    Result,
    Settings,
    Spec,
)
from rotbt.phase2.models import FeatureStore, P2Spec
from rotbt.phase2.strategy import Phase2Rotation

from .models import BOOKS, Artifact, Contract, NativeReturn, OpenTrade, Prepared, TargetRecord
from .targets import feed_names as selected_feeds
from .targets import target


def write(path: Path, value: Model) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value.model_dump_json(indent=2) + "\n")


def feed_names(contract: Contract, prepared: Prepared) -> tuple[str, ...]:
    return selected_feeds(contract, prepared.risk_names)


def decision_days(contract: Contract) -> tuple[date, ...]:
    if contract.rule.endswith("_BH"):
        return ()
    calendar = xcals.get_calendar("XNYS")
    return tuple(
        t.date()
        for t in calendar.sessions_in_range(contract.start, contract.through)
        if t.date() < contract.through and t.month != calendar.next_session(t).month
    )


def simulate(prepared: Prepared, contract: Contract, settings: Settings, folder: Path) -> Artifact:
    trace: list[TargetRecord] = []
    instances: list[Any] = []
    names = feed_names(contract, prepared)
    selected = [next(d for d in prepared.datasets if d.symbol == name) for name in names]
    scheduled = set(decision_days(contract))

    class FixedRotation(HistoricalRotation):
        @override
        def __init__(self) -> None:
            Phase2Rotation.__init__(self)
            instances.append(self)

        @override
        def _normal_close(self, day: date) -> bool:
            return day in scheduled

        @override
        def _decide(self, day: date, reason: str, *, force: bool = False) -> None:
            if contract.rule.endswith("_BH"):
                q = {contract.rule.removesuffix("_BH"): 1.0}
            elif contract.rule == "SPY60_IEF40":
                q = {"SPY": 0.6, "IEF": 0.4}
            else:
                q = target(contract, prepared, day).q
            trace.append(TargetRecord(day=day, q=q, reason=reason))
            self._stage(
                {s: w * self.settings.invested for s, w in q.items()}, day, reason, force=force
            )

    hold = contract.rule.endswith("_BH")
    spec = P2Spec(
        label=contract.rule,
        universe="expanded",
        windows=(252,),
        top=3,
        frequency="hold" if hold else "monthly",
        fixed=(contract.rule.removesuffix("_BH"),) if hold else (),
    )
    dummy = Spec(name=spec.sid, layer="hold", lookback=252, top=3)
    calendar = xcals.get_calendar("XNYS")
    finite_days = tuple(
        t.date()
        for t in calendar.sessions_in_range(
            prepared.first, pd.Timestamp(contract.through) + pd.Timedelta(days=80)
        )
    )
    cerebro: Any = bt.Cerebro(stdstats=False, cheat_on_open=True)
    for data in selected:
        frame = data.frame.loc[: str(contract.through)].copy()
        for field in ("mom21", "mom63", "mom126", "week_end"):
            frame[field] = 0.0
        cerebro.adddata(DailyFeed(dataname=frame, timeframe=bt.TimeFrame.Days), name=data.symbol)
    cerebro.broker.setcash(50000)
    cerebro.broker.set_shortcash(False)
    cerebro.broker.set_coc(False)
    cerebro.broker.set_coo(True)
    cerebro.broker.set_checksubmit(True)
    cerebro.broker.set_fundmode(False)
    cerebro.broker.addcommissioninfo(BrokerFees(schedule=settings.fees))
    cerebro.broker.set_slippage_perc(
        contract.cost / 10000, slip_open=True, slip_match=True, slip_out=True
    )
    cerebro.addstrategy(
        FixedRotation,
        spec=dummy,
        phase2_spec=spec,
        features=FeatureStore(
            frames={
                name: frame.loc[: str(contract.through)].copy()
                for name, frame in prepared.features.frames.items()
            }
        ),
        choices=(),
        splits=tuple(s for d in selected for s in d.splits if s.day <= contract.through),
        settings=settings,
        start=contract.start,
        end=contract.through if contract.mode == "bridge" else date.max,
        calendar_days=finite_days,
        events=tuple(e for d in selected for e in d.dividends if e.ex_date <= contract.through),
    )
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="native", timeframe=bt.TimeFrame.Days, fund=False
    )

    def bundle(strategy: Any, returns: list[NativeReturn], trades: list[OpenTrade]) -> Artifact:
        return Artifact(
            contract=contract,
            targets=trace,
            native_returns=returns,
            open_trades=trades,
            account=Result(
                spec=dummy,
                orders=strategy.orders,
                daily=strategy.daily,
                holdings=strategy.holdings,
                distributions=strategy.distributions,
                trades=strategy.closed_trades,
                native_total_return=math.prod(1 + r.value for r in returns) - 1,
                splits=[
                    s
                    for d in selected
                    for s in d.splits
                    if contract.start <= s.day <= contract.through
                ],
            ),
        )

    try:
        strategy: Any = cerebro.run(runonce=True, preload=True)[0]
        native = [
            NativeReturn(day=stamp.date(), value=value)
            for stamp, value in strategy.analyzers.native.get_analysis().items()
            if contract.start <= stamp.date() <= contract.through
        ]
        trades = [
            OpenTrade(
                symbol=feed._name,
                size=int(trade.size),
                price=trade.price,
                pnl=trade.pnl,
                commission=trade.commission,
            )
            for feed, by_id in strategy._trades.items()
            for group in by_id.values()
            for trade in group
            if trade.isopen
        ]
        result = bundle(strategy, native, trades)
        write(folder / "ARTIFACT.json", result)
        if strategy.pending or strategy.faults:
            raise ValueError("native pending order or execution fault")
        return result
    except BaseException:
        if instances and not (folder / "ARTIFACT.json").exists():
            write(folder / "PARTIAL_NATIVE.json", bundle(instances[0], [], []))
        raise


def export_books(folder: Path, artifact: Artifact) -> None:
    account = artifact.account
    rows: tuple[list[Model], ...] = (
        list(account.daily),
        list(account.orders),
        list(account.holdings),
        list(account.distributions),
        list(account.trades),
    )
    models: tuple[type[Model], ...] = (
        DailyRecord,
        OrderRecord,
        HoldingRecord,
        Distribution,
        ClosedTrade,
    )
    for name, records, model in zip(BOOKS, rows, models, strict=True):
        pd.DataFrame(
            [r.model_dump(mode="json") for r in records], columns=list(model.model_fields)
        ).to_csv(folder / f"{name}.csv.gz", index=False, mode="x")
