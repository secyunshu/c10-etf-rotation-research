"""Causal phase-two adapter: one native broker, including annual policy changes."""

import math
import time
from datetime import date
from typing import Any

import backtrader as bt
import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ..analysis import analyze
from ..broker import BrokerFees, DailyFeed
from ..models import Dataset, Result, Settings, Spec
from .models import CORE, EXPANDED, AnnualChoice, FeatureStore, P2Spec, Simulation
from .strategy import Phase2Rotation


def run_phase2(
    datasets: list[Dataset],
    features: FeatureStore,
    spec: P2Spec,
    settings: Settings,
    start: date,
    end: date,
    choices: tuple[AnnualChoice, ...] = (),
) -> Simulation:
    started = time.process_time()
    period = settings.effective_research_period
    if (
        start >= end
        or start < date(period.first_year, 1, 1)
        or end > date(period.last_year, 12, 31)
    ):
        raise ValueError("invalid frozen research start/end")
    if (
        not math.isfinite(settings.initial_cash)
        or settings.initial_cash <= 0
        or not 0 < settings.invested <= 0.95
    ):
        raise ValueError("invalid initial capital or deployment limit")
    spec = spec.model_copy(deep=True)
    choices = tuple(choice.model_copy(deep=True) for choice in choices)
    years = set(range(start.year, end.year + 1))
    if choices:
        if len({choice.year for choice in choices}) != len(choices):
            raise ValueError("duplicate annual choice year")
        if not years <= {choice.year for choice in choices}:
            raise ValueError("annual choices missing research year")
        if any(choice.cutoff >= date(choice.year, 1, 1) for choice in choices):
            raise ValueError("annual choice cutoff must precede its year")
        if any(
            choice.spec is not None and choice.spec.universe != spec.universe for choice in choices
        ):
            raise ValueError("annual choice universe differs from policy")
    names = CORE if spec.universe == "core" else EXPANDED
    by_name = {data.symbol: data for data in datasets}
    if len(by_name) != len(datasets) or not set(names) <= by_name.keys():
        raise ValueError("missing complete requested universe or duplicate feeds")
    selected = [by_name[name] for name in names]
    split_keys = [(event.symbol, event.day) for data in selected for event in data.splits]
    if len(split_keys) != len(set(split_keys)):
        raise ValueError("duplicate split event before native account execution")
    source_index = pd.DatetimeIndex(selected[0].frame.index)
    if (
        source_index.empty
        or source_index.has_duplicates
        or source_index.hasnans
        or not source_index.is_monotonic_increasing
        or source_index.tz is not None
        or not source_index.equals(source_index.normalize())
    ):
        raise ValueError("invalid source calendar")
    if any(not data.frame.index.equals(source_index) for data in selected):
        raise ValueError("feeds not synchronized")
    if pd.Timestamp(start) not in source_index or pd.Timestamp(end) not in source_index:
        raise ValueError("research start/end coverage missing")
    observed = source_index[source_index <= pd.Timestamp(end)]
    active_specs = [choice.spec for choice in choices if choice.spec is not None] or [spec]
    for rule in active_specs:
        if rule.frequency == "hold" and rule.fixed and not rule.vol_target:
            continue
        first_year = next(
            (
                choice.year
                for choice in choices
                if choice.spec is not None and choice.spec.sid == rule.sid
            ),
            start.year,
        )
        first_open = max(start, date(first_year, 1, 1))
        needed = (
            max(
                (
                    *rule.windows,
                    200 if rule.filter in ("sma", "both") else 0,
                    63 if rule.vol_target or rule.allocation != "equal" else 0,
                )
            )
            + 1
        )
        if int((observed < pd.Timestamp(first_open)).sum()) < needed:
            raise ValueError(f"insufficient prior-close warmup: need {needed} observations")
    # Calendar-only extension fixes period boundaries even for shortened runs;
    # no prices after end are ever passed to Cerebro or to signal calculations.
    calendar = xcals.get_calendar("XNYS")
    extension = calendar.sessions_in_range(
        source_index[-1] + pd.Timedelta(days=1), source_index[-1] + pd.Timedelta(days=40)
    ).tz_localize(None)
    source_days = tuple(timestamp.date() for timestamp in source_index.union(extension))
    clipped_features: dict[str, pd.DataFrame] = {}
    frames: dict[str, pd.DataFrame] = {}
    for data in selected:
        if data.symbol not in features.frames:
            raise ValueError(f"missing features: {data.symbol}")
        feature = features.frames[data.symbol]
        if not isinstance(feature.index, pd.DatetimeIndex):
            raise ValueError("feature calendar must be a DatetimeIndex")
        feature = feature.reindex(observed).copy()
        if (
            feature.index.has_duplicates
            or not feature.index.is_monotonic_increasing
            or not observed.isin(feature.index).all()
        ):
            raise ValueError("missing or invalid feature calendar")
        tr = feature.loc[observed, "tr"].to_numpy(dtype=float)
        if not np.isfinite(tr).all() or (tr <= 0).any():
            raise ValueError("invalid gross total-return input")
        frame = data.frame.loc[observed].copy()
        values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("invalid raw execution data")
        if (
            (frame.high < frame[["open", "close", "low"]].max(axis=1))
            | (frame.low > frame[["open", "close", "high"]].min(axis=1))
        ).any():
            raise ValueError("invalid execution OHLC")
        frame["tr"] = tr
        # Only the inherited ledger reads these feed lines; signals use FeatureStore.
        for column in ("mom21", "mom63", "mom126", "week_end"):
            if column not in frame:
                frame[column] = 0.0
        clipped_features[data.symbol] = feature
        frames[data.symbol] = frame
        for event in data.dividends:
            if event.symbol != data.symbol or (
                event.pay_date is not None and event.pay_date < event.ex_date
            ):
                raise ValueError("invalid cash event")
            if start <= event.ex_date <= end and event.pay_date is None:
                raise ValueError("missing main-window dividend payment date")
    dummy = Spec(name=spec.sid, layer="hold", lookback=spec.windows[0], top=spec.top)
    cerebro: Any = bt.Cerebro(stdstats=False, cheat_on_open=True)
    for name in names:
        cerebro.adddata(DailyFeed(dataname=frames[name], timeframe=bt.TimeFrame.Days), name=name)
    cerebro.broker.setcash(settings.initial_cash)
    cerebro.broker.set_shortcash(False)
    cerebro.broker.set_coc(False)
    cerebro.broker.set_coo(True)
    cerebro.broker.set_checksubmit(True)
    cerebro.broker.set_fundmode(False)
    cerebro.broker.addcommissioninfo(BrokerFees(schedule=settings.fees))
    cerebro.broker.set_slippage_perc(
        settings.slip_bps / 10000,
        slip_open=True,
        slip_match=True,
        slip_out=True,
    )
    cerebro.addstrategy(
        Phase2Rotation,
        spec=dummy,
        phase2_spec=spec,
        features=FeatureStore(frames=clipped_features),
        choices=choices,
        splits=tuple(event for data in selected for event in data.splits),
        settings=settings,
        start=start,
        end=end,
        calendar_days=source_days,
        events=tuple(
            event for data in selected for event in data.dividends if event.ex_date <= end
        ),
    )
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="native", timeframe=bt.TimeFrame.Days, fund=False
    )
    strategy: Phase2Rotation = cerebro.run(runonce=True, preload=True)[0]
    growth = 1.0
    for timestamp, value in strategy.analyzers.native.get_analysis().items():
        if start <= timestamp.date() <= end:
            growth *= 1 + value
    if strategy.pending:
        raise RuntimeError("orders left open")
    account = Result(
        spec=dummy,
        orders=strategy.orders,
        daily=strategy.daily,
        holdings=strategy.holdings,
        distributions=strategy.distributions,
        trades=strategy.closed_trades,
        native_total_return=growth - 1,
        splits=[event for data in selected for event in data.splits if start <= event.day <= end],
    )
    if not account.daily or account.daily[-1].day != end:
        raise RuntimeError("incomplete account ledger")
    analyze(account, settings)
    return Simulation(
        spec=spec,
        account=account,
        faults=strategy.faults,
        elapsed_cpu=time.process_time() - started,
    )
