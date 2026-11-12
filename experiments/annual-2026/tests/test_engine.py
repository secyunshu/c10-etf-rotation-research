"""Synthetic causal and account-lifecycle checks; never run a real native broker."""

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import backtrader as bt
import exchange_calendars as xcals
import pandas as pd
import pytest
from pydantic import ValidationError

from c10standard.models import Receipt
from rotbt.models import CashEvent, Dataset, FeeSchedule, SplitEvent
from rotbt.phase2.strategy import Phase2Rotation
from scripts import ytd_annual_native as native
from scripts import ytd_annual_run as run
from scripts import ytd_annual_signals as signals
from scripts.ytd_annual_models import (
    Binding,
    Contract,
    GroupMap,
    Pool,
    Prepared,
    Schedule,
    pool_sha256,
    schedule_sha256,
)

DAY = date(2025, 12, 31)
FEES = FeeSchedule(commission_per_share=.0049, commission_min=.99,
                   platform_per_share=.005, platform_min=1., settlement_per_share=.003,
                   sec_rate=.0000206, sec_min=.01, nms_per_share=.000003)


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
    monkeypatch.setattr(bt, "Cerebro", Mock(side_effect=AssertionError("native forbidden")))


def schedule(stage="science"):
    rows = []
    calendar = xcals.get_calendar("XNYS")
    for year in range(2022, 2026 if stage == "bridge" else 2027):
        groups = GroupMap(groups={"us_broad_style": ("QQQ", "SPY"),
                                  "technology_semiconductors": ("SOXX",),
                                  "precious_metals": ("GLD", "SLV") if year == 2026
                                  else ("GLD",)})
        names = tuple(sorted(s for members in groups.groups.values() for s in members))
        selection = date(year - 1, 12, 31)
        price = calendar.date_to_session(pd.Timestamp(selection), direction="previous").date()
        execute = calendar.next_session(pd.Timestamp(price)).date()
        rows.append(Pool(year=year, risk_names=names, groups=groups, price_asof=price,
                         selection_asof=selection, execute_on=execute, source_asof=selection,
                         sha256=pool_sha256(year, names, groups, price, selection, execute,
                                            selection)))
    entries = tuple(rows)
    return Schedule(stage=stage, entries=entries, sha256=schedule_sha256(stage, entries))


def market(scores=(.5, .4, .3, .2, .25, .1), observations=300, day=DAY,
           stage="science", role="ANNUAL"):
    days = pd.bdate_range(end=day, periods=observations)
    selected = schedule(stage)
    names = ({s for p in selected.entries for s in p.risk_names} | {"SHY"}
             if role == "ANNUAL" else {role.removesuffix("_BH"), "SHY"})
    datasets = []
    for symbol, score in zip(("QQQ", "SPY", "SOXX", "GLD", "SHY", "SLV"), scores,
                             strict=True):
        if symbol not in names:
            continue
        frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.,
                              "volume": 1e6, "turnover": 20e6,
                              "tr": [100.] * (observations - 1) + [100 * (1 + score)]},
                             index=days)
        datasets.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    return Prepared(role=role, stage=stage, schedule=selected, datasets=datasets,
                    features=signals.features_for(datasets), fees=FEES,
                    last=date(2025, 12, 31) if stage == "bridge" else date(2026, 11, 10))


def contract(prepared, cost=5):
    return Contract(role=prepared.role, cost_bps=cost, stage=prepared.stage,
                    schedule_sha256=prepared.schedule.sha256)


@pytest.mark.parametrize("price,year,execute", [
    (date(2021, 12, 31), 2022, date(2022, 1, 3)),
    (date(2022, 12, 30), 2023, date(2023, 1, 3)),
    (date(2023, 12, 29), 2024, date(2024, 1, 2)),
    (date(2024, 12, 31), 2025, date(2025, 1, 2)),
    (DAY, 2026, date(2026, 1, 2)),
])
def test_year_end_pool_belongs_to_next_execution_year(price, year, execute):
    prepared = market()
    timing = signals.decision_timing(contract(prepared), prepared, price)
    assert timing.pool_year == year
    assert timing.price_asof == price
    assert timing.selection_asof == date(year - 1, 12, 31)
    assert timing.source_asof == date(year - 1, 12, 31)
    assert timing.execute_on == execute


def test_2026_new_winner_cannot_be_ranked_before_2025_year_end():
    prepared = market(scores=(.5, .4, .3, .2, .25, 9.), day=date(2025, 11, 28))
    assert signals.target(contract(prepared), prepared, date(2025, 11, 28)).q == {
        "QQQ": .45, "SOXX": .45, "SHY": pytest.approx(.1)}
    assert "SLV" not in signals.active_pool(contract(prepared), prepared,
                                           date(2025, 11, 28)).risk_names


def test_2026_pool_is_valid_new_contract_without_bypassing_old_schema():
    from scripts.annual_models import Pool as OldPool

    new = schedule().entries[-1]
    assert new.year == 2026
    with pytest.raises(ValidationError):
        OldPool.model_validate(new.model_dump())
    invalid = new.model_dump()
    invalid["year"] = 2027
    with pytest.raises(ValidationError):
        Pool.model_validate(invalid)


def test_schedule_hash_and_stage_must_bind_every_active_pool():
    prepared = market()
    wrong = contract(prepared).model_copy(update={"schedule_sha256": "a" * 64})
    with pytest.raises(ValueError, match="bind"):
        signals.feed_names(wrong, prepared)
    changed = prepared.schedule.model_dump()
    changed["entries"][-1]["groups"]["groups"]["precious_metals"] = ("GLD",)
    with pytest.raises(ValidationError):
        Schedule.model_validate(changed)
    with pytest.raises(ValidationError):
        Contract(role="SPY_BH", stage="bridge", cost_bps=5, schedule_sha256="a" * 64)


def test_incomplete_or_future_market_scope_is_rejected():
    prepared = market()
    for day in (date(2026, 11, 10), date(2026, 11, 11)):
        with pytest.raises(ValueError, match="execution"):
            signals.active_pool(contract(prepared), prepared, day)
    frame = prepared.datasets[0].frame
    frame.loc[pd.Timestamp("2026-11-11")] = frame.iloc[-1]
    with pytest.raises(ValidationError, match="cutoff"):
        Prepared.model_validate(prepared.model_dump())


@pytest.mark.parametrize("scores,expected", [
    ((.5, .4, .3, .2, .25, .1), {"QQQ": .45, "SOXX": .45, "SHY": .1}),
    ((.5, .4, .2, .1, .25, .1), {"QQQ": .45, "SHY": .55}),
    ((.2, .1, .15, .1, .25, .1), {"SHY": 1.}),
])
def test_fixed_two_group_slots_and_shy_gate_preserve_account_weights(scores, expected):
    prepared = market(scores=scores)
    assert signals.target(contract(prepared), prepared, DAY).q == pytest.approx(expected)
    assert signals.oracle(contract(prepared), prepared, DAY).q == pytest.approx(expected)
    assert native.settings_for(prepared, contract(prepared)).invested == .95


def test_cash_momentum_requires_253_prices_and_risk_requires_eligibility():
    prepared = market(observations=252)
    with pytest.raises(ValueError, match="253"):
        signals.target(contract(prepared), prepared, DAY)
    prepared = market()
    for data in prepared.datasets:
        if data.symbol == "QQQ":
            data.frame.loc[data.frame.index[-1], "close"] = 4.99
        if data.symbol == "SOXX":
            data.frame.loc[data.frame.index[-1], "volume"] = 0
    prepared.features = signals.features_for(prepared.datasets)
    assert signals.target(contract(prepared), prepared, DAY).q == {"SPY": .45, "SHY": .55}


def test_full_prefix_and_raw_oracle_exclude_future_prices():
    prepared = market()
    expected = signals.target(contract(prepared), prepared, DAY)
    for data in prepared.datasets:
        data.frame.loc[pd.Timestamp("2026-01-02")] = data.frame.iloc[-1]
        data.frame.loc[pd.Timestamp("2026-01-02"), "tr"] = 1e9
    prepared.features = signals.features_for(prepared.datasets)
    assert signals.target(contract(prepared), prepared, DAY) == expected
    signals.verify_targets(contract(prepared), prepared, (DAY,))
    prepared.features.frames["QQQ"].loc[pd.Timestamp(DAY), "mom252"] = -1.
    with pytest.raises(ValueError, match="target"):
        signals.verify_targets(contract(prepared), prepared, (DAY,))


def test_signal_calendar_has_59_continuous_signals_and_48_bridge_signals():
    prepared = market()
    days = native.signal_days(contract(prepared))
    assert len(days) == 59
    assert days[0] == date(2021, 12, 31) and days[-1] == date(2026, 10, 30)
    assert DAY in days
    bridge = market(stage="bridge")
    assert len(native.signal_days(contract(bridge))) == 48
    assert native.signal_days(contract(bridge))[-1] == date(2025, 11, 28)
    for role in ("SPY_BH", "QQQ_BH"):
        benchmark = market(role=role)
        assert native.signal_days(contract(benchmark)) == (date(2021, 12, 31),)
        assert signals.target(contract(benchmark), benchmark, DAY).q == {
            role.removesuffix("_BH"): 1.}


@pytest.mark.parametrize("stage,expected_end", [("bridge", date(2025, 12, 31)),
                                               ("science", date.max)])
def test_native_account_keeps_one_cash_start_and_only_bridge_liquidates(
        monkeypatch, tmp_path, stage, expected_end):
    prepared = market(stage=stage)
    prepared.datasets[0].dividends = [CashEvent(
        symbol="QQQ", ex_date=DAY, pay_date=date(2026, 1, 5), amount=1., source="fixture")]
    prepared.datasets[0].splits = [SplitEvent(
        symbol="QQQ", day=DAY, shares_after_per_before=2., source="fixture")]
    broker = Mock()
    cerebro = Mock(broker=broker)
    cerebro.run.side_effect = RuntimeError("capture only")
    monkeypatch.setattr(bt, "Cerebro", Mock(return_value=cerebro))
    co = contract(prepared)
    with pytest.raises(RuntimeError, match="capture only"):
        native.simulate(prepared, co, native.settings_for(prepared, co), tmp_path)
    broker.setcash.assert_called_once_with(50000)
    options = cerebro.addstrategy.call_args.kwargs
    assert options["choices"] == () and options["start"] == date(2022, 1, 3)
    assert options["end"] == expected_end
    assert options["events"] == tuple(e for d in prepared.datasets for e in d.dividends)
    assert options["splits"] == tuple(s for d in prepared.datasets for s in d.splits)
    assert cerebro.addanalyzer.call_args.args[0] is bt.analyzers.TimeReturn
    assert {c.kwargs["name"] for c in cerebro.adddata.call_args_list} == {
        d.symbol for d in prepared.datasets}


def test_changed_native_settings_fail_before_broker_creation(tmp_path):
    prepared = market()
    co = contract(prepared)
    settings = native.settings_for(prepared, co).model_copy(update={"invested": .9})
    with pytest.raises(ValueError, match="settings"):
        native.simulate(prepared, co, settings, tmp_path)


def test_delisted_from_pool_receivable_stays_locked_until_next_session_after_payment():
    from rotbt.models import Distribution

    record = Distribution(symbol="REMOVED", ex_date=DAY, pay_date=date(2026, 1, 5),
                          release_date=date(2026, 1, 6), qty=10, per_share=1.,
                          gross_amount=10., net_amount=10.)
    fake = SimpleNamespace(last_entitlement=None, receivables=[record], distributions=[record],
                           start_day=date(2022, 1, 3), end_day=date.max,
                           datas=[], _events_by_day={})
    Phase2Rotation._cash_events(fake, date(2026, 1, 2))
    assert fake.locked == 10 and fake.receivables == [record]
    Phase2Rotation._cash_events(fake, date(2026, 1, 5))
    assert fake.locked == 10 and fake.receivables == [record]
    Phase2Rotation._cash_events(fake, date(2026, 1, 6))
    assert fake.locked == 0 and fake.distributions == [record]


def test_removed_holding_is_sold_at_new_year_open_without_capital_reset():
    fake = SimpleNamespace(p2=SimpleNamespace(delay_days=0),
                           _ordinal={DAY: 0}, _days=(DAY, date(2026, 1, 2)))
    Phase2Rotation._stage(fake, {"SHY": .95}, DAY, "rebalance")
    feeds = [SimpleNamespace(_name=name, open=[100.], datetime=Mock())
             for name in ("REMOVED", "SHY")]
    for feed in feeds:
        feed.datetime.date.return_value = date(2026, 1, 2)
    fake.datas, fake.pending, fake._cash_events = feeds, [], Mock()
    fake.start_day, fake.end_day = date(2022, 1, 3), date.max
    fake.initialized, fake.choices, fake.locked = True, (), 10.
    fake.settings = native.settings_for(market(), contract(market()))
    fake.broker = Mock()
    fake.broker.getcash.return_value = 1000.
    fake.getposition = lambda feed: SimpleNamespace(size=50 if feed._name == "REMOVED" else 0)
    fake.exit_only, fake.p2.band = set(), 0
    fake.sell, fake.buy = Mock(), Mock()
    Phase2Rotation.next_open(fake)
    fake.sell.assert_called_once_with(data=feeds[0], size=50)
    assert fake.available_at_open == 990.
    fake.sell.return_value.addinfo.assert_called_once_with(reason="rebalance")
    fake.broker.setcash.assert_not_called()


def test_catalog_contains_exactly_two_bridges_and_six_continuous_accounts():
    rows = run.catalog("a" * 64, "b" * 64)
    assert len(rows) == len({row.key for row in rows}) == 8
    assert sum(row.terminal for row in rows) == 2
    assert {r.role for r in rows if r.stage == "science"} == {"ANNUAL", "SPY_BH", "QQQ_BH"}
    assert all(r.start == date(2022, 1, 3) for r in rows)
    assert all(r.through == date(2026, 11, 10) for r in rows if not r.terminal)


def test_native_budget_counts_failed_attempts_and_output_switch_cannot_retry(tmp_path):
    for n in range(12):
        run.journal_event(tmp_path, "started", str(n), str(n), tmp_path / str(n))
        run.journal_event(tmp_path, "failed", str(n), str(n), tmp_path / str(n))
    with pytest.raises(ValueError, match="hard12"):
        run.journal_event(tmp_path, "started", "extra", "13", tmp_path / "extra")
    other = tmp_path / "other"
    run.journal_event(other, "started", "same", "1", other / "one")
    with pytest.raises(ValueError, match="already attempted"):
        run.journal_event(other, "started", "same", "2", other / "two")


def test_saved_success_is_revalidated_without_a_second_native_run(monkeypatch, tmp_path):
    prepared = market()
    co = contract(prepared)
    folder = tmp_path / co.key
    folder.mkdir()
    for name in ("ARTIFACT.json", "NATIVE_COMPLETE.json"):
        (folder / name).write_text("{}")
    receipt = Receipt(account=co.key, hashes={}, max_differences={})
    monkeypatch.setattr(run, "validate_saved", Mock(return_value=receipt))
    monkeypatch.setattr(native, "simulate", Mock(side_effect=AssertionError("no repeat")))
    assert run.execute(co, prepared, tmp_path, tmp_path) == receipt


def test_science_requires_both_bridges_before_spending_native_budget(monkeypatch, tmp_path):
    prepared = market()
    co = contract(prepared)
    monkeypatch.setattr(run, "binding", Mock(return_value=Binding(
        contract=co, source_sha256="a" * 64)))
    monkeypatch.setattr(run, "require_bridges", Mock(side_effect=ValueError("both bridges")))
    with pytest.raises(ValueError, match="both bridges"):
        run.execute(co, prepared, tmp_path, tmp_path)
    assert not run.journal_path(tmp_path).exists()


def test_future_union_feed_can_only_add_zero_historical_holdings():
    expected = pd.DataFrame({"day": ["2025-12-30", "2025-12-31"],
                             "symbol": ["SPY", "SPY"], "size": [10, 0], "close": [100., 100.]})
    actual = pd.DataFrame({"day": ["2025-12-30", "2025-12-30", "2025-12-31"],
                           "symbol": ["SPY", "NEW", "SPY"], "size": [10, 0, 10],
                           "close": [100., 80., 100.]})
    assert run.prefix_book(expected, actual, "holdings") == 0
    actual.loc[1, "size"] = 1
    with pytest.raises(ValueError, match="nonzero historical"):
        run.prefix_book(expected, actual, "holdings")


def test_historical_prefix_detects_cash_changes_but_excludes_old_terminal_day():
    old = pd.DataFrame({"day": ["2025-12-30", "2025-12-31"], "equity": [60000., 59900.]})
    new = pd.DataFrame({"day": ["2025-12-30", "2025-12-31", "2026-01-02"],
                        "equity": [60000., 60100., 60200.]})
    assert run.prefix_book(old, new, "daily") == 0
    new.loc[0, "equity"] += .01
    with pytest.raises(ValueError, match="numeric mismatch"):
        run.prefix_book(old, new, "daily")


@pytest.mark.parametrize("symbol", ["SPY", "QQQ"])
def test_continuous_buy_hold_matches_empty_historical_trade_prefix(
    symbol: str, tmp_path: Path
) -> None:
    from c10standard.io import ROOT
    from rotbt.models import ClosedTrade

    reference = ROOT / "standard/reference" / f"{symbol}_BH_5bp_historical2022_2025"
    historical = pd.read_csv(reference / "trades.csv.gz")
    assert historical["closed"].tolist() == ["2025-12-31"]
    continuous = tmp_path / "trades.csv.gz"
    pd.DataFrame(columns=list(ClosedTrade.model_fields)).to_csv(continuous, index=False)
    trades = pd.read_csv(continuous)

    assert run.prefix_book(historical, trades, "trades") == 0
    with pytest.raises(ValueError, match="shape or columns mismatch"):
        run.prefix_book(historical, trades.drop(columns="symbol"), "trades")


def open_account():
    """Representative one-buy book kept open across every year boundary."""
    from c10standard.models import NativeReturn, OpenTrade, TargetRecord
    from rotbt.broker import commission
    from rotbt.models import DailyRecord, HoldingRecord, OrderRecord, Result, Spec
    from scripts.ytd_annual_models import Artifact

    prepared = market(role="SPY_BH")
    co = contract(prepared)
    calendar = xcals.get_calendar("XNYS")
    days = calendar.sessions_in_range(prepared.first, co.through)
    for data in prepared.datasets:
        data.frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.,
                                   "volume": 1e6, "turnover": 20e6, "tr": 100.}, index=days)
    prepared.features = signals.features_for(prepared.datasets)
    price, quantity = 100.05, 474
    fee = commission(quantity, price, FEES)
    cash = 50000. - quantity * price - fee
    equity = cash + quantity * 100.
    account_days = [t.date() for t in calendar.sessions_in_range(co.start, co.through)]
    daily = [DailyRecord(day=d, equity=equity, cash=cash, available=cash, locked=0.,
                          positions=1, gross_exposure=quantity * 100. / equity)
             for d in account_days]
    holdings = [HoldingRecord(day=d, symbol=s, size=quantity if s == "SPY" else 0,
                              close=100.) for d in account_days for s in ("SHY", "SPY")]
    initial = date(2021, 12, 31)
    returns = [NativeReturn(day=d, value=equity / 50000. - 1 if i == 0 else 0.)
               for i, d in enumerate(account_days)]
    artifact = Artifact(
        contract=co,
        account=Result(spec=Spec(name="fixture", layer="hold", lookback=252, top=3),
                       orders=[OrderRecord(day=co.start, symbol="SPY", size=quantity,
                                           price=price, fee=fee, reason="initial")],
                       daily=daily, holdings=holdings, distributions=[], trades=[],
                       native_total_return=equity / 50000. - 1),
        targets=[TargetRecord(day=initial, q={"SPY": 1.}, reason="initial")],
        decision_times=[signals.decision_timing(co, prepared, initial)],
        native_returns=returns,
        open_trades=[OpenTrade(symbol="SPY", size=quantity, price=price, pnl=0., commission=fee)])
    return artifact, prepared


def test_science_retains_open_native_trade_and_reconciles_entire_account():
    artifact, prepared = open_account()
    native.audit(artifact, prepared)
    assert artifact.open_trades[0].size == 474
    assert artifact.account.daily[-1].cash == artifact.account.daily[0].cash


@pytest.mark.parametrize("damage", ["missing_trade", "trade_size", "cash_reset", "terminal"])
def test_science_audit_rejects_open_trade_damage_cash_reset_or_artificial_exit(damage):
    artifact, prepared = open_account()
    if damage == "missing_trade":
        artifact.open_trades.clear()
    elif damage == "trade_size":
        artifact.open_trades[0].size += 1
    elif damage == "cash_reset":
        next(d for d in artifact.account.daily if d.day == date(2026, 1, 2)).cash = 50000.
    else:
        artifact.account.orders[0].reason = "terminal"
    with pytest.raises(ValueError):
        native.audit(artifact, prepared)
