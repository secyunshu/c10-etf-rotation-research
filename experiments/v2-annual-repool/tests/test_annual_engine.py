"""Annual pool causality and one-account lifecycle; never start a native broker."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import backtrader as bt
import exchange_calendars as xcals
import pandas as pd
import pytest
from pydantic import ValidationError

from c10standard.models import GROUPS, RISK, Receipt
from rotbt.models import CashEvent, Dataset, Distribution, FeeSchedule, SplitEvent
from rotbt.phase2.strategy import Phase2Rotation
from scripts import annual_native as native
from scripts import annual_run as run
from scripts import annual_signals as signals
from scripts.annual_models import (
    Binding,
    Contract,
    GroupMap,
    Pool,
    Prepared,
    Schedule,
    pool_sha256,
    schedule_sha256,
)

DAY = date(2022, 12, 30)
FEES = FeeSchedule(commission_per_share=.0049, commission_min=.99,
                   platform_per_share=.005, platform_min=1., settlement_per_share=.003,
                   sec_rate=.0000206, sec_min=.01, nms_per_share=.000003)


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
    monkeypatch.setattr(bt, "Cerebro", Mock(side_effect=AssertionError("native forbidden")))


def schedule(mode="ANNUAL"):
    entries = []
    calendar = xcals.get_calendar("XNYS")
    for year in range(2022, 2026):
        groups = GroupMap(groups=GROUPS if mode == "FIXED" else {
            "us_broad_style": ("QQQ", "SPY"), "technology_semiconductors": ("SOXX",),
            "precious_metals": ("GLD",) if year == 2022 else ("GLD", "SLV")})
        names = RISK if mode == "FIXED" else tuple(sorted(
            s for members in groups.groups.values() for s in members))
        selection = date(year - 1, 12, 31)
        price = calendar.date_to_session(pd.Timestamp(selection), direction="previous").date()
        execute = calendar.next_session(pd.Timestamp(price)).date()
        source = date(2021, 12, 31) if mode == "FIXED" else selection
        entries.append(Pool(year=year, risk_names=names, groups=groups, price_asof=price,
                            selection_asof=selection, execute_on=execute, source_asof=source,
                            sha256=pool_sha256(year, names, groups, price, selection, execute,
                                               source)))
    rows = tuple(entries)
    return Schedule(mode=mode, entries=rows, sha256=schedule_sha256(mode, rows))


def market(scores=(.5, .4, .3, .2, .25, .1), observations=300, day=DAY):
    days = pd.bdate_range(end=day, periods=observations)
    datasets = []
    for symbol, score in zip(("QQQ", "SPY", "SOXX", "GLD", "SHY", "SLV"), scores,
                             strict=True):
        frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.,
                              "volume": 1e6, "turnover": 20e6,
                              "tr": [100.] * (observations - 1) + [100 * (1 + score)]},
                             index=days)
        datasets.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    return Prepared(schedule=schedule(), datasets=datasets,
                    features=signals.features_for(datasets), fees=FEES)


def contract(prepared, cost=5):
    return Contract(mode="ANNUAL", cost_bps=cost, stage="science",
                    schedule_sha256=prepared.schedule.sha256)


@pytest.mark.parametrize("price,year,execute", [
    (date(2021, 12, 31), 2022, date(2022, 1, 3)),
    (date(2022, 12, 30), 2023, date(2023, 1, 3)),
    (date(2023, 12, 29), 2024, date(2024, 1, 2)),
    (date(2024, 12, 31), 2025, date(2025, 1, 2)),
])
def test_year_end_uses_next_execution_year_and_distinct_metadata_time(price, year, execute):
    prepared = market()
    timing = signals.decision_timing(contract(prepared), prepared, price)
    assert timing.pool_year == year
    assert timing.price_asof == price
    assert timing.selection_asof == date(year - 1, 12, 31)
    assert timing.execute_on == execute


def test_midyear_cannot_use_next_year_members_even_when_their_score_is_higher():
    prepared = market(scores=(.5, .4, .3, .2, .25, 9.), day=date(2022, 11, 30))
    co = contract(prepared)
    assert "SLV" not in signals.target(co, prepared, date(2022, 11, 30)).q
    assert "SLV" not in signals.oracle(co, prepared, date(2022, 11, 30)).q
    assert "SLV" in signals.feed_names(co, prepared)
    timing = signals.decision_timing(co, prepared, date(2022, 11, 30))
    assert timing.selection_asof == timing.price_asof


def test_fixed_schedule_preserves_original_members_and_2021_source_each_year():
    fixed = schedule("FIXED")
    assert all(p.risk_names == RISK and p.source_asof == date(2021, 12, 31)
               for p in fixed.entries)
    with pytest.raises(ValidationError, match="four|2022"):
        Schedule(mode="ANNUAL", entries=fixed.entries[:3], sha256="a" * 64)


def test_schedule_and_contract_reject_hash_or_mode_substitution():
    prepared = market()
    with pytest.raises(ValidationError, match="hash"):
        Schedule(mode="ANNUAL", entries=prepared.schedule.entries, sha256="a" * 64)
    co = contract(prepared).model_copy(update={"schedule_sha256": "b" * 64})
    with pytest.raises(ValueError, match="bind"):
        signals.feed_names(co, prepared)
    with pytest.raises(ValidationError):
        Contract(mode="ANNUAL", cost_bps=5, stage="bridge", schedule_sha256="a" * 64)


def test_weekend_information_cannot_claim_friday_selection_or_monday_execution():
    pool = schedule().entries[1]
    for field, wrong in (("selection_asof", date(2022, 12, 30)),
                         ("execute_on", date(2023, 1, 2))):
        with pytest.raises(ValidationError, match="timing"):
            Pool.model_validate(pool.model_copy(update={field: wrong}))


def test_changed_pool_members_are_rejected_even_under_same_schedule_hash():
    prepared = market()
    prepared.schedule.entries[1].groups.groups["precious_metals"] = ("GLD",)
    with pytest.raises(ValidationError, match="coverage"):
        signals.feed_names(contract(prepared), prepared)


def test_calendar_contract_rejects_future_market_data_and_2026_signal():
    prepared = market()
    with pytest.raises(ValueError, match="2025|window|scope"):
        signals.target(contract(prepared), prepared, date(2025, 12, 31))
    frames = [d.model_copy(update={"frame": d.frame.rename(
        index={d.frame.index[-1]: pd.Timestamp("2026-01-02")})}) for d in prepared.datasets]
    with pytest.raises(ValidationError, match="cutoff"):
        Prepared(schedule=prepared.schedule, datasets=frames,
                 features=signals.features_for(frames), fees=FEES)


@pytest.mark.parametrize("scores,expected", [
    ((.5, .4, .3, .2, .25, .1), {"QQQ": .45, "SOXX": .45, "SHY": .1}),
    ((.5, .4, .2, .1, .25, .1), {"QQQ": .45, "SHY": .55}),
    ((.2, .1, .2, .1, .25, .1), {"SHY": 1.}),
])
def test_original_group_slots_gate_and_defensive_weights_are_unchanged(scores, expected):
    prepared = market(scores)
    for function in (signals.target, signals.oracle):
        assert function(contract(prepared), prepared, DAY).q == pytest.approx(expected)
    assert native.settings_for(prepared, contract(prepared)).invested == .95


def test_253_prices_are_required_for_cash_benchmark():
    prepared = market(observations=252)
    for function in (signals.target, signals.oracle):
        with pytest.raises(ValueError, match="253|warmup"):
            function(contract(prepared), prepared, DAY)
    enough = market(observations=253)
    assert signals.target(contract(enough), enough, DAY).q["QQQ"] == .45


def test_raw_oracle_detects_tampered_features_and_prefix_removes_future_rows():
    prepared = market()
    signals.verify_targets(contract(prepared), prepared, (DAY,))
    prepared.features.frames["QQQ"].loc[pd.Timestamp(DAY), "mom252"] = -1
    with pytest.raises(ValueError, match="target"):
        signals.verify_targets(contract(prepared), prepared, (DAY,))


def test_future_price_spike_cannot_change_existing_target_or_prefix_oracle():
    prepared = market()
    expected = signals.target(contract(prepared), prepared, DAY)
    for data in prepared.datasets:
        data.frame.loc[pd.Timestamp("2023-01-03")] = data.frame.iloc[-1]
        data.frame.loc[pd.Timestamp("2023-01-03"), "tr"] = 1e9
    prepared.features = signals.features_for(prepared.datasets)
    assert signals.target(contract(prepared), prepared, DAY) == expected
    signals.verify_targets(contract(prepared), prepared, (DAY,))


def test_illiquid_or_subfive_fund_cannot_take_a_slot():
    prepared = market()
    for data in prepared.datasets:
        if data.symbol == "QQQ":
            data.frame.loc[data.frame.index[-1], "close"] = 4.99
        if data.symbol == "SOXX":
            data.frame.loc[data.frame.index[-1], "volume"] = 0
    prepared.features = signals.features_for(prepared.datasets)
    assert signals.target(contract(prepared), prepared, DAY).q == {"SPY": .45, "SHY": .55}
    assert signals.oracle(contract(prepared), prepared, DAY).q == {"SPY": .45, "SHY": .55}


def test_monthly_calendar_has_48_signals_and_no_annual_liquidations():
    prepared = market()
    days = native.signal_days(contract(prepared))
    assert len(days) == 48 and days[0] == date(2021, 12, 31)
    assert days[-1] == date(2025, 11, 28)
    assert sum(d.month == 12 for d in days) == 4


def test_native_configuration_keeps_union_feed_actions_and_one_account(monkeypatch, tmp_path):
    prepared = market()
    prepared.datasets[3].dividends = [CashEvent(
        symbol="GLD", ex_date=DAY, pay_date=date(2023, 1, 4), amount=1., source="fixture")]
    prepared.datasets[3].splits = [SplitEvent(
        symbol="GLD", day=date(2023, 1, 3), shares_after_per_before=2., source="fixture")]
    broker = Mock()
    cerebro = Mock(broker=broker)
    cerebro.run.side_effect = RuntimeError("capture only")
    monkeypatch.setattr(bt, "Cerebro", Mock(return_value=cerebro))
    co = contract(prepared)
    with pytest.raises(RuntimeError, match="capture only"):
        native.simulate(prepared, co, native.settings_for(prepared, co), tmp_path)
    broker.setcash.assert_called_once_with(50000)
    options = cerebro.addstrategy.call_args.kwargs
    assert options["choices"] == ()
    assert options["start"] == date(2022, 1, 3)
    assert options["end"] == date(2025, 12, 31)
    assert {call.kwargs["name"] for call in cerebro.adddata.call_args_list} == {
        d.symbol for d in prepared.datasets}
    assert options["events"] == tuple(e for d in prepared.datasets for e in d.dividends)
    assert options["splits"] == tuple(s for d in prepared.datasets for s in d.splits)
    assert cerebro.addanalyzer.call_args.args[0] is bt.analyzers.TimeReturn


def test_native_rejects_changed_account_settings_before_broker_creation(tmp_path):
    prepared = market()
    co = contract(prepared)
    settings = native.settings_for(prepared, co).model_copy(update={"invested": .9})
    with pytest.raises(ValueError, match="settings"):
        native.simulate(prepared, co, settings, tmp_path)


def test_removed_holding_target_is_zero_at_next_open_without_reset():
    fake = SimpleNamespace(p2=SimpleNamespace(delay_days=0),
                           _ordinal={DAY: 0}, _days=(DAY, date(2023, 1, 3)))
    Phase2Rotation._stage(fake, {"SHY": .95}, DAY, "rebalance")
    assert fake.targets.get("REMOVED", 0) == 0
    assert fake._execute_on == date(2023, 1, 3)
    assert fake.reason == "rebalance"
    feeds = [SimpleNamespace(_name=name, open=[100.], datetime=Mock())
             for name in ("REMOVED", "SHY")]
    for feed in feeds:
        feed.datetime.date.return_value = date(2023, 1, 3)
    fake.datas = feeds
    fake.pending = []
    fake._cash_events = Mock()
    fake.start_day, fake.end_day = date(2022, 1, 3), date(2025, 12, 31)
    fake.initialized, fake.choices, fake.locked = True, (), 10.
    fake.settings = native.settings_for(market(), contract(market()))
    fake.broker = Mock()
    fake.broker.getcash.return_value = 1000.
    fake.getposition = lambda feed: SimpleNamespace(size=50 if feed._name == "REMOVED" else 0)
    fake.exit_only, fake.p2.band = set(), 0
    fake.sell, fake.buy = Mock(), Mock()
    Phase2Rotation.next_open(fake)
    fake.sell.assert_called_once_with(data=feeds[0], size=50)
    fake.sell.return_value.addinfo.assert_called_once_with(reason="rebalance")
    assert fake.buy.call_args.kwargs["data"] is feeds[1]
    assert fake.available_at_open == 990.
    fake.broker.setcash.assert_not_called()


def test_removed_asset_receivable_remains_locked_until_payment_release():
    record = Distribution(symbol="REMOVED", ex_date=DAY, pay_date=date(2023, 1, 4),
                          release_date=date(2023, 1, 5), qty=10, per_share=1.,
                          gross_amount=10., net_amount=10.)
    fake = SimpleNamespace(last_entitlement=None, receivables=[record], distributions=[record],
                           start_day=date(2022, 1, 3), end_day=date(2025, 12, 31),
                           datas=[], _events_by_day={})
    Phase2Rotation._cash_events(fake, date(2023, 1, 3))
    assert fake.locked == 10 and fake.receivables == [record]
    Phase2Rotation._cash_events(fake, date(2023, 1, 5))
    assert fake.locked == 0 and fake.distributions == [record]


def test_catalog_has_only_two_bridges_and_two_annual_science_accounts():
    rows = run.catalog("a" * 64, "b" * 64)
    assert [row.mode for row in rows] == ["FIXED", "FIXED", "ANNUAL", "ANNUAL"]
    assert len({row.key for row in rows}) == 4
    assert {row.start for row in rows} == {date(2022, 1, 3)}


def test_journal_counts_failures_toward_hard8_and_blocks_output_switch(tmp_path):
    for n in range(8):
        run.journal_event(tmp_path, "started", f"account{n}", str(n), tmp_path / str(n))
        run.journal_event(tmp_path, "failed", f"account{n}", str(n), tmp_path / str(n))
    with pytest.raises(ValueError, match="hard8"):
        run.journal_event(tmp_path, "started", "new", "9", tmp_path / "new")
    root = tmp_path / "other"
    run.journal_event(root, "started", "same", "1", root / "one")
    with pytest.raises(ValueError, match="already attempted"):
        run.journal_event(root, "started", "same", "2", root / "two")


def test_successful_account_is_revalidated_without_simulation(monkeypatch, tmp_path):
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


def test_science_requires_both_bridges_before_any_native_attempt(monkeypatch, tmp_path):
    prepared = market()
    co = contract(prepared)
    monkeypatch.setattr(run, "binding", Mock(return_value=Binding(
        contract=co, source_sha256="a" * 64)))
    monkeypatch.setattr(run, "require_bridges", Mock(side_effect=ValueError("both bridges")))
    with pytest.raises(ValueError, match="both bridges"):
        run.execute(co, prepared, tmp_path, tmp_path)
    assert not run.journal_path(tmp_path).exists()


def test_saved_binding_mismatch_fails_before_loading_artifact(monkeypatch, tmp_path):
    prepared = market()
    co = contract(prepared)
    (tmp_path / "BINDING.json").write_text(Binding(
        contract=co, source_sha256="a" * 64).model_dump_json())
    monkeypatch.setattr(run, "binding", Mock(return_value=Binding(
        contract=co, source_sha256="b" * 64)))
    with pytest.raises(ValueError, match="binding"):
        run.validate_saved(co, prepared, tmp_path, tmp_path)
