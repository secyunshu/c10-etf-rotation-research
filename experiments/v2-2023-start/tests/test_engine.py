"""Economic and lifecycle contracts; tests never start a native broker."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import backtrader as bt
import pandas as pd
import pytest
from pydantic import ValidationError

from c10standard.io import load_strategy
from c10standard.models import Receipt, TargetRecord
from rotbt.models import Dataset, OrderRecord, Result, Spec
from scripts import v2_2023_native as native
from scripts import v2_2023_run as run
from scripts import v2_2023_signals as signals
from scripts.v2_2023_models import (
    Artifact,
    Binding,
    Contract,
    GroupMap,
    Pool,
    Prepared,
    pool_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
DAY = date(2022, 12, 30)


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bt, "Cerebro", Mock(side_effect=AssertionError("native forbidden")))


def market(scores=(.5, .4, .3, .2, .25), observations=300) -> Prepared:
    names = ("GLD", "QQQ", "SOXX", "SPY")
    groups = GroupMap(groups={"us_broad_style": ("QQQ", "SPY"),
                             "technology_semiconductors": ("SOXX",),
                             "precious_metals": ("GLD",)})
    pool = Pool(role="NEW_POOL", risk_names=names, groups=groups,
                sha256=pool_sha256("NEW_POOL", names, groups))
    days = pd.bdate_range(end=DAY, periods=observations)
    datasets = []
    for symbol, score in zip(("QQQ", "SPY", "SOXX", "GLD", "SHY"), scores, strict=True):
        frame = pd.DataFrame(
            {"open": 100., "high": 101., "low": 99., "close": 100.,
             "volume": 1e6, "turnover": 20e6,
             "tr": [100.] * (observations - 1) + [100 * (1 + score)]}, index=days,
        )
        datasets.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    return Prepared(pool=pool, datasets=datasets, features=signals.features_for(datasets),
                    fees=load_strategy(ROOT).fees)


def contract(prepared: Prepared, cost=5) -> Contract:
    return Contract(role="NEW_POOL", cost_bps=cost, stage="science",
                    pool_sha256=prepared.pool.sha256)


def test_catalog_has_two_bridge_and_eight_fresh_accounts_only() -> None:
    rows = run.catalog("a" * 64, "b" * 64)
    assert len(rows) == len({row.key for row in rows}) == 10
    assert sum(row.stage == "bridge" for row in rows) == 2
    assert {row.start for row in rows if row.stage == "science"} == {date(2023, 1, 3)}
    assert all(row.through == date(2025, 12, 31) for row in rows)
    with pytest.raises(ValidationError):
        Contract(role="NEW_POOL", stage="bridge", cost_bps=5, pool_sha256="a" * 64)
    with pytest.raises(ValidationError):
        Contract(role="NEW_POOL", stage="science", cost_bps=10, pool_sha256="a" * 64)


@pytest.mark.parametrize("scores,expected", [
    ((.5, .4, .3, .2, .25), {"QQQ": .45, "SOXX": .45, "SHY": .1}),
    ((.5, .4, .2, .1, .25), {"QQQ": .45, "SHY": .55}),
    ((.2, .1, .2, .1, .25), {"SHY": 1.}),
    ((.5, .5, .3, .2, .25), {"QQQ": .45, "SOXX": .45, "SHY": .1}),
])
def test_group_cap_defensive_gate_and_fixed_slots_preserve_original_v2(scores, expected) -> None:
    prepared = market(scores)
    co = contract(prepared)
    assert signals.target(co, prepared, DAY).q == pytest.approx(expected)
    assert signals.oracle(co, prepared, DAY).q == pytest.approx(expected)
    assert native.settings_for(prepared, co).invested == .95
    assert sum(expected.values()) == pytest.approx(1)


def test_252_prices_cannot_form_252_day_return_and_253_prices_can() -> None:
    short = market(observations=252)
    for function in (signals.target, signals.oracle):
        with pytest.raises(ValueError, match="warmup|253"):
            function(contract(short), short, DAY)
    enough = market(observations=253)
    assert signals.target(contract(enough), enough, DAY).q["QQQ"] == .45


def test_oracle_rebuilds_raw_returns_without_trusting_feature_scores() -> None:
    prepared = market()
    co = contract(prepared)
    expected = signals.oracle(co, prepared, DAY).q
    for frame in prepared.features.frames.values():
        frame["mom252"] = -999.
    assert signals.oracle(co, prepared, DAY).q == expected
    assert signals.target(co, prepared, DAY).q != expected


def test_pool_hash_and_missing_groups_cannot_silently_change_research_universe() -> None:
    prepared = market()
    with pytest.raises(ValidationError, match="hash"):
        Pool(role="NEW_POOL", risk_names=prepared.pool.risk_names,
             groups=prepared.pool.groups, sha256="0" * 64)
    groups = GroupMap(groups={"us_broad_style": ("QQQ",)})
    with pytest.raises(ValidationError, match="group"):
        Pool(role="NEW_POOL", risk_names=prepared.pool.risk_names, groups=groups,
             sha256=pool_sha256("NEW_POOL", prepared.pool.risk_names, groups))
    wrong = Contract(role="NEW_POOL", stage="science", cost_bps=5, pool_sha256="0" * 64)
    with pytest.raises(ValueError, match="pool"):
        signals.target(wrong, prepared, DAY)


def test_mutated_nested_pool_groups_are_rechecked_before_signal() -> None:
    prepared = market()
    prepared.pool.groups.groups["us_broad_style"] = ("QQQ",)
    with pytest.raises(ValueError, match="group|pool"):
        signals.target(contract(prepared), prepared, DAY)


def test_future_market_rows_cannot_enter_finite_prepared_contract() -> None:
    prepared = market()
    prepared.datasets[0].frame.loc[pd.Timestamp("2026-01-02")] = prepared.datasets[0].frame.iloc[-1]
    with pytest.raises(ValidationError, match="cutoff|scope"):
        Prepared(pool=prepared.pool, datasets=prepared.datasets,
                 features=signals.features_for(prepared.datasets), fees=prepared.fees)


@pytest.mark.parametrize("field,value", [("close", 4.), ("turnover", 9e6), ("volume", 0.)])
def test_unqualified_leader_is_removed_before_ranking_without_extra_risk(field, value) -> None:
    prepared = market()
    qqq = next(d for d in prepared.datasets if d.symbol == "QQQ")
    qqq.frame.loc[qqq.frame.index[-20:], field] = value
    prepared.features = signals.features_for(prepared.datasets)
    co = contract(prepared)
    for function in (signals.target, signals.oracle):
        assert function(co, prepared, DAY).q == pytest.approx({"SPY": .45, "SOXX": .45, "SHY": .1})


def test_real_signal_calendar_includes_initial_and_excludes_terminal_close() -> None:
    prepared = market()
    co = contract(prepared)
    assert native.signal_days(co)[0] == DAY
    assert len(native.signal_days(co)) == 36
    assert native.signal_days(co)[-1] == date(2025, 11, 28)
    assert native.settings_for(prepared, co).effective_research_period.first_year == 2023


def test_future_rows_do_not_change_prefix_target() -> None:
    prepared = market()
    co = contract(prepared)
    expected = signals.target(co, prepared, DAY).q
    for dataset in prepared.datasets:
        dataset.frame.loc[pd.Timestamp("2023-01-03")] = dataset.frame.iloc[-1] * 30
    prepared.features = signals.features_for(prepared.datasets)
    assert signals.target(co, prepared, DAY).q == expected
    signals.verify_targets(co, prepared, (DAY,))


def test_seventeen_native_starts_cannot_evade_the_global_budget(tmp_path: Path) -> None:
    for n in range(16):
        run.journal_event(tmp_path, "started", f"probe-{n}", str(n), tmp_path / str(n))
        run.journal_event(tmp_path, "failed", f"probe-{n}", str(n), tmp_path / str(n))
    with pytest.raises(ValueError, match="16"):
        run.journal_event(tmp_path, "started", "probe-17", "17", tmp_path / "17")


def test_same_account_cannot_be_retried_by_switching_output_folder(tmp_path: Path) -> None:
    run.journal_event(tmp_path, "started", "same", "1", tmp_path / "first")
    run.journal_event(tmp_path, "failed", "same", "1", tmp_path / "first")
    with pytest.raises(ValueError, match="already|retry"):
        run.journal_event(tmp_path, "started", "same", "2", tmp_path / "elsewhere")


def test_failed_artifact_is_preserved_without_native_retry(tmp_path: Path) -> None:
    prepared = market()
    co = contract(prepared)
    folder = tmp_path / co.key
    folder.mkdir()
    (folder / "PARTIAL_NATIVE.json").write_text("preserved")
    with pytest.raises(ValueError, match="preserved"):
        run.execute(co, prepared, tmp_path, tmp_path)
    assert (folder / "PARTIAL_NATIVE.json").read_text() == "preserved"


def test_completed_account_is_revalidated_without_another_native_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = market()
    co = contract(prepared)
    folder = tmp_path / co.key
    folder.mkdir()
    (folder / "ARTIFACT.json").write_text("saved")
    (folder / "NATIVE_COMPLETE.json").write_text("completed")
    receipt = Receipt(account=co.key, hashes={}, max_differences={})
    monkeypatch.setattr(run, "validate_saved", Mock(return_value=receipt))
    assert run.execute(co, prepared, tmp_path, tmp_path) == receipt
    assert not run.journal_path(tmp_path).exists()


def test_science_without_two_bridges_cannot_spend_a_native_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = market()
    co = contract(prepared)
    monkeypatch.setattr(run, "binding", lambda *_: Binding(contract=co, source_sha256="a" * 64))
    monkeypatch.setattr(run, "require_bridges",
                        Mock(side_effect=ValueError("both bridges must PASS")))
    with pytest.raises(ValueError, match="bridges"):
        run.execute(co, prepared, tmp_path, tmp_path)
    assert not run.journal_path(tmp_path).exists()
    assert not (tmp_path / co.key).exists()


def test_signal_close_cannot_be_mislabeled_as_an_execution_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = market()
    co = contract(prepared)
    account = Result(spec=Spec(name="fixture", layer="hold", lookback=252, top=2),
                     orders=[OrderRecord(symbol="QQQ", day=DAY, size=100, price=100.,
                                         fee=2., reason="initial")],
                     daily=[], holdings=[], distributions=[], trades=[], native_total_return=0.)
    artifact = Artifact(contract=co, account=account, native_returns=[], open_trades=[],
                        targets=[TargetRecord(day=day, q={"SHY": 1.},
                                              reason="initial" if i == 0 else "rebalance")
                                 for i, day in enumerate(native.signal_days(co))])
    monkeypatch.setattr(native, "reconcile", Mock())
    with pytest.raises(ValueError, match="next-open"):
        native.audit(artifact, prepared)
