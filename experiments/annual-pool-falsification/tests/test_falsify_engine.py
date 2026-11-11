"""Counterfactual policy identity and lifecycle, without any native broker run."""

from datetime import date
from pathlib import Path
from shutil import copytree
from unittest.mock import Mock

import backtrader as bt
import pandas as pd
import pytest
from pydantic import ValidationError

from c10standard.models import Receipt
from rotbt.models import Dataset, FeeSchedule
from scripts import annual_native as native
from scripts import annual_run, annual_signals
from scripts import falsify_run as run
from scripts.annual_models import Prepared, Schedule
from scripts.falsify_models import CaseBinding, CaseReceipt, build_cases, case_hash, catalog
from scripts.falsify_signals import case_scope

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "experiments/v2-annual-repool"
DAY = date(2022, 12, 30)
FEES = FeeSchedule(commission_per_share=.0049, commission_min=.99,
                   platform_per_share=.005, platform_min=1., settlement_per_share=.003,
                   sec_rate=.0000206, sec_min=.01, nms_per_share=.000003)


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
    monkeypatch.setattr(bt, "Cerebro", Mock(side_effect=AssertionError("native forbidden")))


@pytest.fixture
def policies():
    fixed = Schedule.model_validate_json((OLD / "FIXED.json").read_bytes())
    annual = Schedule.model_validate_json((OLD / "ANNUAL.json").read_bytes())
    return fixed, annual, build_cases(fixed, annual)


def account(policies, case_id="B", cost=5):
    fixed, annual, cases = policies
    return next(a for a in catalog(cases, fixed, annual)
                if a.case.id == case_id and a.cost_bps == cost)


def market(policies, observations=300):
    annual = policies[1]
    names = sorted({s for p in annual.entries for s in p.risk_names} | {"SHY"})
    days = pd.bdate_range(end=DAY, periods=observations)
    datasets = []
    for symbol in names:
        score = {"QQQ": .4, "SOXX": .5, "SMH": .9, "SHY": .2}.get(symbol, .1)
        frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.,
                              "volume": 1e6, "turnover": 20e6,
                              "tr": [100.] * (observations - 1) + [100 * (1 + score)]},
                             index=days)
        datasets.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    return Prepared(schedule=annual, datasets=datasets,
                    features=annual_signals.features_for(datasets), fees=FEES)


def test_policies_preserve_common_2022_and_only_registered_later_members(policies):
    fixed, annual, cases = policies
    a, b, c, d, e = cases.cases
    assert a.pools == fixed.entries and d.pools == annual.entries
    assert all(case.pools[0] == fixed.entries[0] for case in cases.cases)
    for i in range(1, 4):
        assert set(b.pools[i].risk_names) == set(fixed.entries[i].risk_names) - {"SOXX"} | {"SMH"}
        assert set(c.pools[i].risk_names) == set(annual.entries[i].risk_names) - {"SMH"} | {"SOXX"}
        assert e.pools[i].risk_names == annual.entries[1].risk_names
        assert e.pools[i].groups == annual.entries[1].groups
        for case in (b, c):
            group = case.pools[i].groups.groups["technology_semiconductors"]
            assert "VGT" in group and len(group) == 2
        for group in fixed.entries[i].groups.groups:
            if group != "technology_semiconductors":
                assert b.pools[i].groups.groups[group] == fixed.entries[i].groups.groups[group]
        for group in annual.entries[i].groups.groups:
            if group != "technology_semiconductors":
                assert c.pools[i].groups.groups[group] == annual.entries[i].groups.groups[group]
        assert b.pool_sources[i].asof["SMH"] == date(2022, 12, 31)
        assert b.pool_sources[i].asof["QQQ"] == date(2021, 12, 31)
        assert c.pool_sources[i].asof["SOXX"] == date(2021, 12, 31)
        assert c.pool_sources[i].asof["QQQ"] == date(2021 + i, 12, 31)
        assert e.pools[i].source_asof == date(2022, 12, 31)
        assert b.pools[i].source_asof == date(2022, 12, 31)
    assert b.pools[1].price_asof == date(2022, 12, 30)
    assert b.pools[1].selection_asof == date(2022, 12, 31)
    assert b.pools[1].execute_on == date(2023, 1, 3)


def test_catalog_has_two_bridges_and_six_science_accounts(policies):
    fixed, annual, cases = policies
    rows = catalog(cases, fixed, annual)
    assert [r.key for r in rows[:2]] == ["A_5bp_bridge", "D_20bp_bridge"]
    assert {r.key for r in rows[2:]} == {f"{s}_{c}bp_science" for s in "BCE" for c in (5, 20)}
    assert all(r.envelope.mode == "ANNUAL" for r in rows[1:])
    assert len({r.key for r in rows}) == 8


def test_relabelled_case_is_rejected_even_if_inner_envelope_is_identical(policies):
    b = account(policies)
    payload = b.model_dump(mode="json")
    payload["case"]["id"] = "C"
    with pytest.raises(ValidationError, match="hash"):
        type(b).model_validate(payload)


def test_rehashed_false_case_definition_cannot_enter_registered_catalog(policies):
    fixed, annual, cases = policies
    bad = cases.cases[1].model_copy(update={"pools": cases.cases[0].pools,
                                          "pool_sources": cases.cases[0].pool_sources})
    bad = bad.model_copy(update={"sha256": case_hash(
        bad.id, bad.pools, bad.pool_sources, bad.intervention_note)})
    wrong = type(cases)(cases=(cases.cases[0], bad, *cases.cases[2:]))
    with pytest.raises(ValueError, match="independently reconstructed"):
        catalog(wrong, fixed, annual)


def test_scope_selects_actual_case_in_production_raw_and_timing_then_restores(policies):
    prepared, b = market(policies), account(policies)
    original = annual_signals.active_pool, annual_run.binding
    with case_scope(b, "a" * 64):
        target = annual_signals.target(b.envelope, prepared, DAY)
        assert target.q == {"SMH": .45, "QQQ": .45, "SHY": pytest.approx(.1)}
        assert annual_signals.oracle(b.envelope, prepared, DAY).q == target.q
        annual_signals.verify_targets(b.envelope, prepared, (DAY,))
        timing = annual_signals.decision_timing(b.envelope, prepared, DAY)
        assert timing.pool_sha256 == b.case.pools[1].sha256
        assert timing.source_asof == date(2022, 12, 31)
        assert annual_run.binding(b.envelope, ROOT).source_sha256 == "a" * 64
        with pytest.raises(ValueError, match="scope"):
            with case_scope(b, "b" * 64):
                pass
    assert (annual_signals.active_pool, annual_run.binding) == original
    with pytest.raises(RuntimeError):
        with case_scope(b, "a" * 64):
            raise RuntimeError("forced failure")
    assert (annual_signals.active_pool, annual_run.binding) == original


def test_scope_applies_no_future_membership_and_rejects_2026_execution(policies):
    prepared, b = market(policies), account(policies)
    with case_scope(b, "a" * 64):
        pool = annual_signals.active_pool(b.envelope, prepared, date(2022, 11, 30))
        assert "SOXX" in pool.risk_names and "SMH" not in pool.risk_names
        with pytest.raises(ValueError, match="window"):
            annual_signals.active_pool(b.envelope, prepared, date(2025, 12, 31))


def test_original_cash_gate_age_liquidity_and_investment_fraction_remain(policies):
    prepared, b = market(policies), account(policies)
    next(d for d in prepared.datasets if d.symbol == "SMH").frame.loc[:, "turnover"] = 1.
    prepared.features = annual_signals.features_for(prepared.datasets)
    with case_scope(b, "a" * 64):
        assert annual_signals.target(b.envelope, prepared, DAY).q == {"QQQ": .45, "SHY": .55}
        assert annual_signals.oracle(b.envelope, prepared, DAY).q == {"QQQ": .45, "SHY": .55}
        with pytest.raises(ValueError, match="253"):
            annual_signals.target(b.envelope, market(policies, 252), DAY)
    settings = native.settings_for(prepared, b.envelope)
    assert settings.initial_cash == 50000 and settings.invested == .95 and settings.fees == FEES


def test_future_price_spike_cannot_change_target_and_prefix_oracle(policies):
    prepared, b = market(policies), account(policies)
    with case_scope(b, "a" * 64):
        before = annual_signals.target(b.envelope, prepared, DAY)
        for data in prepared.datasets:
            data.frame.loc[pd.Timestamp("2023-01-03")] = [100., 101., 99., 100., 1e6, 20e6, 1e10]
        prepared.features = annual_signals.features_for(prepared.datasets)
        assert annual_signals.target(b.envelope, prepared, DAY) == before
        annual_signals.verify_targets(b.envelope, prepared, (DAY,))


def test_journal_counts_failed_attempts_to_hard12_and_prevents_output_switch(tmp_path):
    for n in range(12):
        run.journal_event(tmp_path, "started", f"case{n}", str(n), tmp_path / str(n))
        run.journal_event(tmp_path, "failed", f"case{n}", str(n), tmp_path / str(n))
    with pytest.raises(ValueError, match="hard12"):
        run.journal_event(tmp_path, "started", "extra", "x", tmp_path / "extra")
    fresh = tmp_path / "fresh"
    run.journal_event(fresh, "started", "B_5bp_science", "one", fresh / "one")
    with pytest.raises(ValueError, match="already attempted"):
        run.journal_event(fresh, "started", "B_5bp_science", "two", fresh / "two")


def test_science_requires_both_bridges_before_native_attempt(policies, monkeypatch, tmp_path):
    b, prepared = account(policies), market(policies)
    monkeypatch.setattr(run, "binding", Mock(return_value=CaseBinding(
        account=b, source_sha256="a" * 64)))
    monkeypatch.setattr(run, "require_bridges", Mock(side_effect=ValueError("both bridges")))
    with pytest.raises(ValueError, match="both bridges"):
        run.execute(b, prepared, tmp_path / "out", tmp_path)
    assert not run.journal_path(tmp_path).exists()


def test_completed_revalidates_and_incomplete_failure_is_preserved(policies, monkeypatch, tmp_path):
    b, prepared = account(policies), market(policies)
    folder = tmp_path / b.key
    folder.mkdir()
    failure = folder / "PARTIAL_NATIVE.json"
    failure.write_text("preserved")
    with pytest.raises(ValueError, match="preserved"):
        run.execute(b, prepared, tmp_path, tmp_path)
    assert failure.read_text() == "preserved"
    for name in ("ARTIFACT.json", "NATIVE_COMPLETE.json"):
        (folder / name).write_text("{}")
    receipt = CaseReceipt(account=b.key, native=Receipt(
        account=b.envelope.key, hashes={}, max_differences={}))
    monkeypatch.setattr(run, "validate_saved", Mock(return_value=receipt))
    monkeypatch.setattr(native, "simulate", Mock(side_effect=AssertionError("must not rerun")))
    assert run.execute(b, prepared, tmp_path, tmp_path) == receipt


def test_outer_binding_mismatch_fails_before_native_artifact_read(policies, monkeypatch, tmp_path):
    b, c, prepared = account(policies), account(policies, "C"), market(policies)
    (tmp_path / "CASE_BINDING.json").write_text(CaseBinding(
        account=c, source_sha256="a" * 64).model_dump_json())
    monkeypatch.setattr(run, "binding", Mock(return_value=CaseBinding(
        account=b, source_sha256="a" * 64)))
    with pytest.raises(ValueError, match="case binding"):
        run.validate_saved(b, prepared, tmp_path, tmp_path)


def test_native_failure_preserves_partial_output_and_counts_once(policies, monkeypatch, tmp_path):
    a = account(policies, "A")
    all_data = market(policies)
    names = set(policies[0].entries[0].risk_names) | {"SHY"}
    datasets = [d for d in all_data.datasets if d.symbol in names]
    prepared = Prepared(schedule=policies[0], datasets=datasets,
                        features=annual_signals.features_for(datasets), fees=FEES)
    monkeypatch.setattr(run, "binding", Mock(return_value=CaseBinding(
        account=a, source_sha256="a" * 64)))
    monkeypatch.setattr(run, "verify_targets", Mock())

    def fail_native(prepared, contract, settings, folder):
        (folder / "PARTIAL_NATIVE.json").write_text("partial native state")
        raise ValueError("native fault")

    monkeypatch.setattr(native, "simulate", fail_native)
    original = annual_signals.active_pool, annual_run.binding
    with pytest.raises(ValueError, match="native fault"):
        run.execute(a, prepared, tmp_path / "out", tmp_path)
    folder = tmp_path / "out" / a.key
    assert (folder / "PARTIAL_NATIVE.json").read_text() == "partial native state"
    assert not (folder / "NATIVE_COMPLETE.json").exists()
    assert (annual_signals.active_pool, annual_run.binding) == original
    records = [run.Attempt.model_validate_json(row)
               for row in run.journal_path(tmp_path).read_text().splitlines()]
    assert [row.event for row in records] == ["started", "failed"]
    with pytest.raises(ValueError, match="preserved"):
        run.execute(a, prepared, tmp_path / "out", tmp_path)


@pytest.mark.parametrize("case_id,cost", [("A", 5), ("D", 20)])
def test_bridge_checks_saved_native_nav_against_same_cost_reference(
        policies, tmp_path, case_id, cost):
    a = account(policies, case_id, cost)
    folder = tmp_path / a.key
    copytree(OLD / "reference" / a.envelope.key, folder)
    receipt = Receipt(account=a.envelope.key, hashes={}, max_differences={})
    assert max(run.bridge_receipt(a, receipt, folder, ROOT).bridge_differences.values()) == 0
    daily = pd.read_csv(folder / "daily.csv.gz", float_precision="round_trip")
    daily.loc[0, "equity"] += 1.
    daily.to_csv(folder / "daily.csv.gz", index=False)
    with pytest.raises(ValueError, match="numeric mismatch"):
        run.bridge_receipt(a, receipt, folder, ROOT)
