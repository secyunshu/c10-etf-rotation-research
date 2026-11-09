"""Financial/data contracts; these tests never start a native account."""

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from c10share.io import ROOT, load_accounts, load_market, validate_dataset, verify_hashes
from c10share.models import Prepared, TargetRecord
from c10share.run import journal_event
from c10share.targets import target
from c10share.verify import compare_frames, verify_reference


@pytest.fixture(scope="module")
def prepared() -> Prepared:
    return load_market()


def test_frozen_market_has_exact_calendar_and_universe(prepared: Prepared) -> None:
    assert len(prepared.datasets) == 38
    assert len(prepared.manifest.risk_names) == 36
    assert all(data.frame.index[-1] == pd.Timestamp("2025-12-31") for data in prepared.datasets)
    assert all(data.frame.index[0] == pd.Timestamp("2020-12-31") for data in prepared.datasets)


def test_tampered_market_file_is_rejected_before_use(tmp_path: Path) -> None:
    source = ROOT / "data" / "market"
    shutil.copy(source / "MANIFEST.json", tmp_path)
    shutil.copy(source / "SPY.csv.gz", tmp_path)
    with (tmp_path / "SPY.csv.gz").open("ab") as stream:
        stream.write(b"altered")
    from c10share.models import MarketManifest

    manifest = MarketManifest.model_validate_json((tmp_path / "MANIFEST.json").read_text())
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hashes(tmp_path, {"SPY.csv.gz": manifest.output_hashes["SPY.csv.gz"]})


def test_physical_future_prices_are_rejected_instead_of_clipped(prepared: Prepared) -> None:
    data = prepared.datasets[0].model_copy(deep=True)
    extra = data.frame.iloc[[-1]].copy()
    extra.index = pd.DatetimeIndex(["2026-01-02"])
    data.frame = pd.concat([data.frame, extra])
    with pytest.raises(ValueError, match="cutoff"):
        validate_dataset(data)


def test_stored_total_return_must_reconcile_with_cash_and_splits(prepared: Prepared) -> None:
    data = prepared.datasets[0].model_copy(deep=True)
    data.frame.loc[pd.Timestamp("2024-10-11"), "tr"] *= 1.01
    with pytest.raises(ValueError, match="total return recurrence"):
        validate_dataset(data)


def test_future_features_cannot_change_prior_c10_decision(prepared: Prepared) -> None:
    day = date(2023, 6, 30)
    before = target(
        "C10", prepared.features, day, prepared.manifest.risk_names, prepared.manifest.groups
    )
    changed = prepared.features.model_copy(deep=True)
    for frame in changed.frames.values():
        frame.loc[frame.index > pd.Timestamp(day), :] = 1e12
    assert (
        target("C10", changed, day, prepared.manifest.risk_names, prepared.manifest.groups)
        == before
    )


def test_ineligible_high_score_asset_cannot_receive_risk_capital(prepared: Prepared) -> None:
    day = date(2023, 6, 30)
    changed = prepared.features.model_copy(deep=True)
    frame = changed.frames["SMH"]
    frame.loc[pd.Timestamp(day), "mom252"] = 100.0
    frame.loc[pd.Timestamp(day), "turnover20"] = 1.0
    weights = target("C10", changed, day, prepared.manifest.risk_names, prepared.manifest.groups)
    assert "SMH" not in weights


def test_reference_ledger_and_metrics_reconcile_without_native(prepared: Prepared) -> None:
    account = next(account for account in load_accounts() if account.name == "C10_5bp")
    evidence = verify_reference(account, prepared)
    assert evidence.status == "PASS"
    assert evidence.max_metric_difference <= 1e-10


def test_comparison_rejects_extra_rows_and_economic_drift() -> None:
    reference = pd.DataFrame({"day": ["2022-01-03"], "equity": [50000.0]})
    assert compare_frames(reference, reference.copy(), "daily") == 0.0
    with pytest.raises(ValueError, match="shape"):
        compare_frames(reference, pd.concat([reference, reference]), "daily")
    changed = reference.copy()
    changed.loc[0, "equity"] += 0.01
    with pytest.raises(ValueError, match="numeric mismatch"):
        compare_frames(reference, changed, "daily")


def test_liquidated_benchmark_keeps_earned_unpaid_dividend_in_nav(prepared: Prepared) -> None:
    account = next(account for account in load_accounts() if account.name == "SPY_BH_5bp")
    assert verify_reference(account, prepared).status == "PASS"


def test_failed_native_attempts_still_consume_the_fixed_budget(tmp_path: Path) -> None:
    journal = tmp_path / "attempts.jsonl"
    for i in range(48):
        journal_event(journal, "started", "C10_5bp", str(i), tmp_path)
        journal_event(journal, "failed", "C10_5bp", str(i), tmp_path, "synthetic failure")
    with pytest.raises(ValueError, match="budget exhausted"):
        journal_event(journal, "started", "C10_5bp", "49", tmp_path)
    assert len(journal.read_text().splitlines()) == 96


def test_invalid_journal_status_cannot_silently_evade_budget(tmp_path: Path) -> None:
    with pytest.raises((ValueError, TypeError)):
        journal_event(tmp_path / "attempts.jsonl", "skipped", "C10_5bp", "1", tmp_path)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_target_weight_cannot_pass_reference_validation(weight: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        TargetRecord(day=date(2021, 12, 31), q={"XLE": weight, "SHY": 0.1}, reason="initial")
