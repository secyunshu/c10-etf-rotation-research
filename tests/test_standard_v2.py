"""V2 contract tests are pure; no Backtrader account may run here."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from c10standard.models import GROUPS, RISK, Contract, Prepared, Strategy
from c10standard.run import catalog, execute, journal_event
from c10standard.targets import feed_names, oracle, target
from rotbt.models import Dataset
from rotbt.phase2.models import FeatureStore


def strategy() -> Strategy:
    return Strategy(
        version=2,
        id="C10_HISTORICAL_GROUP_CAP_V2",
        default_rule="C11",
        group_cap=True,
        risk_names=RISK,
        groups=GROUPS,
        fees={
            "commission_per_share": 0.0049,
            "commission_min": 0.99,
            "platform_per_share": 0.005,
            "platform_min": 1.0,
            "settlement_per_share": 0.003,
            "sec_rate": 0.0000206,
            "sec_min": 0.01,
            "nms_per_share": 0.000003,
        },
    )


def prepared() -> Prepared:
    days = pd.bdate_range(end="2025-12-31", periods=300)
    datasets = []
    for symbol, score in (("QQQ", 0.5), ("VTI", 0.4), ("SOXX", 0.3), ("VEA", 0.2), ("SHY", 0.25)):
        tr = [100.0] * 300
        tr[-1] = 100 * (1 + score)
        frame = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000.0,
                "turnover": 20_000_000.0,
                "tr": tr,
            },
            index=days,
        )
        datasets.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    features = FeatureStore(
        frames={
            d.symbol: d.frame.assign(
                mom252=d.frame.tr.pct_change(252),
                raw_close=d.frame.close,
                turnover20=d.frame.turnover.rolling(20).mean(),
                positive20=d.frame.volume.gt(0).rolling(20).sum(),
            )
            for d in datasets
        }
    )
    return Prepared(
        datasets=datasets,
        features=features,
        strategy=strategy(),
        first=days[0].date(),
        last=days[-1].date(),
        risk_names=("QQQ", "VTI", "SOXX", "VEA"),
    )


def test_fixed_matrix_preserves_all_accounts_and_top_two_is_default() -> None:
    accounts = catalog()
    assert len(accounts) == 36 and len({a.key for a in accounts}) == 36
    assert Contract().rule == "C11" and Contract().cost_bps == 5
    assert Contract().cohort == "historical2022_2025"
    assert all(
        a.through < date.max and (a.terminal == (a.cohort == "historical2022_2025"))
        for a in accounts
    )
    assert strategy().group_cap is True


def test_group_slots_and_cash_gate_preserve_the_fixed_risk_budget() -> None:
    data = prepared()
    co = Contract(rule="C00", cost_bps=5, cohort="reset2026")
    result = target(co, data, date(2025, 12, 31))
    assert result.q == pytest.approx({"QQQ": 0.3, "SOXX": 0.3, "SHY": 0.4})
    assert sum(result.q.values()) == pytest.approx(1.0)
    # The native strategy alone applies95percent; these are predeployment targets.
    assert result.q["QQQ"] == pytest.approx(0.3)
    for frame in data.features.frames.values():
        frame.loc[:, "mom252"] = -999.0
    assert oracle(co, data, date(2025, 12, 31)).q == result.q


def test_actual_pool_feeds_and_benchmark_feeds_are_separate() -> None:
    assert len(feed_names(Contract(), RISK)) == 35
    assert feed_names(Contract(rule="SPY_BH"), RISK) == ("SHY", "SPY")
    assert feed_names(Contract(rule="SPY60_IEF40"), RISK) == ("IEF", "SHY", "SPY")


def test_shared_journal_counts_failed_attempts_and_rejects_the_ninety_seventh(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "shared.jsonl"
    for n in range(96):
        journal_event(journal, "started", "C10", str(n), tmp_path / str(n))
        journal_event(journal, "failed", "C10", str(n), tmp_path / str(n), "fixture")
    with pytest.raises(ValueError, match="96"):
        journal_event(journal, "started", "C10", "overflow", tmp_path / "overflow")
    assert (
        sum(json.loads(line)["event"] == "started" for line in journal.read_text().splitlines())
        == 96
    )


def test_failed_native_without_completed_artifact_cannot_be_silently_overwritten(
    tmp_path: Path,
) -> None:
    folder = tmp_path / Contract().key
    folder.mkdir()
    (folder / "PARTIAL_NATIVE.json").write_text("preserved")
    with pytest.raises(ValueError, match="preserved"):
        execute(Contract(), prepared(), tmp_path, tmp_path / "journal.jsonl", tmp_path)
    assert (folder / "PARTIAL_NATIVE.json").read_text() == "preserved"


def test_v2_market_rejects_future_physical_data_and_changed_total_return() -> None:
    import exchange_calendars as xcals

    from c10standard.io import validate_dataset

    days = (
        xcals.get_calendar("XNYS").sessions_in_range("2020-12-31", "2025-12-31").tz_localize(None)
    )
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000.0,
            "turnover": 100_000_000.0,
            "tr": 100.0,
        },
        index=days,
    )
    data = Dataset(symbol="SPY", frame=frame, dividends=[])
    validate_dataset(data, date(2025, 12, 31))
    data.frame.loc[pd.Timestamp("2026-01-02")] = data.frame.iloc[-1]
    with pytest.raises(ValueError, match="cutoff|calendar"):
        validate_dataset(data, date(2025, 12, 31))
    data.frame = frame.loc[:"2025-12-31"].copy()
    data.frame.loc[pd.Timestamp("2024-03-20"), "tr"] *= 1.01
    with pytest.raises(ValueError, match="total return"):
        validate_dataset(data, date(2025, 12, 31))


def test_legacy_journal_cannot_be_adopted_as_the_new_v2_budget(tmp_path: Path) -> None:
    journal = tmp_path / "legacy.jsonl"
    journal.write_text(
        json.dumps(
            {
                "event": "started",
                "account": "C10_5bp",
                "attempt_id": "v1",
                "utc": "2026-11-09T00:00:00Z",
                "output": "legacy",
                "error": "",
            }
        )
        + "\n"
    )
    original = journal.read_bytes()
    with pytest.raises(ValueError, match="standard_id"):
        journal_event(journal, "started", "C10", "v2", tmp_path)
    assert journal.read_bytes() == original


def test_completed_saved_artifact_is_revalidated_without_another_native_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import c10standard.run as run
    from c10standard.models import Receipt

    contract = Contract()
    folder = tmp_path / contract.key
    folder.mkdir()
    (folder / "ARTIFACT.json").write_text("savedaccount")
    (folder / "NATIVE_COMPLETE.json").write_text("completed")
    monkeypatch.setattr(
        run,
        "validate_saved",
        lambda *args: Receipt(account=contract.key, hashes={}, max_differences={}),
    )
    monkeypatch.setattr(run.native, "simulate", lambda *args: pytest.fail("native rerun forbidden"))
    run.execute(contract, prepared(), tmp_path, tmp_path / "journal.jsonl", tmp_path)
    events = [
        json.loads(line)["event"] for line in (tmp_path / "journal.jsonl").read_text().splitlines()
    ]
    assert events == ["revalidated"]
    assert (folder / "ARTIFACT.json").read_text() == "savedaccount"


def test_native_failure_is_registered_before_entry_and_preserves_its_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import c10standard.run as run

    contract = Contract()
    root = Path(__file__).resolve().parents[1]
    journal = tmp_path / "journal.jsonl"

    def fail(*args: object) -> None:
        assert json.loads(journal.read_text().splitlines()[-1])["event"] == "started"
        raise RuntimeError("representative native fixture error")

    monkeypatch.setattr(run.native, "simulate", fail)
    with pytest.raises(RuntimeError, match="representative native fixture error"):
        run.execute(contract, prepared(), tmp_path, journal, root)
    events = [json.loads(line)["event"] for line in journal.read_text().splitlines()]
    assert events == ["started", "failed"]
    assert (tmp_path / contract.key / "FAILURE.txt").exists()
    with pytest.raises(ValueError, match="preserved"):
        run.execute(contract, prepared(), tmp_path, journal, root)
    assert len(journal.read_text().splitlines()) == 2


def test_public_data_and_all_reference_hashes_load_without_private_repository_access() -> None:
    from c10standard.io import ROOT, load_accounts, load_market, reference_folder

    accounts = load_accounts(ROOT)
    assert len(accounts) == 36
    for account in accounts:
        reference_folder(account, ROOT)
    for cutoff, count in (("2025", 1256), ("2026", 1470)):
        market = load_market(ROOT, cutoff)
        assert len(market.datasets) == 35 and len(market.risk_names) == 34
        assert {len(data.frame) for data in market.datasets} == {count}


def test_future_signal_rows_and_ineligible_high_momentum_cannot_change_valid_prior_choice() -> None:
    data = prepared()
    day = date(2025, 12, 31)
    contract = Contract(rule="C11", cohort="reset2026")
    original = target(contract, data, day)
    for frame in data.features.frames.values():
        frame.loc[pd.Timestamp("2026-01-30")] = 1e12
    assert target(contract, data, day) == original
    data.features.frames["QQQ"].loc[pd.Timestamp(day), "turnover20"] = 1.0
    assert "QQQ" not in target(contract, data, day).q


def saved_reference(contract: Contract, market: Prepared):
    """Read actual native exports; absent historical benchmark vectors stay absent."""
    from pydantic import TypeAdapter

    from c10standard.io import ROOT, load_accounts, reference_folder
    from c10standard.models import Artifact, NativeReturn, NativeScalar, OpenTrade, TargetRecord
    from c10standard.targets import feed_names
    from rotbt.models import (
        ClosedTrade,
        DailyRecord,
        Distribution,
        HoldingRecord,
        OrderRecord,
        Result,
        Spec,
    )

    account = next(a for a in load_accounts(ROOT) if a.contract == contract)
    folder = reference_folder(account, ROOT)

    def book(name, model):
        return TypeAdapter(list[model]).validate_python(
            pd.read_csv(folder / f"{name}.csv.gz", float_precision="round_trip").to_dict("records")
        )

    result = Result(
        spec=Spec(name=contract.key, layer="hold", lookback=252, top=3),
        daily=book("daily", DailyRecord),
        orders=book("orders", OrderRecord),
        holdings=book("holdings", HoldingRecord),
        distributions=book("dividends", Distribution),
        trades=book("trades", ClosedTrade),
        native_total_return=NativeScalar.model_validate_json(
            (folder / "native_total_return.json").read_text()
        ).native_total_return,
        splits=[
            event
            for data in market.datasets
            if data.symbol in feed_names(contract, market.risk_names)
            for event in data.splits
            if contract.start <= event.day <= contract.through
        ],
    )
    return Artifact(
        contract=contract,
        account=result,
        targets=TypeAdapter(list[TargetRecord]).validate_json(
            (folder / "targets.json").read_text()
        ),
        native_returns=TypeAdapter(list[NativeReturn]).validate_json(
            (folder / "native_returns.json").read_text()
        )
        if (folder / "native_returns.json").exists()
        else [],
        open_trades=TypeAdapter(list[OpenTrade]).validate_json(
            (folder / "open_trades.json").read_text()
        )
        if (folder / "open_trades.json").exists()
        else [],
    )


def test_ported_audit_checks_real_saved_accounts_and_rejects_a_wrong_native_day_return() -> None:
    from c10standard.audit import audit
    from c10standard.io import load_market, settings_for

    contract = Contract(rule="C11", cohort="continuous2022_2026")
    market = load_market(cutoff="2026")
    artifact = saved_reference(contract, market)
    settings = settings_for(market.strategy, contract)
    audit(artifact, market, settings)
    first = artifact.native_returns[0].model_copy(
        update={"value": artifact.native_returns[0].value + 0.01}
    )
    changed = artifact.model_copy(update={"native_returns": [first, *artifact.native_returns[1:]]})
    with pytest.raises(ValueError, match="daily native TimeReturn"):
        audit(changed, market, settings)


def test_scalar_only_old_benchmark_reference_does_not_require_a_fabricated_native_vector(
    tmp_path: Path,
) -> None:
    import shutil

    from c10standard.io import load_accounts, load_market, reference_folder
    from c10standard.run import compare_reference

    contract = Contract(rule="SPY_BH")
    market = load_market()
    account = next(a for a in load_accounts() if a.contract == contract)
    source = reference_folder(account)
    for path in source.iterdir():
        shutil.copyfile(path, tmp_path / path.name)
    assert not (source / "native_returns.json").exists()
    artifact = saved_reference(contract, market)
    assert artifact.native_returns == []
    result = compare_reference(contract, artifact, tmp_path)
    assert result["native_total_return"] == 0.0
    assert "native_returns.json" not in result


def test_v2_configuration_cannot_describe_rules_different_from_the_executed_standard() -> None:
    original = strategy().model_dump()
    for field, value in (
        ("lookback", 126),
        ("group_cap", False),
        ("deployment", 1.0),
        ("fresh_OOS", True),
    ):
        with pytest.raises(ValueError):
            Strategy.model_validate({**original, field: value})


def test_budget_refusal_is_not_recorded_as_an_attempt_that_never_entered_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import c10standard.run as run

    journal = tmp_path / "shared.jsonl"
    for n in range(96):
        journal_event(journal, "started", "fixture", str(n), tmp_path)
    before = journal.read_bytes()
    monkeypatch.setattr(
        run.native, "simulate", lambda *args: pytest.fail("over-budget native entry")
    )
    with pytest.raises(ValueError, match="96"):
        execute(
            Contract(),
            prepared(),
            tmp_path / "output",
            journal,
            Path(__file__).resolve().parents[1],
        )
    assert journal.read_bytes() == before


def test_saved_wrong_native_scalar_is_rejected_before_verification_without_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import c10standard.run as run
    from c10standard.io import load_market

    contract = Contract()
    market = load_market()
    artifact = saved_reference(contract, market)
    run.native.write(tmp_path / "ARTIFACT.json", artifact)
    run.native.write(tmp_path / "CONTRACT.json", contract)
    run.write_json(
        tmp_path / "NATIVE_COMPLETE.json",
        {"artifact_sha256": run.sha(tmp_path / "ARTIFACT.json")},
    )
    wrong = {"native_total_return": artifact.account.native_total_return + 0.01}
    run.write_json(tmp_path / "native_total_return.json", wrong)
    monkeypatch.setattr(
        run.native, "simulate", lambda *args: pytest.fail("saved validation must not run native")
    )

    with pytest.raises(ValueError, match="saved native total return"):
        run.validate_saved(contract, market, tmp_path)

    assert not (tmp_path / "VERIFIED.json").exists()
    assert json.loads((tmp_path / "native_total_return.json").read_text()) == wrong
