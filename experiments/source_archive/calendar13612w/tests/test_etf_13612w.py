"""Calendar scores and native adapter contracts; no Cerebro or network invocation."""

import json
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pytest

from rotbt.models import Dataset
from rotbt.phase2.models import FeatureStore
from scripts import etf_13612w as study
from scripts.etf_group_cap_targets import GroupMap

ROOT = Path(__file__).resolve().parents[1]


def trial(k: int = 3) -> study.Trial:
    return study.Trial(
        key=f"W13612_K{k}_5bp_reset2026",
        k=k,
        cost_bps=5,
        cohort="reset2026",
        kind="science",
        score="calendar_13612w",
    )


def sample() -> tuple[list[Dataset], FeatureStore, GroupMap]:
    days = (
        xcals.get_calendar("XNYS").sessions_in_range("2020-12-31", "2022-01-31").tz_localize(None)
    )
    datasets = []
    for symbol, score in (("A", 0.5), ("B", 0.4), ("C", 0.3), ("D", 0.2), ("SHY", 0.25)):
        values = pd.Series(100.0, index=days)
        values.loc["2021-12-31"] = 100.0 * (1 + score / 4.75)
        frame = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000.0,
                "turnover": 20_000_000.0,
                "tr": values,
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
    return (
        datasets,
        features,
        GroupMap(groups={"broad": ("A", "B"), "second": ("C",), "third": ("D",)}),
    )


def test_calendar_endpoints_use_true_month_ends_including_holidays_and_short_months() -> None:
    assert study.calendar_endpoints(date(2024, 3, 28)) == {
        0: date(2024, 3, 28),
        1: date(2024, 2, 29),
        3: date(2023, 12, 29),
        6: date(2023, 9, 29),
        12: date(2023, 3, 31),
    }
    assert study.calendar_endpoints(date(2021, 12, 31))[12] == date(2020, 12, 31)
    with pytest.raises(ValueError, match="month.end"):
        study.calendar_endpoints(date(2024, 3, 27))


def test_score_weights_divide_by_four_and_use_calendar_endpoints() -> None:
    _, features, _ = sample()
    frame = features.frames["A"].copy()
    frame.loc["2021-12-31", "tr"] = 150.0
    frame.loc["2021-11-30", "tr"] = 125.0
    frame.loc["2021-09-30", "tr"] = 100.0
    frame.loc["2021-06-30", "tr"] = 75.0
    frame.loc["2020-12-31", "tr"] = 50.0
    assert study.calendar_score(frame, date(2021, 12, 31)) == pytest.approx(
        (12 * 0.2 + 4 * 0.5 + 2 * 1.0 + 2.0) / 4.0
    )
    assert frame.index[-22].date() != date(2021, 11, 30)


def test_group_quota_is_applied_before_shy_gate_and_failed_slots_stay_defensive() -> None:
    datasets, features, groups = sample()
    day = date(2021, 12, 31)
    names = ("A", "B", "C", "D")
    actual = study.decision(trial(), features, names, groups, day)
    assert actual.q == pytest.approx({"A": 0.3, "C": 0.3, "SHY": 0.4})
    for frame in features.frames.values():
        frame.loc[:, "tr"] = -999.0
    assert study.raw_oracle(trial(), datasets, names, groups, day).q == pytest.approx(actual.q)


def test_missing_nonfinite_or_nonpositive_exact_endpoint_fails_instead_of_padding() -> None:
    _, features, _ = sample()
    frame = features.frames["A"].loc[:"2021-12-31"].copy()
    for bad in (float("nan"), 0.0, float("inf")):
        changed = frame.copy()
        changed.loc["2021-11-30", "tr"] = bad
        with pytest.raises(ValueError, match="endpoint"):
            study.calendar_score(changed, date(2021, 12, 31))
    with pytest.raises(ValueError, match="endpoint"):
        study.calendar_score(frame.drop(pd.Timestamp("2021-11-30")), date(2021, 12, 31))


def test_future_data_cannot_change_prior_score_or_selected_targets() -> None:
    _, features, groups = sample()
    day = date(2021, 12, 31)
    names = ("A", "B", "C", "D")
    expected = study.decision(trial(), features, names, groups, day)
    for frame in features.frames.values():
        frame.loc[frame.index > pd.Timestamp(day), :] = 1e12
    assert study.decision(trial(), features, names, groups, day) == expected


def test_catalog_preserves_eighteen_science_two_bridges_and_exact_cohort_contracts() -> None:
    trials = study.catalog(ROOT)
    assert len(trials) == 20 and sum(t.kind == "science" for t in trials) == 18
    assert len({t.key for t in trials}) == 20
    for t in trials:
        co = study.contract_for(t)
        assert co.through < date.max
        assert (co.mode == "bridge") == (t.cohort == "historical2022_2025")


def test_hooks_restore_both_original_functions_after_a_native_failure() -> None:
    from scripts import etf_c10_2026_run as native

    originals = vars(native)["target"], vars(native)["oracle"]
    with pytest.raises(RuntimeError, match="fixture"):
        with study.scoped_hooks(trial(), GroupMap(groups={})):
            assert vars(native)["target"] is not originals[0]
            raise RuntimeError("fixture")
    assert (vars(native)["target"], vars(native)["oracle"]) == originals


def test_separate_journal_counts_failed_attempts_and_refuses_twenty_ninth(tmp_path: Path) -> None:
    for n in range(28):
        study.append_attempt(tmp_path, str(n), "started")
        study.append_attempt(tmp_path, str(n), "failed", "fixture")
    with pytest.raises(ValueError, match="28"):
        study.append_attempt(tmp_path, "overflow", "started")
    assert study.native_count(tmp_path) == 28


def test_saved_success_revalidation_spends_no_additional_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / study.OUT / trial().key
    folder.mkdir(parents=True)
    (folder / "ARTIFACT.json").write_text("saved")
    (folder / "NATIVE_COMPLETE.json").write_text("done")
    monkeypatch.setattr(study, "validate_saved", lambda *args: None)
    monkeypatch.setattr(
        study.native, "_native", lambda *args: pytest.fail("native rerun forbidden")
    )
    study.execute(tmp_path, trial(), None, GroupMap(groups={}))
    assert study.native_count(tmp_path) == 0
    assert (
        json.loads(study.journal(tmp_path).read_text().splitlines()[-1])["state"] == "revalidated"
    )


def test_partial_native_evidence_cannot_be_overwritten(tmp_path: Path) -> None:
    folder = tmp_path / study.OUT / trial().key
    folder.mkdir(parents=True)
    (folder / "ARTIFACT.json").write_text("partial but no complete marker")
    with pytest.raises(ValueError, match="preserved"):
        study.execute(tmp_path, trial(), None, GroupMap(groups={}))
    assert study.native_count(tmp_path) == 0


def test_equal_scores_use_symbol_tie_break_and_strict_shy_gate() -> None:
    datasets, features, groups = sample()
    day = date(2021, 12, 31)
    features.frames["C"].loc[pd.Timestamp(day), "tr"] = features.frames["A"].loc[
        pd.Timestamp(day), "tr"
    ]
    assert study.decision(trial(1), features, ("C", "A"), groups, day).q["A"] == pytest.approx(0.9)
    features.frames["SHY"].loc[pd.Timestamp(day), "tr"] = features.frames["A"].loc[
        pd.Timestamp(day), "tr"
    ]
    assert study.decision(trial(2), features, ("C", "A"), groups, day).q == {"SHY": 1.0}


def test_qualification_excludes_illiquid_or_short_history_despite_a_high_score() -> None:
    _, features, groups = sample()
    day = date(2021, 12, 31)
    for column, value in (("turnover20", 1.0), ("positive20", 19.0), ("raw_close", 4.99)):
        changed = features.model_copy(deep=True)
        changed.frames["A"].loc[pd.Timestamp(day), column] = value
        assert "A" not in study.decision(trial(1), changed, ("A", "C"), groups, day).q
    features.frames["A"] = features.frames["A"].iloc[30:]
    assert "A" not in study.decision(trial(1), features, ("A", "C"), groups, day).q


def test_bridge_hooks_preserve_original252_targets_and_raw_oracle() -> None:
    datasets, features, groups = sample()
    day = date(2021, 12, 31)
    bridge = study.Trial(
        key="bridge_K1_5bp_historical2022_2025",
        k=1,
        cost_bps=5,
        cohort="historical2022_2025",
        kind="bridge",
        score="original252",
    )
    expected = study.original_decision(
        study.role_for(bridge), features, ("A", "B", "C", "D"), groups, day
    )
    with study.scoped_hooks(bridge, groups):
        actual = vars(study.native)["target"]("C10", features, day, ("A", "B", "C", "D"), ())
        audited = vars(study.native)["oracle"]("C10", datasets, ("A", "B", "C", "D"), day)
    assert actual == expected.q and audited.q == pytest.approx(expected.q)


def test_true_accepted_month_end_targets_and_independent_raw_oracle_agree() -> None:
    from scripts.etf_group_cap_run import configuration

    prepared = study.prepared_for(ROOT, trial())
    _, groups = configuration(ROOT)
    contract = study.native.Contract(rule="C00", cost=5, mode="continuous2022")
    days = (date(2021, 12, 31), *study.native.decision_days(contract))
    assert date(2024, 3, 28) in days
    assert study.calendar_endpoints(date(2024, 3, 28))[12] == date(2023, 3, 31)
    assert study.calendar_endpoints(date(2025, 12, 31))[12] == date(2024, 12, 31)
    for day in days:
        actual = study.decision(trial(), prepared.features, prepared.risk_names, groups, day)
        expected = study.raw_oracle(trial(), prepared.datasets, prepared.risk_names, groups, day)
        assert actual.q == pytest.approx(expected.q)


def test_raw_oracle_independently_rejects_a_missing_calendar_lag_endpoint() -> None:
    datasets, _, groups = sample()
    datasets[0].frame = datasets[0].frame.drop(pd.Timestamp("2021-12-31"))
    with pytest.raises(ValueError, match="endpoint"):
        study.raw_oracle(trial(), datasets, ("A", "B", "C", "D"), groups, date(2022, 1, 31))


def test_native_failure_is_metered_before_entry_and_remains_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = next(t for t in study.catalog(ROOT) if t.kind == "science")
    _, groups = study.configuration(ROOT)
    prepared = study.prepared_for(ROOT, selected)
    monkeypatch.setattr(study, "catalog", lambda root: [selected])
    from scripts.etf_group_cap_audit import settings_for

    monkeypatch.setattr(study, "settings_for", lambda root, role: settings_for(ROOT, role))

    def fail(*args: object) -> None:
        assert study.native_count(tmp_path) == 1
        raise RuntimeError("representative native failure")

    monkeypatch.setattr(study.native, "_native", fail)
    with pytest.raises(RuntimeError, match="representative native failure"):
        study.execute(tmp_path, selected, prepared, groups)
    assert study.native_count(tmp_path) == 1
    assert (tmp_path / study.OUT / selected.key / "FAILURE.txt").exists()
    with pytest.raises(ValueError, match="preserved"):
        study.execute(tmp_path, selected, prepared, groups)
    assert study.native_count(tmp_path) == 1


def test_real_completed_artifact_revalidation_rejects_a_wrong_saved_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = next(t for t in study.catalog(ROOT) if t.kind == "bridge" and t.k == 1)
    prepared = study.prepared_for(ROOT, bridge)
    _, groups = study.configuration(ROOT)
    source = (
        ROOT
        / "etf_group_cap_outputs/20261109_v1"
        / "historical34_uncapped_K1_5bp_historical2022_2025/ARTIFACT.json"
    )
    artifact = study.native.Artifact.model_validate_json(source.read_text())
    folder = tmp_path / study.OUT / bridge.key
    folder.mkdir(parents=True)
    study.model_write(folder / "ARTIFACT.json", artifact)
    study.model_write(folder / "TRIAL.json", bridge)
    study.model_write(folder / "CONTRACT.json", study.contract_for(bridge))
    study.write(
        folder / "NATIVE_COMPLETE.json", {"artifact_sha256": study.sha(folder / "ARTIFACT.json")}
    )
    study.write(
        folder / "NATIVE_TOTAL_RETURN.json",
        {"native_total_return": artifact.account.native_total_return + 0.01},
    )
    from scripts.etf_group_cap_audit import settings_for

    monkeypatch.setattr(study, "settings_for", lambda root, role: settings_for(ROOT, role))
    monkeypatch.setattr(
        study.native, "_native", lambda *args: pytest.fail("native rerun forbidden")
    )
    with pytest.raises(ValueError, match="NATIVE_TOTAL_RETURN"):
        study.validate_saved(tmp_path, bridge, prepared, groups)
    assert not (folder / "VERIFIED.json").exists()
    assert study.native_count(tmp_path) == 0
