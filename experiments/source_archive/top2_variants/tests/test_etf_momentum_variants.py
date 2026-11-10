"""Pure fixed-mechanism tests; no account, API or parameter search."""

import json
import math
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pytest

from rotbt.models import Dataset
from scripts import etf_momentum_variants as study
from scripts.etf_group_cap_inputs import features_for
from scripts.etf_group_cap_targets import GroupMap

ROOT = Path(__file__).resolve().parents[1]


def trial(variant: study.Variant = "E01") -> study.Trial:
    return study.Trial(
        key=f"{variant}_K2_5bp_reset2026",
        variant=variant,
        cost_bps=5,
        cohort="reset2026",
        kind="science",
        k=2,
    )


def data() -> list[Dataset]:
    days = (
        xcals.get_calendar("XNYS").sessions_in_range("2020-12-31", "2022-03-31").tz_localize(None)
    )
    result = []
    for symbol, rate in (("A", 0.001), ("B", 0.0008), ("C", 0.0006), ("SHY", 0.0001)):
        tr = [100.0 * math.exp(rate * n) for n in range(len(days))]
        frame = pd.DataFrame(
            {
                "tr": tr,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1e6,
                "turnover": 20e6,
            },
            index=days,
        )
        result.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    return result


def test_skip21_score_excludes_recent_trading_days_exactly() -> None:
    frame = data()[0].frame
    day = frame.index[-1].date()
    metric = study.score("E01", features_for([data()[0]]).frames["A"], day)
    assert metric.rank == pytest.approx(float(frame.tr.iloc[-22] / frame.tr.iloc[-253] - 1))
    frame.loc[frame.index[-21:], "tr"] = 1e8
    assert study.score(
        "E01", features_for([Dataset(symbol="A", frame=frame, dividends=[])]).frames["A"], day
    ).rank == pytest.approx(metric.rank)


def test_equal_calendar_weights_use_exact_month_ends_including_holidays() -> None:
    source = data()[0]
    day = date(2022, 3, 31)
    points = study.prior.calendar_endpoints(day)
    expected = (
        sum(
            float(
                source.frame.tr.loc[pd.Timestamp(day)]
                / source.frame.tr.loc[pd.Timestamp(points[m])]
                - 1
            )
            for m in (1, 3, 6, 12)
        )
        / 4
    )
    assert study.score("E02", features_for([source]).frames["A"], day).rank == pytest.approx(
        expected
    )
    assert study.prior.calendar_endpoints(date(2024, 3, 28))[12] == date(2023, 3, 31)


def test_risk_adjusted_rank_has_one_percent_floor_and_keeps_unscaled_gate() -> None:
    frame = features_for([data()[0]]).frames["A"]
    day = frame.index[-1].date()
    metric = study.score("E03", frame, day)
    assert metric.rank == pytest.approx(metric.r252 / 0.01)
    assert metric.gate_value == metric.r252


def test_regression_uses_last252_log_prices_and_r_squared_not_annual_compounding() -> None:
    source = data()[0]
    metric = study.score("E04", features_for([source]).frames["A"], source.frame.index[-1].date())
    assert metric.rank == pytest.approx(0.001 * 252)
    assert metric.gate_value == pytest.approx(math.exp(0.001 * 252) - 1)
    assert metric.rank != pytest.approx(metric.gate_value)


def test_extra_defenses_apply_after_fixed_group_slots_without_reallocation() -> None:
    source = data()
    features = features_for(source)
    day = source[0].frame.index[-1].date()
    groups = GroupMap(groups={"one": ("A", "B"), "two": ("C",)})
    for frame in features.frames.values():
        frame.loc[pd.Timestamp(day), "mom252"] = -0.1
    features.frames["A"].loc[pd.Timestamp(day), "mom252"] = -0.01
    features.frames["C"].loc[pd.Timestamp(day), "mom252"] = -0.02
    assert study.decision(trial("E05"), features, ("A", "B", "C"), groups, day).q == {"SHY": 1.0}
    features.frames["A"].loc[pd.Timestamp(day), "mom252"] = 0.5
    features.frames["C"].loc[pd.Timestamp(day), "mom252"] = 0.3
    features.frames["A"].loc[pd.Timestamp(day), "sma200"] = 1e9
    result = study.decision(trial("E06"), features, ("A", "B", "C"), groups, day)
    assert result.q == pytest.approx({"C": 0.45, "SHY": 0.55})


def test_independent_raw_oracle_agrees_for_every_variant_without_using_features() -> None:
    datasets = data()
    features = features_for(datasets)
    groups = GroupMap(groups={"one": ("A", "B"), "two": ("C",)})
    day = datasets[0].frame.index[-1].date()
    for variant in study.VARIANTS:
        expected = study.decision(trial(variant), features, ("A", "B", "C"), groups, day)
        actual = study.raw_oracle(trial(variant), datasets, ("A", "B", "C"), groups, day)
        assert actual.q == pytest.approx(expected.q)


def test_future_observations_cannot_change_any_variant_target() -> None:
    datasets = data()
    features = features_for(datasets)
    groups = GroupMap(groups={"one": ("A", "B"), "two": ("C",)})
    day = date(2022, 2, 28)
    expected = {
        v: study.decision(trial(v), features, ("A", "B", "C"), groups, day) for v in study.VARIANTS
    }
    for frame in features.frames.values():
        frame.loc[frame.index > pd.Timestamp(day), :] = 1e12
    assert all(
        study.decision(trial(v), features, ("A", "B", "C"), groups, day) == expected[v]
        for v in study.VARIANTS
    )


def test_catalog_identity_is_exactly36_science_and_two_k2_bridges() -> None:
    trials = study.catalog(ROOT)
    assert len(trials) == 38 and len({t.key for t in trials}) == 38
    assert sum(t.kind == "science" for t in trials) == 36
    assert all(t.k == 2 and study.contract_for(t).rule == "C11" for t in trials)


def test_hooks_restore_after_failure_and_independent48_budget_counts_failed_entries(
    tmp_path: Path,
) -> None:
    original = vars(study.native)["target"], vars(study.native)["oracle"]
    with pytest.raises(RuntimeError):
        with study.scoped_hooks(trial(), GroupMap(groups={})):
            raise RuntimeError("fixture")
    assert (vars(study.native)["target"], vars(study.native)["oracle"]) == original
    for n in range(48):
        study.append_attempt(tmp_path, str(n), "started")
        study.append_attempt(tmp_path, str(n), "failed", "fixture")
    with pytest.raises(ValueError, match="48"):
        study.append_attempt(tmp_path, "overflow", "started")
    assert study.native_count(tmp_path) == 48


def test_success_revalidates_without_native_and_partial_evidence_stays_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = trial()
    folder = tmp_path / study.OUT / selected.key
    folder.mkdir(parents=True)
    (folder / "ARTIFACT.json").write_text("saved")
    (folder / "NATIVE_COMPLETE.json").write_text("complete")
    monkeypatch.setattr(study, "validate_saved", lambda *args: None)
    monkeypatch.setattr(study.native, "_native", lambda *args: pytest.fail("native rerun"))
    study.execute(tmp_path, selected, None, GroupMap(groups={}))
    assert study.native_count(tmp_path) == 0
    assert (
        json.loads(study.journal(tmp_path).read_text().splitlines()[-1])["state"] == "revalidated"
    )
    (folder / "NATIVE_COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="preserved"):
        study.execute(tmp_path, selected, None, GroupMap(groups={}))


def test_volatility_floor_never_hides_nonfinite_or_negative_input() -> None:
    frame = features_for([data()[0]]).frames["A"]
    day = frame.index[-1].date()
    frame.loc[pd.Timestamp(day), "vol63"] = 0.0
    assert study.score("E03", frame, day).rank == pytest.approx(float(frame.mom252.iloc[-1]) / 0.01)
    for invalid in (float("nan"), float("inf"), -0.1):
        frame.loc[pd.Timestamp(day), "vol63"] = invalid
        with pytest.raises(ValueError, match="volatility"):
            study.score("E03", frame, day)


def test_rank_only_mechanisms_do_not_also_replace_the_original_cash_gate() -> None:
    features = features_for(data())
    day = date(2022, 3, 31)
    groups = GroupMap(groups={"one": ("A", "B"), "two": ("C",)})
    for name, momentum, volatility, regression in (
        ("A", 0.2, 10.0, -0.3),
        ("C", 0.1, 0.01, 0.7),
        ("SHY", 0.05, 0.01, 0.8),
    ):
        frame = features.frames[name]
        frame.loc[pd.Timestamp(day), ["mom252", "vol63", "reg252"]] = [
            momentum,
            volatility,
            regression,
        ]
    for variant in ("E03", "E04"):
        assert study.decision(trial(variant), features, ("A", "C"), groups, day).q == pytest.approx(
            {"A": 0.45, "C": 0.45, "SHY": 0.1}
        )


def test_constant_and_negative_log_trends_have_zero_and_negative_regression_scores() -> None:
    datasets = data()
    day = datasets[0].frame.index[-1].date()
    datasets[0].frame["tr"] = 100.0
    datasets[1].frame["tr"] = [100 * math.exp(-0.002 * n) for n in range(len(datasets[1].frame))]
    features = features_for(datasets)
    assert study.score("E04", features.frames["A"], day).rank == 0.0
    assert study.score("E04", features.frames["B"], day).rank == pytest.approx(-0.002 * 252)
    groups = GroupMap(groups={"one": ("A",), "two": ("B",)})
    assert study.raw_oracle(trial("E04"), datasets, ("A", "B"), groups, day).q == {"SHY": 1.0}


def test_real_all_month_end_targets_agree_with_independent_raw_oracles() -> None:
    selected = next(
        t for t in study.catalog(ROOT) if t.variant == "E01" and t.cohort == "continuous2022_2026"
    )
    prepared = study.prepared_for(ROOT, selected)
    _, groups = study.configuration(ROOT)
    trials = {
        v: next(
            t for t in study.catalog(ROOT) if t.variant == v and t.cohort == "continuous2022_2026"
        )
        for v in study.VARIANTS
    }
    for day in (date(2021, 12, 31), *study.native.decision_days(study.contract_for(selected))):
        for variant in study.VARIANTS:
            full = study.decision(
                trials[variant], prepared.features, prepared.risk_names, groups, day
            )
            raw = study.raw_oracle(
                trials[variant], prepared.datasets, prepared.risk_names, groups, day
            )
            assert full.q == pytest.approx(raw.q)


def test_completed_native_revalidation_rejects_an_inconsistent_saved_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = next(
        t for t in study.catalog(ROOT) if t.kind == "bridge" and t.cohort == "historical2022_2025"
    )
    prepared = study.prepared_for(ROOT, bridge)
    _, groups = study.configuration(ROOT)
    source = (
        ROOT
        / "etf_group_cap_outputs/20261109_v1"
        / "historical34_cap1_K2_5bp_historical2022_2025/ARTIFACT.json"
    )
    artifact = study.native.Artifact.model_validate_json(source.read_text())
    folder = tmp_path / study.OUT / bridge.key
    folder.mkdir(parents=True)
    for name, value in (
        ("ARTIFACT.json", artifact),
        ("TRIAL.json", bridge),
        ("CONTRACT.json", study.contract_for(bridge)),
    ):
        study.model_write(folder / name, value)
    study.write(
        folder / "NATIVE_COMPLETE.json", {"artifact_sha256": study.sha(folder / "ARTIFACT.json")}
    )
    wrong = {"native_total_return": artifact.account.native_total_return + 0.01}
    study.write(folder / "NATIVE_TOTAL_RETURN.json", wrong)
    from scripts.etf_group_cap_audit import settings_for

    monkeypatch.setattr(study, "settings_for", lambda root, role: settings_for(ROOT, role))
    monkeypatch.setattr(
        study.native, "_native", lambda *args: pytest.fail("native rerun forbidden")
    )
    with pytest.raises(ValueError, match="NATIVE_TOTAL_RETURN"):
        study.validate_saved(tmp_path, bridge, prepared, groups)
    assert not (folder / "VERIFIED.json").exists()
    assert json.loads((folder / "NATIVE_TOTAL_RETURN.json").read_text()) == wrong
    assert study.native_count(tmp_path) == 0


def test_prefix_rebuild_detects_noncausal_features_and_retains_actual_target_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = data()
    features = features_for(datasets)
    groups = GroupMap(groups={"one": ("A", "B"), "two": ("C",)})
    prepared = study.Prepared(
        datasets=datasets,
        features=features,
        risk_names=("A", "B", "C"),
        groups=(),
        first=date(2020, 12, 31),
        last=date(2025, 12, 31),
    )
    folder = tmp_path / study.RESEARCH
    folder.mkdir(parents=True)
    (folder / "RUN_LOCK.json").write_text("{}")
    registered = study.catalog(ROOT)
    monkeypatch.setattr(study, "catalog", lambda root: registered)
    monkeypatch.setattr(study, "verify_run_lock", lambda root: {})
    monkeypatch.setattr(study, "configuration", lambda root: (None, groups))
    monkeypatch.setattr(study, "prepared_for", lambda *args: prepared)
    monkeypatch.setattr(study.native, "decision_days", lambda contract: ())
    original = float(features.frames["A"].loc[pd.Timestamp("2021-12-31"), "reg252"])
    features.frames["A"].loc[pd.Timestamp("2021-12-31"), "reg252"] = original + 0.01
    with pytest.raises(AssertionError):
        study.causality(tmp_path)
    assert not (folder / "CAUSALITY_RECEIPT.json").exists()
    features.frames["A"].loc[pd.Timestamp("2021-12-31"), "reg252"] = original
    study.causality(tmp_path)
    receipt = json.loads((folder / "CAUSALITY_RECEIPT.json").read_text())
    assert receipt["prefix_rebuilds"] == 1 and receipt["variant_targets"] == 6
    assert all(r["full"] == r["prefix"] == r["raw"] for r in receipt["observations"])
    assert receipt["native_runs"] == 0
