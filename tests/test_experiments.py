"""Portable experiment contracts; every account invocation is stubbed in tests."""

import json
import math
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pytest

from c10experiments import io, run, signals
from c10experiments.models import Experiment
from c10standard.models import Prepared, Strategy
from rotbt.models import Dataset
from rotbt.phase2.features import build_features

ROOT = Path(__file__).resolve().parents[1]


def test_public_adapter_boundaries_reject_invalid_runtime_types_before_use(tmp_path: Path) -> None:
    from beartype.roar import BeartypeCallHintParamViolation
    from typeguard import TypeCheckError

    with pytest.raises((BeartypeCallHintParamViolation, TypeCheckError)):
        run.native_folder(tmp_path, "not-an-experiment")


def sample() -> Prepared:
    days = (
        xcals.get_calendar("XNYS").sessions_in_range("2020-12-31", "2022-03-31").tz_localize(None)
    )
    datasets = []
    for symbol, rate in (
        ("QQQ", 0.001),
        ("VTI", 0.0008),
        ("SOXX", 0.0006),
        ("VEA", 0.0002),
        ("SHY", 0.0001),
    ):
        frame = pd.DataFrame(
            {
                "tr": [100.0 * math.exp(rate * n) for n in range(len(days))],
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1e6,
                "turnover": 20e6,
            },
            index=days,
        )
        datasets.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    features = build_features(datasets)
    for data in datasets:
        features.frames[data.symbol]["raw_close"] = data.frame.close
        features.frames[data.symbol]["turnover20"] = data.frame.turnover.rolling(20).mean()
        features.frames[data.symbol]["positive20"] = data.frame.volume.gt(0).rolling(20).sum()
    strategy = Strategy.model_validate_json((ROOT / "standard/STRATEGY.json").read_text())
    return Prepared(
        datasets=datasets,
        features=features,
        strategy=strategy,
        risk_names=("QQQ", "VTI", "SOXX", "VEA"),
        first=days[0].date(),
        last=days[-1].date(),
    )


def experiment(variant: str = "E01", k: int = 2) -> Experiment:
    return next(
        e
        for e in io.load_experiments(ROOT)
        if e.variant == variant and e.k == k and e.cost_bps == 5 and e.cohort == "reset2026"
    )


def test_exact54_fixed_identities_bind_real_trial_files_and_distinct_native_namespaces() -> None:
    accounts = io.load_experiments(ROOT)
    assert len(accounts) == 54 and len({a.key for a in accounts}) == 54
    assert sum(a.variant == "W13612" for a in accounts) == 18
    for account in accounts:
        io.reference_folder(account, ROOT)
        assert account.contract.rule == {1: "C10", 2: "C11", 3: "C00"}[account.k]
    assert run.native_folder(Path("/tmp/output"), experiment("E01")) != run.native_folder(
        Path("/tmp/output"), experiment("E02")
    )


def test_calendar_weighted_and_equal_scores_use_true_endpoints() -> None:
    prepared = sample()
    frame = prepared.features.frames["QQQ"]
    day = date(2022, 3, 31)
    ends = signals.calendar_endpoints(day)
    returns = {
        m: float(frame.tr.loc[pd.Timestamp(day)] / frame.tr.loc[pd.Timestamp(ends[m])] - 1)
        for m in (1, 3, 6, 12)
    }
    assert signals.metric("W13612", frame, day).rank == pytest.approx(
        sum(w * returns[m] for m, w in ((1, 12), (3, 4), (6, 2), (12, 1))) / 4
    )
    assert signals.metric("E02", frame, day).rank == pytest.approx(sum(returns.values()) / 4)
    assert signals.calendar_endpoints(date(2024, 3, 28))[12] == date(2023, 3, 31)


def test_skip21_regression_and_volatility_floor_keep_their_original_economic_meaning() -> None:
    frame = sample().features.frames["QQQ"]
    day = date(2022, 3, 31)
    assert signals.metric("E01", frame, day).rank == pytest.approx(
        float(frame.tr.iloc[-22] / frame.tr.iloc[-253] - 1)
    )
    assert signals.metric("E04", frame, day).rank == pytest.approx(0.001 * 252)
    risk = signals.metric("E03", frame, day)
    assert risk.rank == pytest.approx(risk.r252 / 0.01) and risk.gate == risk.r252
    for invalid in (float("nan"), float("inf"), -0.1):
        frame.loc[pd.Timestamp(day), "vol63"] = invalid
        with pytest.raises(ValueError, match="volatility"):
            signals.metric("E03", frame, day)


def test_group_quota_and_failed_gate_preserve_slots_without_replacement() -> None:
    prepared = sample()
    day = date(2022, 3, 31)
    for name, momentum, trend in (
        ("QQQ", 0.5, 1e9),
        ("SOXX", 0.3, 1.0),
        ("VEA", 0.2, 1.0),
        ("SHY", 0.1, 1.0),
    ):
        prepared.features.frames[name].loc[pd.Timestamp(day), ["mom252", "sma200"]] = [
            momentum,
            trend,
        ]
    result = signals.target(experiment("E06"), prepared, day)
    assert result.q == pytest.approx({"SOXX": 0.45, "SHY": 0.55})
    for frame in prepared.features.frames.values():
        frame.loc[pd.Timestamp(day), "mom252"] = -0.1
    assert signals.target(experiment("E05"), prepared, day).q == {"SHY": 1.0}


def test_independent_raw_oracle_and_prefix_rebuild_agree_for_all_mechanisms() -> None:
    prepared = sample()
    day = date(2022, 2, 28)
    for variant in ("W13612", "E01", "E02", "E03", "E04", "E05", "E06"):
        account = experiment(variant)
        expected = signals.target(account, prepared, day)
        assert signals.oracle(account, prepared, day).q == pytest.approx(expected.q)
        assert io.prefix_market(prepared, day).features.frames["QQQ"].index[-1].date() == day
        assert signals.target(account, io.prefix_market(prepared, day), day) == expected


def test_scope_hooks_restore_on_failure_and_never_turn_shell_c11_into_standard_identity() -> None:
    import c10standard.audit as audit
    import c10standard.native as native
    import c10standard.run as standard

    before = (
        vars(native)["target"],
        vars(audit)["oracle"],
        standard.compare_reference,
        standard.journal_event,
    )
    with pytest.raises(RuntimeError):
        with run.scoped_hooks(experiment(), ROOT):
            assert standard.compare_reference is not before[2]
            raise RuntimeError("fixture")
    assert (
        vars(native)["target"],
        vars(audit)["oracle"],
        standard.compare_reference,
        standard.journal_event,
    ) == before


def test_new_release_journal_shares72_starts_between_default_and_experiments_and_keeps_failures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "release.jsonl"
    for n in range(72):
        io.journal_event(path, "started", "C11" if n < 3 else "E01", str(n), tmp_path)
        io.journal_event(path, "failed", "C11" if n < 3 else "E01", str(n), tmp_path, "fixture")
    with pytest.raises(ValueError, match="72"):
        io.journal_event(path, "started", "overflow", "overflow", tmp_path)
    assert (
        sum(json.loads(line)["event"] == "started" for line in path.read_text().splitlines()) == 72
    )


def test_reference_hash_drift_is_rejected_before_use(tmp_path: Path) -> None:
    import shutil

    account = experiment()
    source = io.reference_folder(account, ROOT)
    target = tmp_path / account.reference_path
    target.mkdir(parents=True)
    for p in source.iterdir():
        shutil.copyfile(p, target / p.name)
    (target / "NATIVE_TOTAL_RETURN.json").write_text('{"native_total_return":999}')
    with pytest.raises(ValueError, match="hash"):
        io.reference_folder(account, tmp_path)


def test_an_old_budget_journal_is_rejected_without_relabelling_its_attempts(tmp_path: Path) -> None:
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "started",
                "account": "C11",
                "attempt_id": "old",
                "utc": "2026-11-10T00:00:00Z",
                "output": "old",
                "error": "",
            }
        )
        + "\n"
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="release_id"):
        io.journal_event(path, "started", "E01", "new", tmp_path)
    assert path.read_bytes() == before


def test_rank_only_variants_keep_unscaled_shy_gate_and_equal_gate_is_strict() -> None:
    prepared = sample()
    day = date(2022, 3, 31)
    prepared.risk_names = ("QQQ", "SOXX")
    for name, r252, volatility, regression in (
        ("QQQ", 0.2, 10.0, -0.3),
        ("SOXX", 0.1, 0.01, 0.7),
        ("SHY", 0.05, 0.01, 0.8),
    ):
        prepared.features.frames[name].loc[pd.Timestamp(day), ["mom252", "vol63", "reg252"]] = [
            r252,
            volatility,
            regression,
        ]
    for variant in ("E03", "E04"):
        assert signals.target(experiment(variant), prepared, day).q == pytest.approx(
            {"QQQ": 0.45, "SOXX": 0.45, "SHY": 0.1}
        )
    prepared.features.frames["QQQ"].loc[pd.Timestamp(day), "mom252"] = 0.05
    assert "QQQ" not in signals.target(experiment("E03"), prepared, day).q


def test_partial_native_evidence_is_preserved_without_an_account_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = experiment()
    folder = run.native_folder(tmp_path, account)
    folder.mkdir(parents=True)
    (folder / "PARTIAL_NATIVE.json").write_text("preserved")
    monkeypatch.setattr(
        run.native, "simulate", lambda *args: pytest.fail("partial native retry forbidden")
    )
    with pytest.raises(ValueError, match="preserved"):
        run.execute(account, sample(), tmp_path, tmp_path / "journal.jsonl", ROOT)
    assert (folder / "PARTIAL_NATIVE.json").read_text() == "preserved"
    assert not (tmp_path / "journal.jsonl").exists()


def test_default_cli_only_verifies_references_without_starting_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["c10experiments.run", "--root", str(ROOT)])
    monkeypatch.setattr(
        run.native, "simulate", lambda *args: pytest.fail("default CLI cannot run native")
    )
    run.main()


def test_saved_wrong_scalar_is_rejected_by_reused_standard_validator_without_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import TypeAdapter

    import c10standard.run as standard
    from c10share.verify import compare_frames
    from c10standard.models import Artifact, NativeReturn, NativeScalar, OpenTrade, TargetRecord
    from rotbt.models import (
        ClosedTrade,
        DailyRecord,
        Distribution,
        HoldingRecord,
        OrderRecord,
        Result,
        Spec,
    )

    account = next(
        e
        for e in io.load_experiments(ROOT)
        if e.variant == "E05" and e.cost_bps == 5 and e.cohort == "historical2022_2025"
    )
    prepared = io.load_market(ROOT, "2025")
    source = io.reference_folder(account, ROOT)
    baseline = ROOT / "standard/reference/C11_5bp_historical2022_2025"
    # E05 and the frozen baseline have identical books. Reuse its actual saved
    # native vector for this test; never synthesize one from NAV.
    for book in ("daily", "orders", "holdings", "dividends", "trades"):
        compare_frames(
            pd.read_csv(source / f"{book}.csv.gz", float_precision="round_trip"),
            pd.read_csv(baseline / f"{book}.csv.gz", float_precision="round_trip"),
            book,
        )

    def records(name, model):
        return TypeAdapter(list[model]).validate_python(
            pd.read_csv(source / f"{name}.csv.gz", float_precision="round_trip").to_dict("records")
        )

    result = Result(
        spec=Spec(name=account.key, layer="hold", lookback=252, top=3),
        daily=records("daily", DailyRecord),
        orders=records("orders", OrderRecord),
        holdings=records("holdings", HoldingRecord),
        distributions=records("dividends", Distribution),
        trades=records("trades", ClosedTrade),
        native_total_return=NativeScalar.model_validate_json(
            (source / "NATIVE_TOTAL_RETURN.json").read_text()
        ).native_total_return,
        splits=[
            s
            for d in prepared.datasets
            for s in d.splits
            if account.contract.start <= s.day <= account.contract.through
        ],
    )
    artifact = Artifact(
        contract=account.contract,
        account=result,
        targets=TypeAdapter(list[TargetRecord]).validate_json(
            (source / "TARGETS.json").read_text()
        ),
        native_returns=TypeAdapter(list[NativeReturn]).validate_json(
            (baseline / "native_returns.json").read_text()
        ),
        open_trades=TypeAdapter(list[OpenTrade]).validate_json(
            (baseline / "open_trades.json").read_text()
        ),
    )
    folder = run.native_folder(tmp_path, account)
    folder.mkdir(parents=True)
    run.bind(account, tmp_path, ROOT)
    run.native.write(folder / "ARTIFACT.json", artifact)
    run.native.write(folder / "CONTRACT.json", account.contract)
    standard.write_json(
        folder / "NATIVE_COMPLETE.json", {"artifact_sha256": io.sha(folder / "ARTIFACT.json")}
    )
    wrong = {"native_total_return": result.native_total_return + 0.01}
    standard.write_json(folder / "native_total_return.json", wrong)
    monkeypatch.setattr(
        run.native, "simulate", lambda *args: pytest.fail("saved validation cannot native")
    )
    with pytest.raises(ValueError, match="saved native total return"):
        run.verify_saved(account, prepared, tmp_path, ROOT)
    assert not (folder / "VERIFIED.json").exists()
    assert json.loads((folder / "native_total_return.json").read_text()) == wrong
    # Restore only this test's deliberately corrupt exported scalar. Revalidation
    # must use the saved completed account and bind its real outer E05 identity.
    (folder / "native_total_return.json").write_text(
        json.dumps({"native_total_return": result.native_total_return})
    )
    receipt = run.verify_saved(account, prepared, tmp_path, ROOT)
    assert receipt.experiment == account and receipt.native_path.startswith("E05/")
    assert receipt.artifact_sha256 == io.sha(folder / "ARTIFACT.json")
    assert not (tmp_path / "release_attempts.jsonl").exists()


def test_default_account_budget_scope_uses_same_new_journal_and_restores_it(tmp_path: Path) -> None:
    import c10standard.run as standard

    previous = standard.journal_event
    path = tmp_path / "shared.jsonl"
    with run.release_budget():
        standard.journal_event(path, "started", "C11_5bp_historical2022_2025", "default1", tmp_path)
    assert standard.journal_event is previous
    io.journal_event(path, "started", experiment().key, "experiment1", tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 and len({r["release_id"] for r in rows}) == 1


def test_completed_account_execution_resume_never_invokes_native_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import c10standard.run as standard
    from c10experiments.models import ExperimentReceipt
    from c10standard.models import Receipt

    account = experiment()
    folder = run.native_folder(tmp_path, account)
    folder.mkdir(parents=True)
    (folder / "ARTIFACT.json").write_text("completedaccount")
    (folder / "NATIVE_COMPLETE.json").write_text("complete")
    monkeypatch.setattr(
        standard,
        "validate_saved",
        lambda *args: Receipt(account=account.contract.key, hashes={}, max_differences={}),
    )
    monkeypatch.setattr(
        run,
        "verify_outer",
        lambda *args: ExperimentReceipt(
            experiment=account,
            native_path=account.variant,
            binding_sha256="saved",
            artifact_sha256="saved",
            standard_receipt_sha256="saved",
            hashes={},
        ),
    )
    monkeypatch.setattr(
        run.native, "simulate", lambda *args: pytest.fail("saved success native rerun")
    )
    run.execute(account, sample(), tmp_path, tmp_path / "journal.jsonl", ROOT)
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["revalidated"] and rows[0]["account"] == account.key
