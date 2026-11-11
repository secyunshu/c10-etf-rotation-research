"""Conditional QQQ account weights and immutable execution; no native brokers."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from c10experiments.io import prefix_market
from c10standard import native
from c10standard import run as standard
from c10standard.io import ROOT, load_strategy
from c10standard.models import Artifact, Contract, Prepared, Receipt
from rotbt.models import Dataset
from rotbt.phase2.models import FeatureStore
from scripts import qqq_overlay_run as run
from scripts import qqq_overlay_signals as signals

DAY = date(2025, 12, 31)


def market(scores: tuple[float, float, float, float, float] = (.5, .4, .3, .2, .25)) -> Prepared:
    days = pd.bdate_range(end=DAY, periods=300)
    datasets = []
    for symbol, score in zip(("QQQ", "SPY", "SOXX", "GLD", "SHY"), scores, strict=True):
        frame = pd.DataFrame(
            {"open": 100., "high": 101., "low": 99., "close": 100.,
             "volume": 1e6, "turnover": 20e6, "tr": [100.] * 299 + [100 * (1 + score)]},
            index=days,
        )
        datasets.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    prepared = Prepared(
        datasets=datasets, features=FeatureStore(frames={}), strategy=load_strategy(ROOT),
        risk_names=("QQQ", "SPY", "SOXX", "GLD"), first=days[0].date(), last=DAY,
    )
    return prefix_market(prepared, DAY)


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "simulate", Mock(side_effect=AssertionError("native forbidden")))


@pytest.mark.parametrize("scores,expected,cash", [
    ((.5, .4, .3, .2, .25), {"QQQ": .4, "SOXX": .4, "SHY": .15}, .05),
    ((.5, .4, .2, .1, .25), {"QQQ": .4, "SHY": .55}, .05),
    ((.2, .5, .4, .3, .25), {"SPY": .4, "SOXX": .4, "QQQ": .15, "SHY": .05}, 0),
    ((.2, .5, .1, .1, .25), {"SPY": .4, "QQQ": .15, "SHY": .45}, 0),
    ((.2, .1, .1, .1, .25), {"QQQ": .15, "SHY": .85}, 0),
])
def test_post_gate_qqq_branch_preserves_failed_slots_and_cash(scores, expected, cash) -> None:
    prepared = market(scores)
    for function in (signals.target, signals.oracle):
        result = function(prepared, DAY)
        assert result.q == pytest.approx(expected)
        assert sum(result.q.values()) + cash == pytest.approx(1)
    # SPY and overlay QQQ deliberately share the broad/style group.
    assert signals.target(prepared, DAY).q == pytest.approx(signals.oracle(prepared, DAY).q)


@pytest.mark.parametrize("field,value", [("close", 4.), ("volume", 0.), ("turnover", 1e6)])
def test_fixed_qqq_requires_trade_qualification_without_momentum_gate(field, value) -> None:
    prepared = market((-.5, .5, .4, .3, .25))
    qqq = next(d for d in prepared.datasets if d.symbol == "QQQ")
    qqq.frame[field] = value
    prepared = prefix_market(prepared, DAY)
    for function in (signals.target, signals.oracle):
        with pytest.raises(ValueError, match="QQQ"):
            function(prepared, DAY)


def test_qqq_missing_warmup_and_exact_close_fail_explicitly() -> None:
    for length in (252, 299):
        prepared = market()
        qqq = next(d for d in prepared.datasets if d.symbol == "QQQ")
        qqq.frame = qqq.frame.iloc[-252:] if length == 252 else qqq.frame.iloc[:-1]
        prepared = prefix_market(prepared, DAY)
        for function in (signals.target, signals.oracle):
            with pytest.raises(ValueError, match="QQQ"):
                function(prepared, DAY)


def test_missing_qqq_data_is_an_explicit_untradeable_failure() -> None:
    prepared = market()
    prepared.features.frames.pop("QQQ")
    prepared.datasets = [d for d in prepared.datasets if d.symbol != "QQQ"]
    for function in (signals.target, signals.oracle):
        with pytest.raises(ValueError, match="QQQ"):
            function(prepared, DAY)


def test_physical_prefix_and_raw_oracle_ignore_future_prices() -> None:
    prepared = market()
    day = date(2025, 10, 31)
    prefix = prefix_market(prepared, day)
    expected = signals.oracle(prefix, day)
    for data in prepared.datasets:
        data.frame.loc[data.frame.index > str(day), "tr"] *= 100
    for result in (signals.target(prepared, day), signals.target(prefix, day)):
        assert result.q == pytest.approx(expected.q)


@pytest.fixture
def locked(tmp_path: Path) -> Path:
    names = ["src/example.py", "tests/test_example.py", "typings/example.pyi", "pyproject.toml",
             "scripts/qqq_overlay_signals.py", "scripts/qqq_overlay_run.py",
             "standard/STRATEGY.json", "standard/ACCOUNTS.json", "standard/data/2025/input.csv",
             "experiments/qqq-overlay/PLAN.md", "experiments/qqq-overlay/PLAN_REVIEW.json",
             "experiments/qqq-overlay/tests/test_overlay.py"]
    names += [f"standard/reference/{rule}_{cost}bp_historical2022_2025/input.csv"
              for rule in ("C11", "SPY_BH", "QQQ_BH") for cost in (5, 20)]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixed\n")
    run.freeze(tmp_path)
    return tmp_path


def test_fixed_catalog_is_two_historical_bridges_and_two_science() -> None:
    assert [(t.variant, t.contract.cost) for t in run.catalog()] == [
        ("BASE", 5), ("BASE", 20), ("OVERLAY", 5), ("OVERLAY", 20)]
    for contract in (Contract(rule="C10"), Contract(cohort="reset2026")):
        with pytest.raises(ValueError, match="historical C11"):
            run.Trial(variant="OVERLAY", contract=contract)


def test_source_lock_covers_new_sources_but_never_2026_contents(locked: Path) -> None:
    initial = run.verify_lock(locked)
    path = locked / "standard/data/2026/PRIVATE.csv"
    path.parent.mkdir(parents=True)
    path.write_text("never read")
    assert run.verify_lock(locked) == initial
    (locked / "scripts/qqq_overlay_report.py").write_text("new source")
    with pytest.raises(ValueError, match="source lock"):
        run.verify_lock(locked)


def test_binding_rejects_unchanged_c11_shell_labelled_as_overlay(locked: Path) -> None:
    place = run.folder(locked / "out", run.catalog()[0])
    place.mkdir(parents=True)
    run.write_binding(run.catalog()[0], place, locked)
    with pytest.raises(ValueError, match="binding"):
        run.check_binding(run.catalog()[2], place, locked)


def test_hard_eight_budget_counts_failures_and_rejects_other_journals(locked: Path) -> None:
    trial, journal = run.catalog()[0], run.journal_path(locked)
    output = run.folder(locked / "out", trial)
    for index in range(8):
        run.journal_event(locked, journal, "started", trial, str(index), output)
        run.journal_event(locked, journal, "failed", trial, str(index), output, "failure")
    with pytest.raises(ValueError, match="8-native"):
        run.journal_event(locked, journal, "started", trial, "nine", output)
    with pytest.raises(ValueError, match="shared journal"):
        run.journal_event(locked, locked / "old.jsonl", "started", trial, "old", output)
    assert len(journal.read_text().splitlines()) == 16
    assert str(locked) not in journal.read_text()


def test_prior_family_journal_cannot_reset_or_consume_overlay_budget(locked: Path) -> None:
    journal = run.journal_path(locked)
    journal.parent.mkdir(parents=True)
    original = '{"standard_id":"C10_HISTORICAL_GROUP_CAP_V2","event":"started"}\n'
    journal.write_text(original)
    with pytest.raises(ValueError):
        run.journal_event(locked, journal, "started", run.catalog()[0], "new", locked / "out")
    assert journal.read_text() == original


def test_hooks_set_science_invested_once_and_restore_everything(locked: Path) -> None:
    old = (native.target, run.audit_module.oracle, standard.compare_reference,
           standard.journal_event, standard.settings_for)
    for trial in (run.catalog()[0], run.catalog()[2]):
        with pytest.raises(RuntimeError, match="abort"):
            with run.scoped_hooks(trial, locked):
                assert standard.settings_for(market().strategy, trial.contract).invested == (
                    .95 if trial.variant == "BASE" else 1)
                if trial.variant == "OVERLAY":
                    assert standard.compare_reference(Contract(), Mock(spec=Artifact), locked) == {}
                    assert native.target(Contract(), market(), DAY).q == pytest.approx(
                        {"QQQ": .4, "SOXX": .4, "SHY": .15})
                raise RuntimeError("abort")
        assert (native.target, run.audit_module.oracle, standard.compare_reference,
                standard.journal_event, standard.settings_for) == old


def test_science_requires_two_complete_bridges(locked: Path, monkeypatch) -> None:
    execute = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(standard, "execute", execute)
    with pytest.raises(ValueError, match="bridge"):
        run.execute(run.catalog()[2], market(), locked / "out", run.journal_path(locked), locked)
    execute.assert_not_called()


@pytest.mark.parametrize("corruption", ["partial", "large", "nan"])
def test_partial_or_failed_bridge_cannot_authorize_science(locked: Path, corruption) -> None:
    output = locked / "out"
    for trial in run.catalog()[:2]:
        place = run.folder(output, trial)
        place.mkdir(parents=True)
        run.write_binding(trial, place, locked)
        differences = dict.fromkeys(
            ["daily", "orders", "holdings", "dividends", "trades", "targets",
             "native_total_return", "metrics.json"], 0.)
        if corruption == "partial":
            differences.pop("orders")
        else:
            differences["daily"] = 1. if corruption == "large" else float("nan")
        native.write(place / "VERIFIED.json", Receipt(
            account=trial.contract.key, hashes={}, max_differences=differences))
    with pytest.raises(ValueError, match="bridge|valid number"):
        run.require_bridges(output, locked)


def test_success_resume_validates_without_native_or_new_attempt(locked: Path, monkeypatch) -> None:
    trial, output = run.catalog()[0], locked / "out"
    place = run.folder(output, trial)
    place.mkdir(parents=True)
    run.write_binding(trial, place, locked)
    for name in ("ARTIFACT.json", "NATIVE_COMPLETE.json", "VERIFIED.json"):
        (place / name).write_text("{}")
    receipt = Receipt(account=trial.contract.key, hashes={}, max_differences={"daily": 0.})
    validate = Mock(return_value=receipt)
    monkeypatch.setattr(standard, "validate_saved", validate)
    assert run.execute(trial, market(), output, run.journal_path(locked), locked) == receipt
    validate.assert_called_once()
    assert not run.journal_path(locked).exists()


def test_failed_attempt_is_preserved_and_new_output_cannot_repeat(
    locked: Path, monkeypatch,
) -> None:
    trial, output = run.catalog()[0], locked / "out"
    monkeypatch.setattr(
        standard, "load_accounts", Mock(return_value=[Mock(contract=trial.contract)]))
    monkeypatch.setattr(standard, "reference_folder", Mock())
    simulate = Mock(side_effect=RuntimeError("representative broker failure"))
    monkeypatch.setattr(native, "simulate", simulate)
    with pytest.raises(RuntimeError, match="broker failure"):
        run.execute(trial, market(), output, run.journal_path(locked), locked)
    place = run.folder(output, trial)
    failure = (place / "FAILURE.txt").read_text()
    with pytest.raises(ValueError, match="no automatic native retry"):
        run.execute(trial, market(), output, run.journal_path(locked), locked)
    with pytest.raises(ValueError, match="native evidence preserved"):
        run.execute(trial, market(), locked / "other", run.journal_path(locked), locked)
    assert (place / "FAILURE.txt").read_text() == failure
    assert simulate.call_count == 1
    assert [run.Attempt.model_validate_json(row).event
            for row in run.journal_path(locked).read_text().splitlines()] == ["started", "failed"]


def test_default_cli_only_verifies(monkeypatch) -> None:
    staged = Mock()
    monkeypatch.setattr(run, "run_stage", staged)
    monkeypatch.setattr("sys.argv", ["qqq_overlay_run"])
    run.main()
    assert staged.call_args.args[0] == "verify"
