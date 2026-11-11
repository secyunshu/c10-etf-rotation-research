"""Path experiment orchestration contracts; all native calls are mocked."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from test_standard_v2 import prepared

from c10paths import io, run
from c10standard import native
from c10standard import run as standard
from c10standard.models import Artifact, Contract, Receipt


@pytest.fixture
def locked(tmp_path: Path) -> Path:
    for name in (
        "src/example.py", "tests/test_example.py", "pyproject.toml",
        "standard/STRATEGY.json", "standard/ACCOUNTS.json",
        "standard/data/input.csv", "standard/reference/input.csv",
        "experiments/wechat-momentum-paths/PLAN.md",
        "experiments/wechat-momentum-paths/PLAN_REVIEW.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixed\n")
    io.freeze(tmp_path)
    return tmp_path


def test_fixed_catalog_is_eighteen_science_accounts_and_two_bridges() -> None:
    trials = io.catalog()
    assert len(trials) == len({trial.key for trial in trials}) == 20
    assert [(t.contract.cost, t.contract.cohort) for t in trials if t.variant == "BASE"] == [
        (5, "historical2022_2025"), (20, "continuous2022_2026")
    ]
    assert all(t.contract.rule == "C11" for t in trials)
    with pytest.raises(ValueError, match="C11"):
        io.Trial(variant="P01", contract=Contract(rule="C10"))


def test_source_change_or_new_source_cannot_silently_reuse_lock(locked: Path) -> None:
    initial = io.verify_lock(locked)
    path = locked / "src/example.py"
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="source lock"):
        io.verify_lock(locked)
    path.write_text("fixed\n")
    assert io.verify_lock(locked) == initial
    (locked / "src/new.py").write_text("new\n")
    with pytest.raises(ValueError, match="source lock"):
        io.verify_lock(locked)


def test_binding_prevents_cross_variant_or_source_mislabelling(locked: Path) -> None:
    trial = io.Trial(variant="P01", contract=Contract())
    output = locked / "results"
    place = io.folder(output, trial)
    place.mkdir(parents=True)
    io.write_binding(trial, place, locked)
    io.check_binding(trial, place, locked)
    with pytest.raises(ValueError, match="binding"):
        io.check_binding(io.Trial(variant="P02", contract=Contract()), place, locked)


def test_independent_budget_counts_failed_native_and_redacts_output(locked: Path) -> None:
    journal = io.journal_path(locked)
    trial = io.catalog()[0]
    output = io.folder(locked / "results", trial)
    for index in range(28):
        run.journal_event(locked, journal, "started", trial, str(index), output)
        run.journal_event(locked, journal, "failed", trial, str(index), output, "native failed")
    with pytest.raises(ValueError, match="28"):
        run.journal_event(locked, journal, "started", trial, "overflow", output)
    assert str(locked) not in journal.read_text()
    assert len(journal.read_text().splitlines()) == 56
    with pytest.raises(ValueError, match="shared journal"):
        run.journal_event(locked, locked / "other.jsonl", "started", trial, "reset", output)


def test_hooks_restore_after_failure_and_science_has_no_fake_reference(
    locked: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = io.Trial(variant="P01", contract=Contract())
    old_target, old_oracle = native.target, run.audit_module.oracle
    old_compare, old_journal = standard.compare_reference, standard.journal_event
    with pytest.raises(RuntimeError, match="abort"):
        with run.scoped_hooks(trial, locked):
            assert native.target is not old_target
            assert run.audit_module.oracle is not old_oracle
            assert standard.compare_reference(Contract(), Mock(spec=Artifact), locked, locked) == {}
            raise RuntimeError("abort")
    assert native.target is old_target and run.audit_module.oracle is old_oracle
    assert standard.compare_reference is old_compare and standard.journal_event is old_journal


def test_science_requires_both_verified_bridges_before_standard_execution(
    locked: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute = Mock(side_effect=AssertionError("native must remain uncalled"))
    monkeypatch.setattr(standard, "execute", execute)
    trial = io.Trial(variant="P01", contract=Contract())
    with pytest.raises(ValueError, match="bridge"):
        run.execute(trial, prepared(), locked / "results", io.journal_path(locked), locked)
    execute.assert_not_called()


def test_successful_resume_revalidates_without_native_or_budget(
    locked: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = io.catalog()[0]
    output = locked / "results"
    place = io.folder(output, trial)
    place.mkdir(parents=True)
    io.write_binding(trial, place, locked)
    (place / "ARTIFACT.json").write_text("{}")
    (place / "NATIVE_COMPLETE.json").write_text("{}")
    (place / "VERIFIED.json").write_text("{}")
    receipt = Receipt(account=trial.contract.key, hashes={}, max_differences={"daily": 0.0})
    validate = Mock(return_value=receipt)
    simulate = Mock(side_effect=AssertionError("native rerun forbidden"))
    monkeypatch.setattr(standard, "validate_saved", validate)
    monkeypatch.setattr(native, "simulate", simulate)
    assert run.execute(trial, prepared(), output, io.journal_path(locked), locked) == receipt
    validate.assert_called_once()
    simulate.assert_not_called()
    assert not io.journal_path(locked).exists()


def test_incomplete_native_failure_is_preserved_without_retry(
    locked: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = io.catalog()[0]
    place = io.folder(locked / "results", trial)
    place.mkdir(parents=True)
    io.write_binding(trial, place, locked)
    (place / "FAILURE.txt").write_text("original failure")
    simulate = Mock(side_effect=AssertionError("native rerun forbidden"))
    monkeypatch.setattr(native, "simulate", simulate)
    with pytest.raises(ValueError, match="no automatic native retry"):
        run.execute(trial, prepared(), locked / "results", io.journal_path(locked), locked)
    simulate.assert_not_called()
    assert (place / "FAILURE.txt").read_text() == "original failure"


def test_cli_defaults_to_verification_without_native(monkeypatch: pytest.MonkeyPatch) -> None:
    staged = Mock()
    monkeypatch.setattr(run, "run_stage", staged)
    monkeypatch.setattr("sys.argv", ["c10paths.run"])
    run.main()
    assert staged.call_args.args[0] == "verify"


@pytest.mark.parametrize("corruption", ["partial", "large", "nan"])
def test_science_cannot_accept_partial_or_failed_bridge_comparisons(
    locked: Path, corruption: str
) -> None:
    output = locked / "results"
    for trial in io.catalog()[:2]:
        place = io.folder(output, trial)
        place.mkdir(parents=True)
        io.write_binding(trial, place, locked)
        differences = dict.fromkeys(
            ["daily", "orders", "holdings", "dividends", "trades", "targets",
             "native_total_return", "metrics.json" if trial.contract.terminal else "ytd.json"],
            0.0,
        )
        if corruption == "partial":
            differences.pop("orders")
        else:
            differences["daily"] = 1.0 if corruption == "large" else float("nan")
        receipt = Receipt(
            account=trial.contract.key, hashes={}, max_differences=differences,
        )
        native.write(place / "VERIFIED.json", receipt)
    with pytest.raises(ValueError, match="bridge|valid number"):
        run.require_bridges(output, locked)


def test_actual_executor_failure_counts_once_and_cannot_escape_in_another_output(
    locked: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = io.catalog()[0]
    journal = io.journal_path(locked)
    output = locked / "results"
    simulate = Mock(side_effect=RuntimeError("representative native broker failure"))
    monkeypatch.setattr(
        standard, "load_accounts", Mock(return_value=[Mock(contract=trial.contract)])
    )
    monkeypatch.setattr(standard, "reference_folder", Mock())
    monkeypatch.setattr(native, "simulate", simulate)
    with pytest.raises(RuntimeError, match="broker failure"):
        run.execute(trial, prepared(), output, journal, locked)
    records = [run.Attempt.model_validate_json(line) for line in journal.read_text().splitlines()]
    assert [r.event for r in records] == ["started", "failed"]
    assert all(r.trial == trial for r in records)
    assert (io.folder(output, trial) / "FAILURE.txt").is_file()
    with pytest.raises(ValueError, match="failed native evidence"):
        run.execute(trial, prepared(), locked / "new-output", journal, locked)
    assert simulate.call_count == 1


def test_successful_trial_in_another_output_reuses_original_evidence_without_native(
    locked: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = io.catalog()[0]
    journal = io.journal_path(locked)
    previous = io.folder(locked / "original-results", trial)
    for event in ("started", "native_completed", "finished"):
        run.journal_event(locked, journal, event, trial, "completed-attempt", previous)
    receipt = Receipt(account=trial.contract.key, hashes={}, max_differences={"daily": 0.0})
    execute = Mock(return_value=receipt)
    monkeypatch.setattr(standard, "execute", execute)
    with pytest.raises(ValueError, match="original-results/BASE/C11_5bp_historical2022_2025"):
        run.execute(trial, prepared(), locked / "new-output", journal, locked)
    execute.assert_not_called()
    assert len(journal.read_text().splitlines()) == 3
