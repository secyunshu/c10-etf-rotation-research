"""The current default is the authorised annual Top2 policy, not a relabelled fixed pool."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from c10standard.io import ROOT
from scripts import annual_run
from scripts.current_standard import Policy, budget_scope, journal_event, load_policy, run_current


def test_default_policy_resolves_real_annual_schedule_and_preserves_old_fixed_version() -> None:
    policy = load_policy(ROOT)
    assert policy.top_k == 2 and policy.universe_refresh == "annual"
    assert policy.schedule_path == "experiments/v2-annual-repool/ANNUAL.json"
    assert not policy.annual_2026_verified
    assert json.loads((ROOT / "standard/STRATEGY.json").read_text())["risk_names"]


def test_current_pointer_refuses_a_changed_config(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "standard/current", tmp_path / "standard/current")
    shutil.copyfile(ROOT / "standard/CURRENT.json", tmp_path / "standard/CURRENT.json")
    path = tmp_path / "standard/current/STRATEGY.json"
    path.write_text(path.read_text().replace('"top_k": 2', '"top_k": 1'))
    with pytest.raises(ValueError, match="hash"):
        load_policy(tmp_path)


def test_policy_refuses_fixed_pool_or_silent_2026_extension() -> None:
    base = json.loads((ROOT / "standard/current/STRATEGY.json").read_text())
    for update in (
        {"universe_refresh": "fixed"},
        {"verified_trade_years": [2022, 2023, 2024, 2025, 2026]},
        {"annual_2026_verified": True},
        {"top_k": 1},
    ):
        with pytest.raises(ValueError):
            Policy.model_validate({**base, **update})


def test_release_journal_counts_failures_and_refuses_resetting_previous_family(
    tmp_path: Path,
) -> None:
    for n in range(8):
        journal_event(tmp_path, "started", f"account{n}", str(n), tmp_path / "runs")
        journal_event(tmp_path, "failed", f"account{n}", str(n), tmp_path / "runs", "fixture")
    with pytest.raises(ValueError, match="8"):
        journal_event(tmp_path, "started", "extra", "extra", tmp_path / "runs")


def test_old_journal_cannot_be_adopted_as_current_release_budget(tmp_path: Path) -> None:
    import scripts.current_standard as current

    path = current.journal_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"experiment":"V2_ANNUAL_REPOOL_V1","event":"started"}\n')
    with pytest.raises(ValueError):
        journal_event(tmp_path, "started", "new", "new", tmp_path / "runs")


def test_explicit_execution_runs_annual_science_after_bridges_then_full_reference_check(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.current_standard as current

    monkeypatch.setattr(current, "verify_release", lambda root: None)
    calls = []
    monkeypatch.setattr(annual_run, "run_stage", lambda stage, *args: calls.append(stage))
    monkeypatch.setattr(current, "compare_reference_books", lambda *args: calls.append("compare"))
    monkeypatch.setattr(current, "build_report", lambda *args: calls.append("report"))
    current.run_current(ROOT, tmp_path, execute=True)
    assert calls == ["bridge", "science", "compare", "report"]


def test_budget_scope_restores_original_functions_on_failure(tmp_path: Path) -> None:
    old = (annual_run.journal_path, annual_run.journal_event)
    with pytest.raises(RuntimeError):
        with budget_scope(tmp_path):
            assert annual_run.journal_path(tmp_path) != old[0](tmp_path)
            raise RuntimeError("fixture")
    assert (annual_run.journal_path, annual_run.journal_event) == old


def test_default_call_verifies_references_without_starting_native(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.current_standard as current

    forbidden = Mock(side_effect=AssertionError("native forbidden"))
    monkeypatch.setattr(annual_run, "run_stage", forbidden)
    monkeypatch.setattr(current, "verify_release", lambda root: None)
    monkeypatch.setattr(current, "build_report", Mock())
    run_current(ROOT, tmp_path, execute=False)
    forbidden.assert_not_called()


@pytest.mark.parametrize("change", ["modified", "added", "deleted"])
def test_release_lock_pins_annual_reference_books_before_native(
    tmp_path: Path, monkeypatch, change
) -> None:
    import shutil

    import scripts.current_standard as current

    reference = "experiments/v2-annual-repool/reference/ANNUAL_5bp_science/daily.csv.gz"
    frozen = current.release_snapshot(ROOT)
    assert reference in frozen.hashes
    for name in frozen.hashes:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (tmp_path / "standard/current/RELEASE_LOCK.json").write_text(frozen.model_dump_json())
    monkeypatch.setattr(current, "verify_annual_lock", lambda root: None)
    current.verify_release(tmp_path)
    path = tmp_path / reference
    if change == "modified":
        path.write_bytes(path.read_bytes() + b"changed")
    elif change == "added":
        path.with_name("EXTRA.json").write_text("{}")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="lock"):
        current.verify_release(tmp_path)
