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
    for update in ({"universe_refresh": "fixed"}, {"verified_trade_years": [2022, 2023, 2024, 2025, 2026]},
                   {"annual_2026_verified": True}, {"top_k": 1}):
        with pytest.raises(ValueError):
            Policy.model_validate({**base, **update})


def test_release_journal_counts_failures_and_refuses_resetting_previous_family(tmp_path: Path) -> None:
    for n in range(8):
        journal_event(tmp_path, "started", f"account{n}", str(n), tmp_path / "runs")
        journal_event(tmp_path, "failed", f"account{n}", str(n), tmp_path / "runs", "fixture")
    with pytest.raises(ValueError, match="8"):
        journal_event(tmp_path, "started", "extra", "extra", tmp_path / "runs")


def test_budget_scope_restores_original_functions_on_failure(tmp_path: Path) -> None:
    old = (annual_run.journal_path, annual_run.journal_event)
    with pytest.raises(RuntimeError):
        with budget_scope(tmp_path):
            assert annual_run.journal_path(tmp_path) != old[0](tmp_path)
            raise RuntimeError("fixture")
    assert (annual_run.journal_path, annual_run.journal_event) == old


def test_default_call_verifies_references_without_starting_native(tmp_path: Path, monkeypatch) -> None:
    import scripts.current_standard as current
    forbidden = Mock(side_effect=AssertionError("native forbidden"))
    monkeypatch.setattr(annual_run, "run_stage", forbidden)
    monkeypatch.setattr(current, "verify_release", lambda root: None)
    monkeypatch.setattr(current, "build_report", Mock())
    run_current(ROOT, tmp_path, execute=False)
    forbidden.assert_not_called()
