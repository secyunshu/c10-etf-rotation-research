"""Latest-complete scope and source locks must fail explicitly on future input."""

from pathlib import Path

import pytest

from scripts.ytd_annual_io import freeze, require_acceptance, snapshot, verify_lock


def test_new_release_lock_detects_market_change(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.ytd_annual_io.require_acceptance", lambda root: None)
    for name in (
        "pyproject.toml",
        "standard/STRATEGY.json",
        "standard/ACCOUNTS.json",
        "experiments/annual-2026/PLAN.md",
        "experiments/annual-2026/PLAN_REVIEW.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    market = tmp_path / "experiments/annual-2026/data/science/SHY.csv.gz"
    market.parent.mkdir(parents=True)
    market.write_bytes(b"historical")
    assert market.relative_to(tmp_path).as_posix() in snapshot(tmp_path).hashes
    freeze(tmp_path)
    market.write_bytes(b"future_or_changed")
    with pytest.raises(ValueError, match="lock"):
        verify_lock(tmp_path)


def test_missing_or_failed_data_acceptance_blocks_native_source_freeze(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="acceptance"):
        require_acceptance(tmp_path)
    folder = tmp_path / "experiments/annual-2026"
    folder.mkdir(parents=True)
    (folder / "DATA_ACCEPTANCE.json").write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError):
        require_acceptance(tmp_path)
