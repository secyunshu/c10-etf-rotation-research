"""The new experiment budget and data scope cannot silently inherit later payloads."""

from pathlib import Path

import pytest

from scripts.falsify_io import freeze, snapshot, verify_lock


def test_source_lock_rejects_new_input_change_without_reading_2026(tmp_path: Path) -> None:
    for name in (
        "pyproject.toml",
        "standard/STRATEGY.json",
        "standard/ACCOUNTS.json",
        "experiments/annual-pool-falsification/PLAN.md",
        "experiments/annual-pool-falsification/PLAN_REVIEW.json",
    ):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}")
    later = tmp_path / "standard/data/2026/QQQ.csv.gz"
    later.parent.mkdir(parents=True)
    later.write_bytes(b"not used")
    assert all("/2026/" not in p for p in snapshot(tmp_path).hashes)
    freeze(tmp_path)
    (tmp_path / "pyproject.toml").write_text("changed")
    with pytest.raises(ValueError, match="lock"):
        verify_lock(tmp_path)
