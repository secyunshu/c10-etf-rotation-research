"""Annual source locks must exclude later economic periods and detect tampering."""

from pathlib import Path

import pytest

from scripts.annual_io import freeze, snapshot, verify_lock


def test_source_snapshot_has_no_2026_market_or_reference_payload(tmp_path: Path) -> None:
    for name in (
        "pyproject.toml",
        "standard/STRATEGY.json",
        "standard/ACCOUNTS.json",
        "experiments/v2-annual-repool/PLAN.md",
        "experiments/v2-annual-repool/PLAN_REVIEW.json",
    ):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    later = tmp_path / "standard/data/2026/QQQ.csv.gz"
    later.parent.mkdir(parents=True)
    later.write_bytes(b"futurepayload")
    assert all("/2026/" not in name for name in snapshot(tmp_path).hashes)
    freeze(tmp_path)
    (tmp_path / "standard/STRATEGY.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="lock"):
        verify_lock(tmp_path)
