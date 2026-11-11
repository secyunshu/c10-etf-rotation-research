"""A source snapshot must be complete and immutable before fresh-start comparisons."""

from datetime import date
from pathlib import Path

import pytest

from c10standard.io import ROOT, load_market
from scripts.v2_2023_data import validate_dataset
from scripts.v2_2023_io import snapshot


def test_missing_day_or_future_price_cannot_enter_the_new_pool() -> None:
    data = load_market(ROOT, "2025").datasets[0]
    validate_dataset(data)
    broken = data.model_copy(update={"frame": data.frame.iloc[1:].copy()})
    with pytest.raises(ValueError):
        validate_dataset(broken)
    broken = data.model_copy(update={"frame": data.frame.copy()})
    broken.frame.loc["2026-01-02"] = broken.frame.iloc[-1]
    with pytest.raises(ValueError):
        validate_dataset(broken)


def test_cash_payment_before_ex_date_is_rejected() -> None:
    data = next(d for d in load_market(ROOT, "2025").datasets if d.dividends)
    first = data.dividends[0].model_copy(update={"pay_date": date(2020, 1, 1)})
    with pytest.raises(ValueError):
        validate_dataset(data.model_copy(update={"dividends": [first, *data.dividends[1:]]}))


def test_lock_collection_never_includes_2026_market_payloads(tmp_path: Path) -> None:
    for name in (
        "pyproject.toml",
        "standard/STRATEGY.json",
        "standard/ACCOUNTS.json",
        "experiments/v2-2023-start/PLAN.md",
        "experiments/v2-2023-start/PLAN_REVIEW.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    p = tmp_path / "standard/data/2026/SHY.csv.gz"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"must not be read")
    assert all("/2026/" not in n for n in snapshot(tmp_path).hashes)
