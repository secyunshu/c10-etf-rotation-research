"""Report identity and capital tests; no native account or market API is used."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.etf_13612w import catalog
from scripts.etf_13612w_report import (
    Binding,
    make_period,
    paired_comparisons,
    planned_bindings,
    validate_registry,
)


def binding(
    score: str = "calendar_13612w", k: int = 1, cost: int = 5, cohort: str = "continuous2022_2026"
) -> Binding:
    rule = {1: "C10", 2: "C11", 3: "C00"}[k]
    key = (
        f"W13612_K{k}_{cost}bp_{cohort}"
        if score == "calendar_13612w"
        else f"{rule}_{cost}bp_{cohort}"
    )
    return Binding(
        key=key,
        score=score,
        rule=rule,
        k=k,
        cost_bps=cost,
        cohort=cohort,
        path=Path("/tmp") / key,
        mode="new_native" if score == "calendar_13612w" else "reuse",
        source_hashes={},
    )


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.DataFrame(
        {"equity": [150_000.0, 135_000.0, 165_000.0]},
        index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"]),
    )
    orders = pd.DataFrame(
        {
            "day": ["2025-12-31", "2026-01-02"],
            "size": [100, 200],
            "price": [100.0, 110.0],
            "fee": [5.0, 3.0],
        }
    )
    return daily, orders


def test_2026_continuous_uses_own_december_nav_and_has_no_annualized_return() -> None:
    daily, orders = frames()
    result = make_period(binding(), daily, orders, date(2026, 1, 2), date(2026, 1, 5), "2026YTD")
    assert result.opening == 150_000
    assert result.total_return == pytest.approx(0.1)
    assert result.max_drawdown == pytest.approx(0.1)
    assert result.cagr_252 is None
    assert result.fees == 3.0 and result.orders == 1
    with pytest.raises(ValueError, match="December 31"):
        make_period(
            binding(), daily.iloc[1:], orders, date(2026, 1, 2), date(2026, 1, 5), "2026YTD"
        )


def test_reset_account_starts_from_registered_cash_and_keeps_first_session_loss() -> None:
    daily, orders = frames()
    daily = daily.iloc[1:].copy()
    daily["equity"] = [45_000.0, 55_000.0]
    result = make_period(
        binding(cohort="reset2026"), daily, orders, date(2026, 1, 2), date(2026, 1, 5), "2026YTD"
    )
    assert result.opening == 50_000
    assert result.total_return == pytest.approx(0.1)
    assert result.max_drawdown == pytest.approx(0.1)


def test_bridges_are_excluded_and_registry_relabeling_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    trials = catalog(root)
    roles = planned_bindings(root, trials, [])
    assert len(roles) == 18
    assert all(row.score == "calendar_13612w" and not row.key.startswith("bridge") for row in roles)
    validate_registry(roles, roles)
    mislabeled = [roles[0].model_copy(update={"k": 2}), *roles[1:]]
    with pytest.raises(ValueError, match="registry"):
        validate_registry(mislabeled, roles)
    with pytest.raises(ValueError, match="registry"):
        validate_registry([*roles, roles[0]], roles)


def test_pairing_requires_same_k_cost_cohort_and_period() -> None:
    daily, orders = frames()
    new = make_period(
        binding(k=2, cost=20), daily, orders, date(2026, 1, 2), date(2026, 1, 5), "2026YTD"
    )
    old = make_period(
        binding("original252", k=2, cost=20),
        daily,
        orders,
        date(2026, 1, 2),
        date(2026, 1, 5),
        "2026YTD",
    )
    pair = paired_comparisons([old, new])[0]
    assert pair.new_key == new.key and pair.original_key == old.key
    assert pair.return_difference == 0 and pair.drawdown_difference == 0
    for altered in (
        old.model_copy(update={"k": 1}),
        old.model_copy(update={"cost_bps": 5}),
        old.model_copy(update={"cohort": "reset2026"}),
        old.model_copy(update={"period": "full2022_2025"}),
    ):
        with pytest.raises(ValueError, match="unique matched"):
            paired_comparisons([new, altered])


def test_empty_report_window_fails_instead_of_producing_zero_return() -> None:
    daily, orders = frames()
    with pytest.raises(ValueError, match="empty"):
        make_period(binding(), daily, orders, date(2027, 1, 1), date(2027, 1, 5), "missing")
