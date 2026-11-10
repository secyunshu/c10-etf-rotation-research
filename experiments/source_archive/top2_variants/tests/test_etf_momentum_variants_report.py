"""Pure report contracts; no native account, unseen result, or full bootstrap run."""

from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from rotbt.models import BootstrapResult
from scripts.etf_momentum_variants_report import (
    FAMILY,
    ExcessPanel,
    Metric,
    Role,
    annual_support,
    paired_rows,
    pareto_rows,
    period_row,
    statistics_rows,
    target_hash,
    validate_panel,
)


def role(variant: str = "E01", cost: int = 5) -> Role:
    return Role(
        key=f"{variant}_K2_{cost}bp_continuous2022_2026",
        variant=variant,
        rule="C11",
        k=2,
        cost_bps=cost,
        cohort="continuous2022_2026",
        path=Path("/tmp") / variant,
        mode="new_native",
        source_hashes={},
    )


def metric(variant: str = "E01", cost: int = 5) -> Metric:
    daily = pd.DataFrame(
        {"equity": [150_000.0, 135_000.0, 165_000.0]},
        index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"]),
    )
    orders = pd.DataFrame({"day": ["2026-01-02"], "fee": [3.0], "size": [100], "price": [100.0]})
    return period_row(
        role(variant, cost), daily, orders, date(2026, 1, 2), date(2026, 1, 5), "2026YTD"
    )


def test_continuous_ytd_keeps_prior_nav_first_day_loss_and_blank_cagr() -> None:
    row = metric()
    assert row.opening == 150_000 and row.total_return == pytest.approx(0.1)
    assert row.max_drawdown == pytest.approx(0.1) and row.cagr_252 is None


def test_pairs_require_same_period_cost_and_cohort_for_each_named_baseline() -> None:
    new, base, weighted = metric(), metric("BASE252"), metric("W13612")
    pairs = paired_rows([new, base, weighted])
    assert {row.baseline for row in pairs} == {"BASE252", "W13612"}
    assert all(row.return_difference == 0 for row in pairs)
    with pytest.raises(ValueError, match="unique matched"):
        paired_rows([new, base.model_copy(update={"cost_bps": 20}), weighted])
    with pytest.raises(ValueError, match="unique matched"):
        paired_rows([new, base, weighted.model_copy(update={"period": "full2022_2025"})])


def test_bootstrap_requires_all_six_columns_and_exact_historical_sessions() -> None:
    days = tuple(
        x.date() for x in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2025-12-31")
    )
    panel = ExcessPanel(names=FAMILY, days=days, differences=np.zeros((1003, 6)))
    validate_panel(panel)  # Identical zero paths remain in the six-column family.
    with pytest.raises(ValueError, match="six"):
        validate_panel(
            panel.model_copy(update={"names": FAMILY[:-1], "differences": np.zeros((1003, 5))})
        )
    with pytest.raises(ValueError, match="historical"):
        validate_panel(panel.model_copy(update={"days": days[:-1] + (date(2026, 1, 2),)}))
    with pytest.raises(ValueError, match="historical"):
        validate_panel(panel.model_copy(update={"days": days[1:]}))


def test_pointwise_ci_is_arithmetic_annualization_and_support_uses_each_variant_p() -> None:
    result = BootstrapResult(
        names=FAMILY,
        block=5,
        family_p=0.001,
        adjusted_p=[0.01, 0.2, 1, 1, 1, 1],
        mean_bp=[1.0] * 6,
        ci_low_bp=[-1.0] * 6,
        ci_high_bp=[3.0] * 6,
    )
    rows = statistics_rows(result, 5)
    assert rows[0].annualized_arithmetic_excess == pytest.approx(0.0252)
    assert rows[0].pointwise_ci_low == pytest.approx(-0.0252)
    complete = [
        row.model_copy(update={"cost_bps": cost, "block": block})
        for cost in (5, 20)
        for block in (5, 20, 60)
        for row in rows
    ]
    assert annual_support(complete, "E01")
    assert not annual_support(complete, "E02")
    assert not annual_support(complete[:-6], "E01")


def test_economic_target_hash_ignores_reason_and_mapping_order_but_keeps_weights() -> None:
    first = '[{"day":"2025-01-31","q":{"QQQ":0.45,"SHY":0.55},"reason":"new"}]'
    same = '[{"reason":"old","q":{"SHY":0.55,"QQQ":0.45},"day":"2025-01-31"}]'
    changed = '[{"day":"2025-01-31","q":{"QQQ":0.44,"SHY":0.56},"reason":"new"}]'
    assert target_hash(first) == target_hash(same)
    assert target_hash(first) != target_hash(changed)


def test_pareto_requires_both_costs_and_ignores_roundoff_as_improvement() -> None:
    rows = []
    for variant in (*FAMILY, "BASE252"):
        for cost in (5, 20):
            for period in ("2022", "2023", "2024", "2025", "full2022_2025"):
                cagr = 0.2 + (0.01 if variant == "E01" and cost == 5 else 1e-12)
                rows.append(
                    metric(variant, cost).model_copy(
                        update={
                            "cohort": "historical2022_2025",
                            "period": period,
                            "cagr_252": cagr,
                            "max_drawdown": 0.1,
                        }
                    )
                )
    result = pareto_rows(rows, [])
    assert next(r for r in result if r.variant == "E01" and r.cost_bps == 5).pareto_at_cost
    assert not any(r.pareto_both_costs for r in result)
    assert not any(r.pareto_at_cost for r in result if r.variant != "E01")
