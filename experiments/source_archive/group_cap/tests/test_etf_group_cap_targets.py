"""Pure frozen-catalog and allocation checks; no Cerebro or API calls."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from rotbt.models import Dataset
from rotbt.phase2.models import FeatureStore
from scripts.etf_group_cap_audit import raw_oracle
from scripts.etf_group_cap_targets import GroupMap, Role, decision, load_roles


def role(top: int = 3, capped: bool = True) -> Role:
    return Role(
        key=f"legacy36_{'cap1' if capped else 'uncapped'}_K{top}_5bp_reset2026",
        pool="legacy36",
        cap_one_per_group=capped,
        top_k=top,
        cost_bps=5,
        cohort="reset2026",
        mode="new_native",
        alias_of=None,
    )


def source() -> tuple[list[Dataset], FeatureStore, GroupMap]:
    days = pd.bdate_range(end="2025-12-31", periods=300)
    datasets = []
    for name, score in [("A", 0.5), ("B", 0.4), ("C", 0.3), ("D", 0.2), ("SHY", 0.25)]:
        tr = [100.0] * len(days)
        tr[-1] = 100 * (1 + score)
        frame = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000.0,
                "turnover": 20_000_000.0,
                "tr": tr,
            },
            index=days,
        )
        datasets.append(Dataset(symbol=name, frame=frame, dividends=[]))
    features = FeatureStore(
        frames={
            d.symbol: d.frame.assign(
                mom252=d.frame.tr.pct_change(252),
                raw_close=d.frame.close,
                turnover20=d.frame.turnover.rolling(20).mean(),
                positive20=d.frame.volume.gt(0).rolling(20).sum(),
            )
            for d in datasets
        }
    )
    groups = GroupMap(groups={"broad": ("A", "B"), "other": ("C",), "third": ("D",)})
    return datasets, features, groups


def test_group_cap_preserves_global_ranking_without_buying_duplicate_risk_group() -> None:
    _, features, groups = source()
    result = decision(role(), features, ("A", "B", "C", "D"), groups, date(2025, 12, 31))
    assert result.q == pytest.approx({"A": 0.3, "C": 0.3, "SHY": 0.4})


def test_a_failed_selected_slot_stays_defensive_instead_of_increasing_other_positions() -> None:
    _, features, groups = source()
    result = decision(role(2), features, ("C", "D"), groups, date(2025, 12, 31))
    assert result.q == pytest.approx({"C": 0.45, "SHY": 0.55})


def test_unconstrained_control_and_top_one_identity_do_not_change_existing_hypothesis() -> None:
    _, features, groups = source()
    names = ("A", "B", "C", "D")
    assert decision(role(2, False), features, names, groups, date(2025, 12, 31)).q == pytest.approx(
        {"A": 0.45, "B": 0.45, "SHY": 0.1}
    )
    assert (
        decision(role(1), features, names, groups, date(2025, 12, 31)).q
        == decision(role(1, False), features, names, groups, date(2025, 12, 31)).q
    )


def test_raw_dataset_oracle_agrees_without_using_precomputed_feature_signals() -> None:
    datasets, features, groups = source()
    expected = decision(role(), features, ("A", "B", "C", "D"), groups, date(2025, 12, 31))
    for frame in features.frames.values():
        frame.loc[:, "mom252"] = -999.0
    actual = raw_oracle(role(), datasets, ("A", "B", "C", "D"), groups, expected.day)
    assert actual.q == expected.q


def test_future_observations_cannot_change_frozen_prior_close_target() -> None:
    _, features, groups = source()
    day = date(2025, 12, 31)
    expected = decision(role(), features, ("A", "B", "C", "D"), groups, day)
    for frame in features.frames.values():
        frame.loc[pd.Timestamp("2026-01-30")] = 999999.0
    assert decision(role(), features, ("A", "B", "C", "D"), groups, day) == expected


def test_missing_group_and_invalid_role_identity_fail_instead_of_silently_changing_pool() -> None:
    _, features, groups = source()
    with pytest.raises(ValueError, match="group"):
        decision(role(), features, ("UNKNOWN",), groups, date(2025, 12, 31))
    with pytest.raises(ValidationError):
        role().model_copy(update={"key": "wrong"}).__class__.model_validate(
            {**role().model_dump(), "key": "wrong"}
        )


def test_frozen_catalog_counts_reuse_and_aliases_are_not_extra_native_experiments() -> None:
    roles = load_roles(Path(__file__).resolve().parents[1])
    assert len(roles) == 72
    assert sum(r.mode == "new_native" for r in roles) == 42
    assert sum(r.mode == "reuse" for r in roles) == 18
    assert sum(r.mode == "alias" for r in roles) == 12
    assert max(r.through for r in roles) == date(2026, 11, 6)
