from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from c10standard.io import ROOT
from scripts.current_standard import load_policy, select_target
from scripts.ytd_annual_data import load_prepared


def test_current_pointer_loads_ungated_conditional_sleeve() -> None:
    policy = load_policy(ROOT)
    assert policy.policy_version == "2.3.0"
    assert policy.shy_momentum_gate is False
    assert policy.risk_slot_weight == .45 and policy.sleeve_weight == .10


def test_published_policy_rejects_changed_weights(tmp_path: Path) -> None:
    folder = tmp_path / "standard/v2_3"
    folder.mkdir(parents=True)
    (tmp_path / "standard/CURRENT.json").write_bytes((ROOT / "standard/CURRENT.json").read_bytes())
    strategy = (ROOT / "standard/v2_3/STRATEGY.json").read_text()
    (folder / "STRATEGY.json").write_text(strategy.replace('"risk_slot_weight": 0.45',
                                                             '"risk_slot_weight": 0.40'))
    with pytest.raises(ValueError, match="hash"):
        load_policy(tmp_path)


@pytest.mark.parametrize(("first", "sleeve"), [("QQQ", "SHY"), ("XLE", "QQQ")])
def test_2026_selector_ignores_shy_momentum_and_applies_both_sleeves(
    first: str, sleeve: str
) -> None:
    market = load_prepared("ANNUAL", "science")
    day = date(2026, 10, 30)
    for symbol, frame in market.features.frames.items():
        frame.loc[pd.Timestamp(day), "mom252"] = {first: 1., "SMH": .5, "SHY": 100.}.get(
            symbol, -.2
        )
    target = select_target(market.schedule.entries[-1], market.features, day)
    assert target.q == {first: .45, "SMH": .45, sleeve: .10}
