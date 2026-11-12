from datetime import date

import pytest

from scripts import annual_signals
from scripts.annual_data import load_prepared
from scripts.annual_models import Contract
from scripts.conditional_sleeve import allocation, select


@pytest.mark.parametrize(("picks", "sleeve"), [(('QQQ', 'GLD'), 'SHY'),
                                              (('SMH', 'SPY'), 'SHY'),
                                              (('SMH', 'GLD'), 'QQQ')])
def test_index_pick_routes_ten_percent_to_short_treasuries_or_qqq(
    picks: tuple[str, str], sleeve: str
) -> None:
    assert allocation(picks) == {picks[0]: .45, picks[1]: .45, sleeve: .10}


def test_top_two_stay_invested_when_shy_outperforms_both() -> None:
    prepared = load_prepared("ANNUAL")
    contract = Contract(mode="ANNUAL", stage="science", cost_bps=5,
                        schedule_sha256=prepared.schedule.sha256)
    day = date(2023, 10, 31)
    qualified = annual_signals.oracle(contract, prepared, day).q
    expected = tuple(s for s in qualified if s != "SHY")
    assert len(expected) == 2
    prepared.features.frames["SHY"].loc[str(day), "mom252"] = 100.0
    assert annual_signals.target(contract, prepared, day).q == {"SHY": 1.0}
    assert select(contract, prepared, day).q == allocation(expected)
