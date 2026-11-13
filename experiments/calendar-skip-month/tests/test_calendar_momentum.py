from datetime import date

import pandas as pd
import pytest

from scripts.calendar_momentum import calendar_ends, calendar_score


def test_november_holding_excludes_october_only_in_french_score() -> None:
    start, full, skip = calendar_ends(date(2026, 10, 30))
    assert (start.date(), full.date(), skip.date()) == (
        date(2025, 10, 31), date(2026, 10, 30), date(2026, 9, 30)
    )
    tr = pd.Series([100., 120., 90., 200.], index=pd.to_datetime([
        "2025-10-31", "2026-09-30", "2026-10-30", "2026-11-02",
    ]))
    assert calendar_score(tr, start, full) == pytest.approx(-.10)
    assert calendar_score(tr, start, skip) == pytest.approx(.20)


def test_first_holding_month_uses_known_month_end_instead_of_calendar_weekend() -> None:
    start, full, skip = calendar_ends(date(2022, 12, 30))
    assert (start.date(), full.date(), skip.date()) == (
        date(2021, 12, 31), date(2022, 12, 30), date(2022, 11, 30)
    )
    with pytest.raises(ValueError, match="month end"):
        calendar_ends(date(2022, 12, 29))
