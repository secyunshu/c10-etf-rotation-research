from datetime import date
from typing import override

from scripts.conditional_sleeve_ytd import ResetContract
from scripts.semimonthly_v23 import decision_days


class November(ResetContract):
    @property
    @override
    def through(self) -> date:
        return date(2026, 11, 13)


def test_weekend_midmonth_uses_prior_close_and_preserves_month_end() -> None:
    monthly = decision_days(November(), "monthly")
    semi = decision_days(November(), "semimonthly")
    assert set(monthly) < set(semi)
    assert date(2026, 2, 13) in semi and date(2026, 2, 17) not in semi
    assert date(2026, 10, 15) in semi and date(2026, 10, 30) in semi
    assert date(2026, 11, 13) not in semi
