from typing import Any, override

import backtrader as bt

from .models import FeeSchedule


def commission(size: float, price: float, fees: FeeSchedule) -> float:
    q = abs(size)
    if q == 0:
        return 0.0
    total = max(fees.commission_min, fees.commission_per_share * q)
    total += max(fees.platform_min, fees.platform_per_share * q)
    total += (fees.settlement_per_share + fees.nms_per_share) * q
    if size < 0:
        total += max(fees.sec_min, fees.sec_rate * q * price)
    return float(total)


class BrokerFees(bt.CommInfoBase):
    params = (
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_FIXED),
        ("leverage", 1.0),
        ("schedule", None),
    )

    @override
    def _getcommission(self, size: float, price: float, pseudoexec: bool) -> float:
        schedule: FeeSchedule = self.p.schedule
        return commission(size, price, schedule)


class DailyFeed(bt.feeds.PandasData):
    lines = ("tr", "mom21", "mom63", "mom126", "week_end")
    params = (("tr", -1), ("mom21", -1), ("mom63", -1), ("mom126", -1), ("week_end", -1))


def native_return(analyzer: Any) -> float:
    return float(analyzer.get_analysis()["rnorm100"])
