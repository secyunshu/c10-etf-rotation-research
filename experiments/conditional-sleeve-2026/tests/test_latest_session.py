from datetime import date

import pandas as pd
import pytest

from rotbt.models import Dataset
from scripts.conditional_sleeve_ytd import ClosingBar, ResetContract, append_bar


def test_year_open_contract_marks_latest_close_without_liquidating() -> None:
    contract = ResetContract()
    assert contract.start == date(2026, 1, 2)
    assert contract.through == date(2026, 11, 11)
    assert contract.mode == "reset2026"
    assert not contract.terminal


def test_mark_uses_added_raw_close_and_keeps_frozen_history() -> None:
    history = pd.DataFrame({"close": [500.], "tr": [280.]},
                           index=pd.DatetimeIndex(["2026-11-10"], name="date"))
    data = Dataset(symbol="SMH", frame=history, dividends=[])
    bar = ClosingBar(symbol="SMH", day=date(2026, 11, 11), open=501, high=514,
                     low=499, close=510, volume=10_000_000, turnover=5_100_000_000)
    marked = append_bar(data, bar)
    assert len(data.frame) == 1
    assert marked.frame.close.tolist() == [500., 510.]
    assert marked.frame.tr.iloc[-1] == pytest.approx(285.6)
    with pytest.raises(ValueError, match="symbol or date"):
        append_bar(data, bar.model_copy(update={"symbol": "QQQ"}))
