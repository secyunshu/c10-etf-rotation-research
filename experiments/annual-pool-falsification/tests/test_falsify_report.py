"""Mechanism claims need additive wealth contrasts and full paths, not rounded CAGR."""

import pandas as pd
import pytest

from scripts.falsify_report import contrasts


def test_factorial_contrasts_use_final_wealth_and_keep_interaction() -> None:
    table = pd.DataFrame(
        [
            {"case": name, "cost_bps": 5, "final_equity": value}
            for name, value in zip("ABCDE", (100, 130, 105, 140, 140), strict=True)
        ]
    )
    result = contrasts(table, 5)
    assert result.loc[result.contrast == "B-A", "wealth_difference"].item() == 30
    assert result.loc[result.contrast == "D-C", "wealth_difference"].item() == 35
    assert result.loc[result.contrast == "D-B-C+A", "wealth_difference"].item() == 5
    assert result.loc[result.contrast == "D-E", "wealth_difference"].item() == 0
    with pytest.raises(ValueError, match="exactly one"):
        contrasts(table.iloc[:-1], 5)
