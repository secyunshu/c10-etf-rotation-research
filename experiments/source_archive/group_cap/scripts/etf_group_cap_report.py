"""Describe audited native account books without executing or selecting a strategy."""

import json
import math
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from beartype import beartype
from pydantic import ConfigDict

from rotbt.models import Model
from scripts.etf_group_cap_audit import verify_hashes
from scripts.etf_group_cap_targets import Role, load_roles


class Statistics(Model):
    days: int
    opening: float
    closing: float
    total_return: float
    max_drawdown: float
    cagr_252: float
    annualized_volatility: float
    sharpe_rf0: float
    fees: float
    orders: int
    turnover: float


class Binding(Model):
    model_config = ConfigDict(extra="ignore")
    role: Role
    path: Path
    mode: Literal["reuse", "alias", "new_native"]


class Registry(Model):
    model_config = ConfigDict(extra="ignore")
    roles: list[Binding]


@beartype
def summarize(
    daily: pd.DataFrame, orders: pd.DataFrame, first: date, last: date, opening: float
) -> Statistics:
    """Calculate statistics from native close NAV, including period opening capital."""
    equity = daily.loc[str(first) : str(last), "equity"].to_numpy(dtype=float)
    if not len(equity):
        raise ValueError("empty native NAV period")
    values = np.r_[opening, equity]
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("invalid native equity")
    changes = values[1:] / values[:-1] - 1
    sigma = float(np.std(changes, ddof=1)) if len(changes) > 1 else 0.0
    dates = pd.to_datetime(orders.day)
    selected = orders.loc[(dates >= pd.Timestamp(first)) & (dates <= pd.Timestamp(last))]
    return Statistics(
        days=len(equity),
        opening=opening,
        closing=float(equity[-1]),
        total_return=float(equity[-1] / opening - 1),
        max_drawdown=float(np.max(1 - values / np.maximum.accumulate(values))),
        cagr_252=float((equity[-1] / opening) ** (252 / len(equity)) - 1),
        annualized_volatility=sigma * math.sqrt(252),
        sharpe_rf0=float(np.mean(changes)) / sigma * math.sqrt(252) if sigma else 0.0,
        fees=float(selected.fee.sum()),
        orders=len(selected),
        turnover=float((selected["size"] * selected.price).abs().sum()) / float(np.mean(equity)),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "etf_group_cap_outputs/20261109_v1"
    report = root / "research/ETF_GROUP_CAP_20261109"
    registry = Registry.model_validate_json((out / "REGISTRY.json").read_text())
    if [b.role for b in registry.roles] != load_roles(root):
        raise ValueError("report registry differs from frozen72 roles")
    receipt = json.loads((out / "FULL_AUDIT.json").read_text())
    if receipt["status"] != "PASS":
        raise ValueError("audited completed accounts required")
    verify_hashes(receipt["hashes"])
    rows: list[dict[str, str | int | float | bool]] = []
    months: list[dict[str, str | int | float | bool]] = []
    for binding in registry.roles:
        role, folder = binding.role, binding.path
        daily = pd.read_csv(folder / "daily.csv.gz", parse_dates=["day"]).set_index("day")
        orders = pd.read_csv(folder / "orders.csv.gz")
        if role.cohort == "historical2022_2025":
            periods = [(str(y), date(y, 1, 1), date(y, 12, 31)) for y in range(2022, 2026)]
            periods.append(("full2022_2025", role.start, role.through))
        else:
            periods = [("2026YTD", date(2026, 1, 2), role.through)]
        labels: dict[str, str | int | float | bool] = {
            "key": role.key,
            "pool": role.pool,
            "cap": role.cap_one_per_group,
            "k": role.top_k,
            "cost_bps": role.cost_bps,
            "cohort": role.cohort,
            "mode": binding.mode,
        }
        for name, first, last in periods:
            before = daily.loc[daily.index < pd.Timestamp(first), "equity"]
            opening = float(before.iloc[-1]) if len(before) else 50_000.0
            stats = summarize(daily, orders, first, last, opening)
            rows.append({**labels, "period": name, **stats.model_dump()})
        if role.cohort != "historical2022_2025":
            previous = 50_000.0
            if role.cohort == "continuous2022_2026":
                previous = float(
                    daily.loc[[pd.Timestamp("2025-12-31")], "equity"].to_numpy(dtype=float)[0]
                )
            window = daily.loc["2026-01-02" : str(role.through), "equity"]
            for month in sorted(set(pd.DatetimeIndex(window.index).strftime("%Y-%m"))):
                part = window.loc[month]
                end = float(part.iloc[-1])
                months.append({**labels, "month": month, "total_return": end / previous - 1})
                previous = end
    frame = pd.DataFrame(rows)
    frame.to_csv(report / "PERIOD_METRICS.csv", index=False)
    pd.DataFrame(months).to_csv(report / "MONTHLY_2026.csv", index=False)
    # Top1 aliases must be economically identical; no duplicate independent successes.
    for pool in ("legacy36", "historical34"):
        part = frame.loc[(frame.pool == pool) & (frame.k == 1)]
        for _, pair in part.groupby(["cohort", "cost_bps", "period"]):
            if len(pair) != 2 or not np.array_equal(
                pair.iloc[0][["total_return", "max_drawdown"]].to_numpy(),
                pair.iloc[1][["total_return", "max_drawdown"]].to_numpy(),
            ):
                raise ValueError("top-one cap must be exact identity")
    print(f"Reported {len(registry.roles)} roles, {len(frame)} period rows; top1 aliases exact")
    verify_hashes(receipt["hashes"])


if __name__ == "__main__":
    main()
