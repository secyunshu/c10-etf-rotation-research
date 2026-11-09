"""Frozen C00–35 targets; original native broker owns all execution and cash."""

from datetime import date
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from rotbt.models import Model
from rotbt.phase2.models import FeatureStore, SignalContext
from rotbt.phase2.signals import _cap_weights  # pyright: ignore[reportPrivateUsage]


class Rule(Model):
    id: str

    @property
    def label(self) -> str:
        return self.id

    frequency: str = "M"
    band: float = 0
    effect: str


def catalog() -> list[Rule]:
    effects = (
        "anchor",
        "no_cash",
        "all_trend",
        "bucket_static",
        "momentum126",
        "momentum189",
        "skip_month",
        "multi_rank",
        "regression",
        "vol_score",
        "top1",
        "top2",
        "top4",
        "top5",
        "risk100",
        "risk60",
        "rank_weight",
        "inverse_vol",
        "correlation_gate",
        "correlation_gate",
        "class_limit",
        "post_sma",
        "cash_margin",
        "pre_sma",
        "stock_sma",
        "band2",
        "band5",
        "rank_buffer",
        "hurdle",
        "weekly",
        "fixed_biweekly",
        "idle_cash",
        "sgov_cash",
        "no_metals",
        "no_commodity",
        "stocks_only",
    )
    return [
        Rule(
            id=f"C{i:02d}",
            effect=effect,
            frequency="W" if i == 29 else "2W" if i == 30 else "M",
            band=0.02 if i in (3, 25) else 0.05 if i == 26 else 0,
        )
        for i, effect in enumerate(effects)
    ]


def target(
    rule: Rule | str,
    features: FeatureStore,
    day: date,
    names: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
    context: SignalContext | None = None,
) -> dict[str, float]:
    rule = next(r for r in catalog() if r.label == rule) if isinstance(rule, str) else rule
    number = int(rule.label[1:])
    if rule.label not in {r.label for r in catalog()}:
        raise ValueError("unregistered return template")
    if len(names) != len(set(names)) or "SHY" in names or len(groups) != 4:
        raise ValueError("invalid risk universe or four economic groups")
    members = [s for group in groups for s in group]
    if len(members) != len(set(members)) or set(members) != set(names):
        raise ValueError("economic groups must partition risk universe")
    visible: dict[str, Any] = {
        s: features.frames[s].loc[: pd.Timestamp(day)] for s in (*names, "SHY")
    }
    if any(frame.empty or frame.index[-1] != pd.Timestamp(day) for frame in visible.values()):
        raise ValueError("missing exact prior observation")

    def value(s: str, column: str) -> float:
        v = float(visible[s][column].iloc[-1])
        if not isfinite(v):
            raise ValueError(f"nonfinite target feature {s}/{column}")
        return v

    eligible = [
        s
        for s in names
        if len(visible[s]) >= 253
        and value(s, "turnover20") >= 1e7
        and value(s, "positive20") >= 20
        and value(s, "raw_close") >= 5
    ]
    if number == 33:
        eligible = [s for s in eligible if s not in ("GLD", "SLV")]
    if number == 34:
        eligible = [s for s in eligible if s != "PDBC"]
    if number == 35:
        eligible = [s for s in eligible if s in groups[0]]
    if number == 23:
        eligible = [s for s in eligible if value(s, "tr") > value(s, "sma200")]
    score = {s: value(s, "mom252") for s in eligible}
    cash = {s: value(s, "mom252") for s in (*eligible, "SHY")}
    if number in (4, 5, 6):
        column = {4: "mom126", 5: "mom189", 6: "momskip21"}[number]
        score = {s: value(s, column) for s in eligible}
        cash = {s: value(s, column) for s in (*eligible, "SHY")}
    elif number == 7:
        ranks: dict[str, list[int]] = {s: [] for s in eligible}
        for window in (63, 126, 252):
            ordered = sorted(eligible, key=lambda s: (-value(s, f"mom{window}"), s))
            for rank, s in enumerate(ordered):
                ranks[s].append(rank)
        score = {s: -float(np.mean(ranks[s])) for s in eligible}
        cash = {
            s: float(np.mean([value(s, f"mom{w}") for w in (63, 126, 252)]))
            for s in (*eligible, "SHY")
        }
    elif number == 8:
        score = {s: value(s, "reg252") for s in eligible}
    elif number == 9:
        score = {s: value(s, "mom252") / max(0.01, value(s, "vol63")) for s in eligible}
    ranked = sorted(eligible, key=lambda s: (-score[s], s))
    margin = 0.02 if number == 22 else 0

    def cash_ok(s: str) -> bool:
        return number == 1 or cash[s] > cash["SHY"] + margin

    if number == 2:
        chosen = [s for s in ranked if cash_ok(s)]
        q = {s: min(0.3, 0.9 / len(chosen)) for s in chosen}
    elif number == 3:
        q = {
            s: min(0.3, 0.25 / len(selected))
            for group in groups
            if (selected := [s for s in group if s in eligible])
            for s in selected
        }
    else:
        count = {10: 1, 11: 2, 12: 4, 13: 5}.get(number, 3)
        chosen = []
        if number in (18, 19, 20):
            for s in ranked:
                if len(chosen) >= 3:
                    break
                if number == 20:
                    group = next(g for g in groups if s in g)
                    if sum(x in group for x in chosen) >= 2:
                        continue
                else:
                    reject = False
                    for prior in chosen:
                        a = visible[s]["ret"].iloc[-126:].to_numpy(dtype=float)
                        b = visible[prior]["ret"].iloc[-126:].to_numpy(dtype=float)
                        if (
                            len(a) != 126
                            or len(b) != 126
                            or not np.isfinite(a).all()
                            or not np.isfinite(b).all()
                            or np.std(a) == 0
                            or np.std(b) == 0
                        ):
                            raise ValueError("uncomparable correlation history")
                        correlation = float(np.corrcoef(a, b)[0, 1])
                        if not isfinite(correlation):
                            raise ValueError("uncomparable correlation")
                        if correlation > (0.8 if number == 18 else 0.9):
                            reject = True
                            break
                    if reject:
                        continue
                chosen.append(s)
        elif number in (27, 28):
            if context is None or context.day != day:
                raise ValueError("actual account context required")
            held = {h.symbol for h in context.holdings if h.shares > 0 and h.symbol != "SHY"}
            chosen = [
                s
                for s in ranked
                if s in held and cash_ok(s) and (number == 28 or ranked.index(s) < 5)
            ][:3]
            for s in ranked:
                if len(chosen) >= 3:
                    break
                if s not in chosen and cash_ok(s):
                    chosen.append(s)
            if number == 28:
                while chosen:
                    challengers = [s for s in ranked if s not in chosen and cash_ok(s)]
                    if not challengers:
                        break
                    challenger = challengers[0]
                    weakest = min(chosen, key=lambda s: (cash[s], s))
                    if cash[challenger] <= cash[weakest] + 0.02:
                        break
                    chosen.remove(weakest)
                    chosen.append(challenger)
        else:
            chosen = ranked[:count]
        weights = {
            s: {10: 0.9, 11: 0.45, 12: 0.225, 13: 0.18, 14: 1 / 3, 15: 0.2}.get(number, 0.3)
            for s in chosen
        }
        if number == 16:
            weights = {s: w for s, w in zip(chosen, (0.4, 0.3, 0.2), strict=False)}
        elif number == 17 and chosen:
            weights = {
                s: float(w) * 0.9
                for s, w in zip(
                    chosen,
                    _cap_weights(np.array([1 / max(0.01, value(s, "vol63")) for s in chosen]), 0.5),
                    strict=True,
                )
            }
        q = {
            s: w
            for s, w in weights.items()
            if cash_ok(s)
            and (number != 21 or value(s, "tr") > value(s, "sma200"))
            and (number != 24 or s not in groups[0] or value(s, "tr") > value(s, "sma200"))
        }
    residual = max(0.0, 1 - sum(q.values()))
    if number == 32:
        q["SGOV"] = q.get("SGOV", 0) + residual
    elif number != 31:
        q["SHY"] = residual
    if (
        not set(q) <= set(names) | {"SHY"}
        or any(not isfinite(v) or v < 0 for v in q.values())
        or sum(q.values()) > 1 + 1e-10
    ):
        raise ValueError("invalid deployment-before target")
    return {s: float(w) for s, w in q.items() if w > 1e-12}
