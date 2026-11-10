"""Frozen group-cap identities and targets; never executes an account."""

import math
from datetime import date
from pathlib import Path
from typing import Literal

from beartype import beartype
from pydantic import ConfigDict, TypeAdapter, model_validator

from rotbt.models import Model
from rotbt.phase2.models import FeatureStore
from scripts.etf_c10_2026_audit import TargetRecord

RESEARCH = "research/ETF_GROUP_CAP_20261109"
Pool = Literal["legacy36", "historical34"]
Cohort = Literal["historical2022_2025", "reset2026", "continuous2022_2026"]


class Role(Model):
    model_config = ConfigDict(frozen=True)
    key: str
    pool: Pool
    cap_one_per_group: bool
    top_k: Literal[1, 2, 3]
    cost_bps: Literal[5, 20]
    cohort: Cohort
    mode: Literal["reuse", "alias", "new_native"]
    alias_of: str | None

    @model_validator(mode="after")
    def fixed_identity(self) -> "Role":
        cap = "cap1" if self.cap_one_per_group else "uncapped"
        if self.key != f"{self.pool}_{cap}_K{self.top_k}_{self.cost_bps}bp_{self.cohort}":
            raise ValueError("role identity disagrees with economic contract")
        if (self.mode == "alias") != (self.alias_of is not None):
            raise ValueError("alias identity requires an explicit original role")
        return self

    @property
    def start(self) -> date:
        return date(2026, 1, 2) if self.cohort == "reset2026" else date(2022, 1, 3)

    @property
    def through(self) -> date:
        return date(2025, 12, 31) if self.cohort == "historical2022_2025" else date(2026, 11, 6)

    @property
    def rule(self) -> Literal["C10", "C11", "C00"]:
        return {1: "C10", 2: "C11", 3: "C00"}[self.top_k]  # type: ignore[return-value]


class GroupMap(Model):
    model_config = ConfigDict(extra="ignore")
    groups: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def unique_groups(self) -> "GroupMap":
        names = [s for group in self.groups.values() for s in group]
        if len(names) != len(set(names)) or "SHY" in names:
            raise ValueError("risk group membership must be unique and exclude defender SHY")
        return self

    def group_for(self, symbol: str) -> str:
        for name, members in self.groups.items():
            if symbol in members:
                return name
        raise ValueError(f"missing risk group: {symbol}")


class Pools(Model):
    model_config = ConfigDict(extra="ignore")
    legacy36: tuple[str, ...]
    historical34: tuple[str, ...]

    def names(self, pool: Pool) -> tuple[str, ...]:
        return self.legacy36 if pool == "legacy36" else self.historical34


def load_roles(root: Path) -> list[Role]:
    roles = TypeAdapter(list[Role]).validate_json(
        (root / RESEARCH / "CATALOG_ROLES.json").read_text()
    )
    by_key = {r.key: r for r in roles}
    if len(by_key) != 72 or [
        sum(r.mode == m for r in roles) for m in ("reuse", "alias", "new_native")
    ] != [18, 12, 42]:
        raise ValueError("frozen72-role catalog incomplete")
    for role in roles:
        if role.alias_of:
            original = by_key[role.alias_of]
            if (
                role.top_k != 1
                or original.cap_one_per_group
                or role.model_dump(exclude={"key", "cap_one_per_group", "mode", "alias_of"})
                != original.model_dump(exclude={"key", "cap_one_per_group", "mode", "alias_of"})
            ):
                raise ValueError("only identical top-one group-cap roles may alias")
    return roles


@beartype
def decision(
    role: Role, features: FeatureStore, names: tuple[str, ...], groups: GroupMap, day: date
) -> TargetRecord:
    scores: dict[str, float] = {}
    for symbol in (*names, "SHY"):
        if symbol != "SHY":
            groups.group_for(symbol)
        frame = features.frames[symbol].loc[: str(day)]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("missing exact prior signal close")
        if len(frame) < 253:
            if symbol == "SHY":
                raise ValueError("insufficient SHY observations")
            continue
        row = frame.iloc[-1]
        values = [float(row[k]) for k in ("mom252", "raw_close", "turnover20", "positive20")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite group-cap signal input")
        if symbol == "SHY" or (values[1] >= 5 and values[2] >= 1e7 and values[3] == 20):
            scores[symbol] = values[0]
    selected: list[str] = []
    used: set[str] = set()
    for symbol in sorted((s for s in scores if s != "SHY"), key=lambda s: (-scores[s], s)):
        group = groups.group_for(symbol)
        if not role.cap_one_per_group or group not in used:
            selected.append(symbol)
            used.add(group)
        if len(selected) == role.top_k:
            break
    q = {s: 0.9 / role.top_k for s in selected if scores[s] > scores["SHY"]}
    q["SHY"] = max(0.0, 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")
