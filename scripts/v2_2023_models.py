"""Separate fixed 2023-account contracts; the published V2 contracts stay intact."""

import hashlib
import json
from datetime import date, datetime
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from c10standard.models import GROUPS, RISK, NativeReturn, OpenTrade, TargetRecord
from rotbt.models import Dataset, FeeSchedule, Model, Result
from rotbt.phase2.models import FeatureStore

PoolRole = Literal["NEW_POOL", "OLD_POOL"]
Role = Literal["NEW_POOL", "OLD_POOL", "SPY_BH", "QQQ_BH"]
Stage = Literal["bridge", "science"]
Event = Literal["started", "native_completed", "finished", "failed", "revalidated"]
FIRST, LAST = date(2020, 12, 31), date(2025, 12, 31)


class GroupMap(Model):
    groups: dict[str, tuple[str, ...]]


def pool_sha256(role: PoolRole, risk_names: tuple[str, ...], groups: GroupMap) -> str:
    payload = {"role": role, "risk_names": risk_names, "groups": groups.groups}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class Pool(Model):
    model_config = ConfigDict(frozen=True, revalidate_instances="always")
    role: PoolRole
    risk_names: tuple[str, ...]
    groups: GroupMap
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bound_universe(self) -> "Pool":
        members = [s for group in self.groups.groups.values() for s in group]
        if (not self.risk_names or "SHY" in self.risk_names
                or tuple(sorted(set(self.risk_names))) != self.risk_names
                or sorted(members) != list(self.risk_names)
                or not set(self.groups.groups) <= set(GROUPS)):
            raise ValueError("pool risk names and economic group coverage differ")
        if self.role == "OLD_POOL" and (self.risk_names != RISK or self.groups.groups != GROUPS):
            raise ValueError("old pool must equal the published historical V2 groups")
        if self.sha256 != pool_sha256(self.role, self.risk_names, self.groups):
            raise ValueError("pool hash does not bind its membership and groups")
        return self


class Contract(Model):
    model_config = ConfigDict(frozen=True)
    role: Role
    cost_bps: Literal[5, 20]
    stage: Stage
    pool_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def fixed_bridge(self) -> "Contract":
        if self.stage == "bridge" and self.role != "OLD_POOL":
            raise ValueError("bridge requires original OLD_POOL")
        return self

    @property
    def key(self) -> str:
        return f"{self.role}_{self.cost_bps}bp_{self.stage}"

    @property
    def start(self) -> date:
        return date(2022, 1, 3) if self.stage == "bridge" else date(2023, 1, 3)

    @property
    def through(self) -> date:
        return LAST


class Prepared(Model):
    pool: Pool
    datasets: list[Dataset]
    features: FeatureStore
    fees: FeeSchedule
    first: date = FIRST
    last: date = LAST

    @model_validator(mode="after")
    def input_identity(self) -> "Prepared":
        names = [d.symbol for d in self.datasets]
        if (self.first, self.last) != (FIRST, LAST):
            raise ValueError("physical input scope must be 2020-12-31 through 2025")
        if any(d.frame.empty or d.frame.index.min().date() < FIRST
               or d.frame.index.max().date() > LAST for d in self.datasets):
            raise ValueError("physical market cutoff violated")
        if (len(names) != len(set(names)) or set(names) != {*self.pool.risk_names, "SHY"}
                or set(self.features.frames) != set(names)):
            raise ValueError("pool, datasets and feature membership differ")
        if tuple(getattr(self.fees, name) for name in FeeSchedule.model_fields) != (
                .0049, .99, .005, 1., .003, .0000206, .01, .000003):
            raise ValueError("original V2 fee schedule changed")
        return self


class Artifact(Model):
    contract: Contract
    account: Result
    targets: list[TargetRecord]
    native_returns: list[NativeReturn]
    open_trades: list[OpenTrade]


class Binding(Model):
    experiment: Literal["V2_REPOOL_2023_V1"] = "V2_REPOOL_2023_V1"
    contract: Contract
    source_sha256: str


class NativeComplete(Model):
    artifact_sha256: str


class Attempt(Model):
    experiment: Literal["V2_REPOOL_2023_V1"] = "V2_REPOOL_2023_V1"
    event: Event
    account: str
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""
