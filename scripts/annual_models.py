"""Independent annual-pool contracts; the published V2 contracts stay intact."""

import hashlib
import json
from datetime import date, datetime
from typing import Literal

import exchange_calendars as xcals
import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from c10standard.models import GROUPS, RISK, NativeReturn, OpenTrade, TargetRecord
from rotbt.models import Dataset, FeeSchedule, Model, Result
from rotbt.phase2.models import FeatureStore
from scripts.v2_2023_models import GroupMap as GroupMap

Mode = Literal["FIXED", "ANNUAL"]
Stage = Literal["bridge", "science"]
Event = Literal["started", "native_completed", "finished", "failed", "revalidated"]
FIRST, START, LAST = date(2020, 12, 31), date(2022, 1, 3), date(2025, 12, 31)


def pool_sha256(year: int, risk_names: tuple[str, ...], groups: GroupMap,
                price_asof: date, selection_asof: date, execute_on: date,
                source_asof: date) -> str:
    payload = {"year": year, "risk_names": risk_names, "groups": groups.groups,
               "price_asof": str(price_asof), "selection_asof": str(selection_asof),
               "execute_on": str(execute_on), "source_asof": str(source_asof)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class Pool(Model):
    model_config = ConfigDict(frozen=True, revalidate_instances="always")
    year: int = Field(ge=2022, le=2025)
    risk_names: tuple[str, ...]
    groups: GroupMap
    price_asof: date
    selection_asof: date
    execute_on: date
    source_asof: date
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bound_pool(self) -> "Pool":
        members = [s for names in self.groups.groups.values() for s in names]
        if (not self.risk_names or "SHY" in self.risk_names
                or tuple(sorted(set(self.risk_names))) != self.risk_names
                or sorted(members) != list(self.risk_names)
                or not set(self.groups.groups) <= set(GROUPS)):
            raise ValueError("pool risk names and economic group coverage differ")
        calendar = xcals.get_calendar("XNYS")
        selection = date(self.year - 1, 12, 31)
        price = calendar.date_to_session(pd.Timestamp(selection), direction="previous")
        if (self.selection_asof != selection or self.price_asof != price.date()
                or self.execute_on != calendar.next_session(price).date()
                or self.source_asof > self.selection_asof):
            raise ValueError("pool price, selection, source and execution timing disagree")
        if self.sha256 != pool_sha256(self.year, self.risk_names, self.groups, self.price_asof,
                                      self.selection_asof, self.execute_on, self.source_asof):
            raise ValueError("pool hash does not bind membership and four information dates")
        return self


def schedule_sha256(mode: Mode, entries: tuple[Pool, ...]) -> str:
    payload = {"mode": mode, "entries": [entry.sha256 for entry in entries]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class Schedule(Model):
    model_config = ConfigDict(frozen=True, revalidate_instances="always")
    mode: Mode
    entries: tuple[Pool, ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def four_pools(self) -> "Schedule":
        if tuple(p.year for p in self.entries) != (2022, 2023, 2024, 2025):
            raise ValueError("schedule requires exactly four ordered pools for 2022 through 2025")
        for pool in self.entries:
            if self.mode == "FIXED":
                if (pool.risk_names != RISK or pool.groups.groups != GROUPS
                        or pool.source_asof != date(2021, 12, 31)):
                    raise ValueError("fixed schedule must preserve published 2021 pool and source")
            elif pool.source_asof != date(pool.year - 1, 12, 31):
                raise ValueError("annual pool source must be the preceding year end")
        if self.sha256 != schedule_sha256(self.mode, self.entries):
            raise ValueError("schedule hash does not bind all four pools and mode")
        return self


class Contract(Model):
    model_config = ConfigDict(frozen=True)
    mode: Mode
    cost_bps: Literal[5, 20]
    stage: Stage
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def registered_pair(self) -> "Contract":
        if (self.stage == "bridge") != (self.mode == "FIXED"):
            raise ValueError("only FIXED bridges and ANNUAL science are registered")
        return self

    @property
    def key(self) -> str:
        return f"{self.mode}_{self.cost_bps}bp_{self.stage}"

    @property
    def start(self) -> date:
        return START

    @property
    def through(self) -> date:
        return LAST


class Prepared(Model):
    schedule: Schedule
    datasets: list[Dataset]
    features: FeatureStore
    fees: FeeSchedule
    first: date = FIRST
    last: date = LAST

    @model_validator(mode="after")
    def input_identity(self) -> "Prepared":
        names = [d.symbol for d in self.datasets]
        union = {s for p in self.schedule.entries for s in p.risk_names} | {"SHY"}
        if (self.first, self.last) != (FIRST, LAST):
            raise ValueError("physical input scope must be 2020-12-31 through 2025")
        if any(d.frame.empty or d.frame.index.min().date() < FIRST
               or d.frame.index.max().date() > LAST for d in self.datasets):
            raise ValueError("physical market cutoff violated")
        if (len(names) != len(set(names)) or set(names) != union
                or set(self.features.frames) != union):
            raise ValueError("annual union, datasets and feature membership differ")
        if tuple(getattr(self.fees, name) for name in FeeSchedule.model_fields) != (
                .0049, .99, .005, 1., .003, .0000206, .01, .000003):
            raise ValueError("original V2 fee schedule changed")
        return self


class DecisionTiming(Model):
    pool_year: int
    price_asof: date
    selection_asof: date
    source_asof: date
    execute_on: date
    pool_sha256: str


class Artifact(Model):
    contract: Contract
    account: Result
    targets: list[TargetRecord]
    decision_times: list[DecisionTiming]
    native_returns: list[NativeReturn]
    open_trades: list[OpenTrade]


class Binding(Model):
    experiment: Literal["V2_ANNUAL_REPOOL_V1"] = "V2_ANNUAL_REPOOL_V1"
    contract: Contract
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NativeComplete(Model):
    artifact_sha256: str


class Attempt(Model):
    experiment: Literal["V2_ANNUAL_REPOOL_V1"] = "V2_ANNUAL_REPOOL_V1"
    event: Event
    account: str
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""
