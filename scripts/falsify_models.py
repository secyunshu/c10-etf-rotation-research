"""Typed counterfactual identities separate from the unchanged native envelope."""

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from beartype import beartype
from pydantic import ConfigDict, Field, model_validator

from c10standard.models import Receipt
from c10standard.native import write
from rotbt.models import Model
from scripts.annual_models import Contract, Event, GroupMap, Pool, Schedule, pool_sha256

CaseId = Literal["A", "B", "C", "D", "E"]
Stage = Literal["bridge", "science"]


class AssetSources(Model):
    year: int
    asof: dict[str, date]


@beartype
def case_hash(case_id: CaseId, pools: tuple[Pool, ...], sources: tuple[AssetSources, ...],
              note: str) -> str:
    payload = {"id": case_id, "pools": [p.sha256 for p in pools],
               "sources": [s.model_dump(mode="json") for s in sources], "note": note}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class Case(Model):
    model_config = ConfigDict(frozen=True, revalidate_instances="always")
    id: CaseId
    pools: tuple[Pool, ...]
    pool_sources: tuple[AssetSources, ...]
    intervention_note: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity(self) -> "Case":
        if tuple(p.year for p in self.pools) != (2022, 2023, 2024, 2025):
            raise ValueError("case requires four ordered actual pools")
        if len(self.pool_sources) != 4:
            raise ValueError("case requires four asset source maps")
        for pool, source in zip(self.pools, self.pool_sources, strict=True):
            if (source.year != pool.year or set(source.asof) != set(pool.risk_names)
                    or max(source.asof.values()) != pool.source_asof
                    or any(d > pool.selection_asof for d in source.asof.values())):
                raise ValueError("asset sources differ from pool membership or cutoff")
        if self.sha256 != case_hash(self.id, self.pools, self.pool_sources,
                                    self.intervention_note):
            raise ValueError("case hash does not bind actual policy and sources")
        return self


class Cases(Model):
    cases: tuple[Case, ...]

    @model_validator(mode="after")
    def registered(self) -> "Cases":
        if tuple(c.id for c in self.cases) != ("A", "B", "C", "D", "E"):
            raise ValueError("exactly the five registered cases are required")
        return self


class Account(Model):
    model_config = ConfigDict(frozen=True, revalidate_instances="always")
    case: Case
    cost_bps: Literal[5, 20]
    stage: Stage
    envelope: Contract

    @model_validator(mode="after")
    def registered(self) -> "Account":
        bridge = (self.case.id, self.cost_bps) in (("A", 5), ("D", 20))
        if ((self.stage == "bridge") != bridge
                or (self.stage == "science" and self.case.id not in ("B", "C", "E"))
                or self.envelope.cost_bps != self.cost_bps
                or self.envelope.mode != ("FIXED" if self.case.id == "A" else "ANNUAL")):
            raise ValueError("unregistered counterfactual account or envelope")
        return self

    @property
    def key(self) -> str:
        return f"{self.case.id}_{self.cost_bps}bp_{self.stage}"


class CaseBinding(Model):
    experiment: Literal["ANNUAL_POOL_FALSIFICATION_V1"] = "ANNUAL_POOL_FALSIFICATION_V1"
    envelope_role: Literal["execution_container_only"] = "execution_container_only"
    account: Account
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaseReceipt(Model):
    account: str
    native: Receipt
    bridge_differences: dict[str, float] = Field(default_factory=dict)


class Attempt(Model):
    experiment: Literal["ANNUAL_POOL_FALSIFICATION_V1"] = "ANNUAL_POOL_FALSIFICATION_V1"
    event: Event
    account: str
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""


@beartype
def build_cases(fixed: Schedule, annual: Schedule) -> Cases:
    Schedule.model_validate(fixed)
    Schedule.model_validate(annual)
    if fixed.mode != "FIXED" or annual.mode != "ANNUAL" or fixed.entries[0] != annual.entries[0]:
        raise ValueError("reference policies must share their original 2022 pool")
    notes: tuple[tuple[CaseId, str], ...] = (
        ("A", "Original 2021 fixed pool; reference policy."),
        ("B", "From 2023 force SMH using 2022 source; other assets retain 2021 sources."),
        ("C", "Annual pools with SOXX forced from 2023 using 2021 source; "
              "no AUM eligibility claim."),
        ("D", "Original annual pools; reference policy."),
        ("E", "Apply 2022 source pool from 2023 and retain it through 2025."),
    )
    cases = []
    for case_id, note in notes:
        pools, sources = [], []
        for i, template in enumerate(fixed.entries):
            base = fixed.entries[i] if case_id in ("A", "B") else annual.entries[i]
            if case_id == "E" and i > 0:
                base = annual.entries[1]
            groups = GroupMap(groups={g: names for g, names in base.groups.groups.items()})
            asof = {s: base.source_asof for s in base.risk_names}
            if i > 0 and case_id in ("B", "C"):
                old, new = ("SOXX", "SMH") if case_id == "B" else ("SMH", "SOXX")
                if old not in asof or new in asof:
                    raise ValueError("semiconductor intervention must replace exactly one asset")
                groups = GroupMap(groups={g: tuple(new if s == old else s for s in names)
                                          for g, names in groups.groups.items()})
                del asof[old]
                asof[new] = date(2022 if case_id == "B" else 2021, 12, 31)
            names = tuple(sorted(asof))
            source = max(asof.values())
            pools.append(Pool(year=template.year, risk_names=names, groups=groups,
                              price_asof=template.price_asof,
                              selection_asof=template.selection_asof,
                              execute_on=template.execute_on, source_asof=source,
                              sha256=pool_sha256(template.year, names, groups,
                                                 template.price_asof, template.selection_asof,
                                                 template.execute_on, source)))
            sources.append(AssetSources(year=template.year, asof=asof))
        rows, dates = tuple(pools), tuple(sources)
        cases.append(Case(id=case_id, pools=rows, pool_sources=dates, intervention_note=note,
                          sha256=case_hash(case_id, rows, dates, note)))
    return Cases(cases=tuple(cases))


@beartype
def baseline_schedules(root: Path) -> tuple[Schedule, Schedule]:
    folder = root / "experiments/v2-annual-repool"
    return (Schedule.model_validate_json((folder / "FIXED.json").read_bytes()),
            Schedule.model_validate_json((folder / "ANNUAL.json").read_bytes()))


@beartype
def load_cases(root: Path) -> Cases:
    from scripts.falsify_io import STUDY

    saved = Cases.model_validate_json((root / STUDY / "CASES.json").read_bytes())
    if saved != build_cases(*baseline_schedules(root)):
        raise ValueError("actual case definitions differ from the registered interventions")
    return saved


@beartype
def initialize_cases(root: Path) -> Cases:
    from scripts.falsify_io import STUDY

    path = root / STUDY / "CASES.json"
    if not path.exists():
        write(path, build_cases(*baseline_schedules(root)))
    return load_cases(root)


@beartype
def catalog(cases: Cases, fixed: Schedule, annual: Schedule) -> tuple[Account, ...]:
    if cases != build_cases(fixed, annual):
        raise ValueError("case catalog differs from independently reconstructed definitions")
    rows = []
    for case_id, costs, stage in (("A", (5,), "bridge"), ("D", (20,), "bridge"),
                                 ("B", (5, 20), "science"), ("C", (5, 20), "science"),
                                 ("E", (5, 20), "science")):
        case = next(c for c in cases.cases if c.id == case_id)
        schedule = fixed if case_id == "A" else annual
        for cost in costs:
            rows.append(Account.model_validate({"case": case, "cost_bps": cost, "stage": stage,
                        "envelope": Contract.model_validate({"mode": schedule.mode,
                            "cost_bps": cost, "stage": "bridge" if case_id == "A" else "science",
                            "schedule_sha256": schedule.sha256})}))
    return tuple(rows)
