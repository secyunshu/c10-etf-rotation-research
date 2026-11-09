"""Typed boundaries for the fixed publication, not a parameter-search interface."""

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from rotbt.models import Dataset, Model, Settings
from rotbt.phase2.models import FeatureStore, P2Spec

RuleName = Literal["C10", "C11", "C00", "SPY_BH", "QQQ_BH", "SPY60_IEF40"]
RULES: tuple[RuleName, ...] = ("C10", "C11", "C00", "SPY_BH", "QQQ_BH", "SPY60_IEF40")
FIRST, LAST = date(2020, 12, 31), date(2025, 12, 31)
START = date(2022, 1, 3)
RISK = tuple(
    "BND EWJ GLD IEF IEMG IJR INDA MCHI PDBC QQQ SCHD SGOV SLV SMH SPHQ SPMO "
    "TLT USHY USMV VCIT VEA VGSH VGT VNQ VOO VTI VTIP XLB XLC XLE XLF XLI XLP "
    "XLU XLV XLY".split()
)
GROUPS = (
    tuple(
        "EWJ IEMG IJR INDA MCHI QQQ SCHD SMH SPHQ SPMO USMV VEA VGT VNQ VOO VTI XLB "
        "XLC XLE XLF XLI XLP XLU XLV XLY".split()
    ),
    tuple("BND IEF SGOV TLT USHY VCIT VGSH VTIP".split()),
    ("GLD", "SLV"),
    ("PDBC",),
)
BOOKS = ("daily", "orders", "holdings", "dividends", "trades")
REFERENCE_FILES = {f"{book}.csv.gz" for book in BOOKS} | {
    "task.json",
    "targets.json",
    "metrics.json",
    "verified.json",
}


class MarketManifest(Model):
    scope: Literal["modern38"]
    identity: Literal["MODERN38_C10_BENCHMARK_20201231"]
    risk_names: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    first: date
    last: date
    output_hashes: dict[str, str]
    original_manifest_sha256: str

    @model_validator(mode="after")
    def frozen_contract(self) -> "MarketManifest":
        expected = {f"{s}.csv.gz" for s in (*RISK, "SHY", "SPY")} | {
            "CASH.json",
            "SPLITS.json",
            "POTENTIAL_AMBIGUITIES.json",
        }
        if (
            self.risk_names != RISK
            or self.groups != GROUPS
            or (self.first, self.last) != (FIRST, LAST)
            or set(self.output_hashes) != expected
        ):
            raise ValueError("frozen market identity, calendar or files changed")
        return self


class Account(Model):
    name: str
    rule: RuleName
    cost_bps: Literal[5, 20]
    original_key: str
    original_reference_folder: str
    original_task_sha256: str
    reference_files: dict[str, str]

    @model_validator(mode="after")
    def fixed_account(self) -> "Account":
        if self.name != f"{self.rule}_{self.cost_bps}bp":
            raise ValueError("account identity mismatch")
        if set(self.reference_files) != REFERENCE_FILES:
            raise ValueError("reference files changed")
        return self


class Task(Model):
    key: str
    spec: P2Spec
    settings: Settings
    start: date
    end: date
    stage: str
    scenario: str
    choices: tuple[()] = ()
    engine: Literal["phase2"]


class Prepared(Model):
    datasets: list[Dataset]
    features: FeatureStore
    manifest: MarketManifest


class TargetRecord(Model):
    day: date
    q: dict[str, Annotated[float, Field(allow_inf_nan=False)]]
    reason: Literal["initial", "rebalance"]


class Verified(Model):
    status: Literal["PASS"]
    native_return: float = Field(allow_inf_nan=False)
    equity_return: float = Field(allow_inf_nan=False)
    days: int
    orders: int
    cash_events: int
    min_available: float
    final_locked: float
    terminal_positions: int


class ReferenceCheck(Model):
    account: str
    status: Literal["PASS"] = "PASS"
    days: int
    orders: int
    targets: int
    max_metric_difference: float
    max_target_difference: float


class RunComparison(Model):
    account: str
    status: Literal["PASS"] = "PASS"
    absolute_tolerance: float = 1e-10
    max_abs_difference: dict[str, float]


class AttemptRecord(Model):
    event: Literal["started", "finished", "failed"]
    account: str
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""
