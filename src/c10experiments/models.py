"""Typed identities outside the C10/C11/C00 execution shells."""

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import ConfigDict, model_validator

from c10standard.models import BOOKS, Cohort, Contract
from rotbt.models import Model

Variant = Literal["W13612", "E01", "E02", "E03", "E04", "E05", "E06"]
VARIANTS: tuple[Variant, ...] = ("W13612", "E01", "E02", "E03", "E04", "E05", "E06")
RELEASE_ID = "TOP2_DEFAULT_AND_EXPERIMENTS_20261111"


class Experiment(Model):
    model_config = ConfigDict(frozen=True)
    key: str
    variant: Variant
    k: Literal[1, 2, 3]
    cost_bps: Literal[5, 20]
    cohort: Cohort
    reference_path: str
    reference_hashes: dict[str, str]

    @property
    def contract(self) -> Contract:
        rule: Literal["C10", "C11", "C00"] = (
            "C10" if self.k == 1 else "C11" if self.k == 2 else "C00"
        )
        return Contract(rule=rule, cost_bps=self.cost_bps, cohort=self.cohort)

    @property
    def original_trial(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "key": self.key,
            "k": self.k,
            "cost_bps": self.cost_bps,
            "cohort": self.cohort,
            "kind": "science",
        }
        if self.variant == "W13612":
            values["score"] = "calendar_13612w"
        else:
            values["variant"] = self.variant
        return values

    @model_validator(mode="after")
    def fixed_identity(self) -> "Experiment":
        family = "calendar13612w" if self.variant == "W13612" else "top2-momentum-variants"
        required = {f"{b}.csv.gz" for b in BOOKS} | {
            "TRIAL.json",
            "TARGETS.json",
            "NATIVE_TOTAL_RETURN.json",
            "METRICS.json" if self.contract.terminal else "YTD.json",
        }
        if (
            (self.variant != "W13612" and self.k != 2)
            or self.key != f"{self.variant}_K{self.k}_{self.cost_bps}bp_{self.cohort}"
            or self.reference_path != f"experiments/{family}/reference/{self.key}"
            or set(self.reference_hashes) != required
        ):
            raise ValueError("fixed experiment identity/reference path or filenames changed")
        if (
            PurePosixPath(self.reference_path).is_absolute()
            or ".." in PurePosixPath(self.reference_path).parts
        ):
            raise ValueError("unsafe experiment reference path")
        return self


class Score(Model):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)
    rank: float
    r252: float
    gate: float
    above_trend: bool


class Attempt(Model):
    release_id: Literal["TOP2_DEFAULT_AND_EXPERIMENTS_20261111"]
    event: Literal["started", "native_completed", "finished", "failed", "revalidated"]
    account: str
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""


class Binding(Model):
    experiment: Experiment
    contract: Contract
    native_path: str
    source_hashes: dict[str, str]


class ExperimentReceipt(Model):
    status: Literal["PASS"] = "PASS"
    experiment: Experiment
    native_path: str
    binding_sha256: str
    artifact_sha256: str
    standard_receipt_sha256: str
    hashes: dict[str, str]
