import hashlib
import json
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import ConfigDict, Field

from ..models import Model, Result, Settings

CORE = ("SPY", "TLT", "GLD", "SHY")
EXPANDED = ("SPY", "QQQ", "EFA", "TLT", "IEF", "GLD", "DBC", "SHY")


class P2Spec(Model):
    model_config = ConfigDict(validate_default=True)
    label: str = ""
    family: str = "A"
    universe: Literal["core", "expanded"] = "core"
    score: Literal[
        "return", "volatility", "regression", "rank_return", "rank_volatility", "rank_regression"
    ] = "return"
    windows: tuple[int, ...] = (126,)
    top: int = 1
    frequency: Literal["weekly", "monthly", "hold"] = "weekly"
    filter: Literal["none", "cash", "sma", "both"] = "cash"
    allocation: Literal["equal", "inverse", "minvar"] = "equal"
    cap: float | None = None
    vol_target: float = 0
    stop: Literal["none", "trailing", "volatility"] = "none"
    stop_value: float = 0
    take_profit: float = 0
    band: float = 0
    min_days: int = 0
    hurdle: float = 0
    hysteresis: bool = False
    bucket_caps: bool = False
    fixed: tuple[str, ...] = ()
    phase_shift: bool = False
    delay_days: int = 0
    removed: tuple[str, ...] = ()

    @property
    def sid(self) -> str:
        encoded = json.dumps(self.model_dump(exclude={"label", "family"}), sort_keys=True)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    @property
    def complexity(self) -> int:
        return sum(
            (
                self.score != "return" or len(self.windows) > 1,
                self.filter in ("sma", "both"),
                self.vol_target > 0,
                self.allocation != "equal",
                self.stop != "none",
                self.take_profit > 0,
                self.band > 0,
                self.min_days > 0,
                self.hurdle > 0,
            )
        )


class Alias(Model):
    label: str
    family: str
    spec_id: str


class Catalog(Model):
    specs: list[P2Spec]
    aliases: list[Alias]
    neighbors: dict[str, tuple[str, ...]]


class FeatureStore(Model):
    frames: dict[str, pd.DataFrame]


class HeldAsset(Model):
    symbol: str
    shares: int
    age: int


class SignalContext(Model):
    day: date
    holdings: tuple[HeldAsset, ...] = ()
    blocked: tuple[str, ...] = ()


class Allocation(Model):
    symbol: str
    weight: float


class TargetSignal(Model):
    allocations: tuple[Allocation, ...]
    numerical_failure: bool = False
    error: str = ""


class Fault(Model):
    day: date
    spec_id: str
    reason: str


class AnnualChoice(Model):
    year: int
    spec: P2Spec | None
    cutoff: date
    family: str = "joint"


class Simulation(Model):
    spec: P2Spec
    account: Result
    faults: list[Fault] = Field(default_factory=list)
    elapsed_cpu: float


class TrainingRecord(Model):
    spec_id: str
    cutoff: date
    fold_returns: tuple[float, ...]
    fold_drawdowns: tuple[float, ...]
    fold_excess_cagrs: tuple[float, ...]
    training_return: float
    training_drawdown: float
    training_turnover: float
    valid: bool = True
    first_failure: date | None = None


class Selection(Model):
    cutoff: date
    family: str
    chosen_id: str | None
    scores: dict[str, float]
    basic_eligible: tuple[str, ...]
    neighbor_eligible: tuple[str, ...]


class TrialTask(Model):
    key: str
    spec: P2Spec
    settings: Settings
    start: date
    end: date
    stage: str
    scenario: str
    output_dir: str
    choices: tuple[AnnualChoice, ...] = ()
    engine: Literal["phase2", "phase1"] = "phase2"


class RunSummary(Model):
    key: str
    spec_id: str
    stage: str
    scenario: str
    start: date
    end: date
    status: str
    cpu_seconds: float
    final_equity: float | None = None
    cagr: float | None = None
    total_return: float | None = None
    max_dd: float | None = None
    sharpe: float | None = None
    turnover: float | None = None
    fees: float | None = None
    faults: list[Fault] = Field(default_factory=list)
    error: str = ""
    cached: bool = False
