from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

SYMBOLS = ("SPY", "TLT", "GLD", "SHY")
RISK = SYMBOLS[:3]


class Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class FeeSchedule(Model):
    commission_per_share: float
    commission_min: float
    platform_per_share: float
    platform_min: float
    settlement_per_share: float
    sec_rate: float
    sec_min: float
    nms_per_share: float


class Fees(Model):
    initial_cash: float
    notional: float
    max_positions: int
    slippage_bps: float
    fees: FeeSchedule


class ResearchPeriod(Model):
    model_config = ConfigDict(revalidate_instances="always")

    first_year: int = Field(default=2020, ge=2014, le=2025)
    last_year: int = Field(default=2025, ge=2014, le=2025)
    split_year: int = Field(default=2022, ge=2014, le=2024)

    @model_validator(mode="after")
    def ordered_years(self) -> "ResearchPeriod":
        if not self.first_year <= self.split_year < self.last_year:
            raise ValueError("research period requires first_year <= split_year < last_year")
        return self


class Settings(Model):
    fees: FeeSchedule
    initial_cash: float = 25000
    invested: float = 0.95
    slip_bps: float = Field(default=2, ge=0)
    tax: float = Field(default=0, ge=0, le=1)
    research_period: ResearchPeriod | None = None

    @property
    def effective_research_period(self) -> ResearchPeriod:
        if self.research_period is None:
            return ResearchPeriod()
        return ResearchPeriod.model_validate(self.research_period)


class Spec(Model):
    name: str
    layer: Literal["M", "F", "S", "T", "equal", "hold"]
    lookback: int
    top: int
    fixed: tuple[str, ...] = ()


class CashEvent(Model):
    symbol: str
    ex_date: date
    pay_date: date | None
    amount: float = Field(gt=0)
    source: str


class SplitEvent(Model):
    symbol: str
    day: date
    shares_after_per_before: float = Field(gt=0, allow_inf_nan=False)
    source: str


class Dataset(Model):
    symbol: str
    frame: pd.DataFrame
    dividends: list[CashEvent]
    splits: list[SplitEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_split_events(self) -> "Dataset":
        keys = [(event.symbol, event.day) for event in self.splits]
        if len(keys) != len(set(keys)) or any(event.symbol != self.symbol for event in self.splits):
            raise ValueError("duplicate split event or mismatched symbol")
        return self


class DataCheck(Model):
    symbol: str
    rows: int
    main_rows: int
    events: int
    max_price_relative_error: float
    trr_cumulative_difference: float
    trr_max_difference: float


class OrderRecord(Model):
    symbol: str
    day: date
    size: int
    price: float
    fee: float
    reason: str


class HoldingRecord(Model):
    day: date
    symbol: str
    size: int
    close: float


class DailyRecord(Model):
    day: date
    equity: float
    cash: float
    locked: float
    available: float
    positions: int
    gross_exposure: float


class Distribution(Model):
    symbol: str
    ex_date: date
    pay_date: date
    release_date: date
    qty: int
    per_share: float
    gross_amount: float
    net_amount: float


class Episode(Model):
    entry_day: date
    entry_price: float
    index: float = 0
    peak: float = 1


class ClosedTrade(Model):
    symbol: str
    opened: date
    closed: date
    gross_pnl: float
    net_pnl_ex_dividends: float
    commission: float


class Result(Model):
    spec: Spec
    orders: list[OrderRecord]
    daily: list[DailyRecord]
    holdings: list[HoldingRecord]
    distributions: list[Distribution]
    trades: list[ClosedTrade]
    native_total_return: float
    splits: list[SplitEvent] = Field(default_factory=list)


class BootstrapResult(Model):
    names: tuple[str, ...]
    block: int
    family_p: float
    adjusted_p: list[float]
    mean_bp: list[float]
    ci_low_bp: list[float]
    ci_high_bp: list[float]


class Metrics(Model):
    period: str
    days: int
    cagr: float
    total_return: float
    sharpe: float
    max_dd: float
    fees: float
    dividends: float
    orders: int
    turnover: float
    positive_years: int


class Trial(Model):
    name: str
    scenario: str
    status: str
    error: str = ""
    metrics: list[Metrics] = Field(default_factory=list)
    initial_cash: float
    final_equity: float | None = None
    final_locked: float | None = None


class Job(Model):
    spec: Spec
    settings: Settings
    scenario: str
    input_dir: str
    output_dir: str
