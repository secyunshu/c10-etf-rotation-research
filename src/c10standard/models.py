"""Fixed publication contracts; no parameter-search or trading interface."""

from datetime import date, datetime
from typing import Literal

from pydantic import ConfigDict, model_validator

from rotbt.models import Dataset, FeeSchedule, Model, Result
from rotbt.phase2.models import FeatureStore

IDENTITY = "C10_HISTORICAL_GROUP_CAP_V2"
FIRST, HISTORICAL_LAST, LAST = date(2020, 12, 31), date(2025, 12, 31), date(2026, 11, 6)
RISK = tuple(
    "AGG BIL EWJ GLD HYG IEF IEMG IJR INDA MCHI PDBC QQQ QUAL SLV SOXX SPY TIP TLT "
    "USMV VCIT VEA VGT VIG VNQ VTI XLB XLC XLE XLF XLI XLP XLU XLV XLY".split()
)
NAMES = tuple(sorted((*RISK, "SHY")))
SOURCE_GROUPS = {
    "us_broad_style": tuple("VTI VOO QQQ IJR SCHD SPHQ USMV SPMO SPY VIG QUAL".split()),
    "technology_semiconductors": ("VGT", "SMH", "SOXX"),
    "financials": ("XLF",),
    "healthcare": ("XLV",),
    "energy": ("XLE",),
    "industrials": ("XLI",),
    "utilities": ("XLU",),
    "consumer_discretionary": ("XLY",),
    "consumer_staples": ("XLP",),
    "communications": ("XLC",),
    "materials": ("XLB",),
    "real_estate": ("VNQ",),
    "developed_foreign": ("VEA", "EWJ"),
    "emerging_foreign": ("IEMG", "INDA", "MCHI"),
    "short_treasury": ("SGOV", "VGSH", "BIL"),
    "nominal_treasury_duration": ("IEF", "TLT"),
    "inflation_bonds": ("VTIP", "TIP"),
    "investment_grade_bonds": ("BND", "VCIT", "AGG"),
    "high_yield_credit": ("USHY", "HYG"),
    "precious_metals": ("GLD", "SLV"),
    "broad_commodities": ("PDBC",),
}
GROUPS = {group: tuple(s for s in members if s in RISK) for group, members in SOURCE_GROUPS.items()}
Rule = Literal["C10", "C11", "C00", "SPY_BH", "QQQ_BH", "SPY60_IEF40"]
RULES: tuple[Rule, ...] = ("C10", "C11", "C00", "SPY_BH", "QQQ_BH", "SPY60_IEF40")
Cohort = Literal["historical2022_2025", "reset2026", "continuous2022_2026"]
COHORTS: tuple[Cohort, ...] = ("historical2022_2025", "reset2026", "continuous2022_2026")
BOOKS = ("daily", "orders", "holdings", "dividends", "trades")


class Contract(Model):
    model_config = ConfigDict(frozen=True)
    rule: Rule = "C11"
    cost_bps: Literal[5, 20] = 5
    cohort: Cohort = "historical2022_2025"

    @property
    def key(self) -> str:
        return f"{self.rule}_{self.cost_bps}bp_{self.cohort}"

    @property
    def cost(self) -> Literal[5, 20]:
        return self.cost_bps

    @property
    def start(self) -> date:
        return date(2026, 1, 2) if self.cohort == "reset2026" else date(2022, 1, 3)

    @property
    def through(self) -> date:
        return HISTORICAL_LAST if self.terminal else LAST

    @property
    def terminal(self) -> bool:
        return self.cohort == "historical2022_2025"

    @property
    def mode(self) -> Literal["bridge", "reset2026", "continuous2022"]:
        return (
            "bridge"
            if self.terminal
            else "reset2026"
            if self.cohort == "reset2026"
            else "continuous2022"
        )


class Strategy(Model):
    model_config = ConfigDict(extra="ignore", frozen=True)
    version: Literal[2]
    id: Literal["C10_HISTORICAL_GROUP_CAP_V2"]
    default_rule: Literal["C11"]
    group_cap: Literal[True]
    risk_names: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]
    fees: FeeSchedule
    defender: Literal["SHY"] = "SHY"
    lookback: Literal[252] = 252
    frequency: Literal["monthly"] = "monthly"
    allowed_top_k: tuple[Literal[1], Literal[2], Literal[3]] = (1, 2, 3)
    minimum_observations: Literal[253] = 253
    positive_volume_days: Literal[20] = 20
    minimum_price_usd: float = 5.0
    adv20_min_usd: float = 1e7
    initial_cash: float = 50000.0
    predeployment_risk: float = 0.9
    deployment: float = 0.95
    tax: float = 0.0
    costs_bps: tuple[Literal[5], Literal[20]] = (5, 20)
    historical_period: tuple[date, date] = (date(2022, 1, 3), HISTORICAL_LAST)
    ytd_period: tuple[date, date] = (date(2026, 1, 2), LAST)
    historical_terminal: Literal["last_day_open_liquidation"] = "last_day_open_liquidation"
    ytd_terminal: Literal["close_mark_to_market"] = "close_mark_to_market"
    continuous_2026_denominator: Literal["own2025-12-31closeNAV_without_prior_liquidation"] = (
        "own2025-12-31closeNAV_without_prior_liquidation"
    )
    fresh_OOS: Literal[False] = False

    @model_validator(mode="after")
    def fixed_contract(self) -> "Strategy":
        if self.risk_names != RISK or self.groups != GROUPS:
            raise ValueError("fixed historical pool or 21 economic groups changed")
        if (
            self.minimum_price_usd,
            self.adv20_min_usd,
            self.initial_cash,
            self.predeployment_risk,
            self.deployment,
            self.tax,
            self.historical_period,
            self.ytd_period,
        ) != (
            5.0,
            1e7,
            50000.0,
            0.9,
            0.95,
            0.0,
            (date(2022, 1, 3), HISTORICAL_LAST),
            (date(2026, 1, 2), LAST),
        ):
            raise ValueError("fixed execution/qualification budget or periods changed")
        if tuple(self.fees.model_dump().values()) != (
            0.0049,
            0.99,
            0.005,
            1.0,
            0.003,
            0.0000206,
            0.01,
            0.000003,
        ):
            raise ValueError("fixed fee schedule changed")
        return self


class MarketManifest(Model):
    source_manifest_sha256: str = ""
    id: Literal["C10_HISTORICAL_GROUP_CAP_V2"]
    cohort: Literal["2025", "2026"]
    first: date
    last: date
    names: tuple[str, ...]
    output_hashes: dict[str, str]

    @model_validator(mode="after")
    def fixed_identity(self) -> "MarketManifest":
        last = HISTORICAL_LAST if self.cohort == "2025" else LAST
        expected = {f"{s}.csv.gz" for s in NAMES} | {"CASH.json", "SPLITS.json"}
        if (self.first, self.last, self.names) != (FIRST, last, NAMES) or set(
            self.output_hashes
        ) != expected:
            raise ValueError("fixed V2 market membership or physical cutoff changed")
        return self


class Prepared(Model):
    datasets: list[Dataset]
    features: FeatureStore
    strategy: Strategy
    risk_names: tuple[str, ...]
    first: date
    last: date


class Account(Model):
    key: str
    rule: Rule
    cost_bps: Literal[5, 20]
    cohort: Cohort
    reference_files: dict[str, str]
    upstream_account_id: str = ""
    upstream_native_evidence_sha256: str = ""

    @property
    def contract(self) -> Contract:
        return Contract(rule=self.rule, cost_bps=self.cost_bps, cohort=self.cohort)

    @model_validator(mode="after")
    def reference_contract(self) -> "Account":
        expected = {f"{n}.csv.gz" for n in BOOKS} | {
            "targets.json",
            "native_total_return.json",
            "metrics.json" if self.contract.terminal else "ytd.json",
        }
        optional = {"native_returns.json", "open_trades.json"}
        if (
            self.key != f"{self.rule}_{self.cost_bps}bp_{self.cohort}"
            or not expected <= self.reference_files.keys()
            or set(self.reference_files) - expected - optional
        ):
            raise ValueError("fixed reference account identity or filenames changed")
        return self


class TargetRecord(Model):
    model_config = ConfigDict(allow_inf_nan=False)
    day: date
    q: dict[str, float]
    reason: str


class NativeReturn(Model):
    day: date
    value: float


class OpenTrade(Model):
    symbol: str
    size: int
    price: float
    pnl: float
    commission: float


class Artifact(Model):
    contract: Contract
    account: Result
    targets: list[TargetRecord]
    native_returns: list[NativeReturn]
    open_trades: list[OpenTrade]


class NativeScalar(Model):
    native_total_return: float


class Attempt(Model):
    standard_id: Literal["C10_HISTORICAL_GROUP_CAP_V2"]
    event: Literal["started", "native_completed", "finished", "failed", "revalidated"]
    account: str
    attempt_id: str
    utc: datetime
    output: str
    error: str = ""


class Receipt(Model):
    status: Literal["PASS"] = "PASS"
    account: str
    hashes: dict[str, str]
    max_differences: dict[str, float]
