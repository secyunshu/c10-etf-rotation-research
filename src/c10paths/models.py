"""Finite typed score tables for the fixed 34-risk-asset path experiment."""

from datetime import date
from typing import Literal

from pydantic import ConfigDict, model_validator

from c10standard.models import RISK
from rotbt.models import Model

Variant = Literal["P01", "P02", "P03"]
VARIANTS: tuple[Variant, ...] = ("P01", "P02", "P03")


class Score(Model):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)
    rank: float
    r252: float
    eligible: bool


class Scores(Model):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)
    variant: Variant
    day: date
    assets: dict[str, Score]
    shy_r252: float

    @model_validator(mode="after")
    def fixed_universe(self) -> "Scores":
        if set(self.assets) != set(RISK):
            raise ValueError("score table must retain all 34 fixed risk assets")
        return self
