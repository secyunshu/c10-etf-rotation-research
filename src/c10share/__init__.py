"""Portable reproduction of twelve frozen historical Backtrader accounts."""

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))
