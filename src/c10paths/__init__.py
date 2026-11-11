"""Three preregistered path signals; native accounting remains in c10standard."""

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))
