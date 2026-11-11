"""Scoped pool substitution; preserve both original signal implementations."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from threading import Lock

from beartype import beartype

from scripts import annual_run, annual_signals
from scripts.annual_models import Binding, Contract, Pool, Prepared
from scripts.falsify_models import Account

_SCOPE = Lock()


@contextmanager
@beartype
def case_scope(account: Account, source_sha256: str) -> Iterator[None]:
    Account.model_validate(account)
    if not _SCOPE.acquire(blocking=False):
        raise ValueError("counterfactual scope cannot be nested or concurrent")
    original_pool, original_binding = annual_signals.active_pool, annual_run.binding

    def active_pool(contract: Contract, prepared: Prepared, day: date) -> Pool:
        if contract != account.envelope:
            raise ValueError("case scope received a different execution envelope")
        reference = original_pool(contract, prepared, day)
        actual = next(p for p in account.case.pools if p.year == reference.year)
        if not set(actual.risk_names) <= set(annual_signals.feed_names(contract, prepared)):
            raise ValueError("actual pool exceeds the execution feed union")
        return actual

    def binding(contract: Contract, root: Path) -> Binding:
        if contract != account.envelope:
            raise ValueError("case scope binding received a different execution envelope")
        return Binding(contract=contract, source_sha256=source_sha256)

    try:
        annual_signals.active_pool, annual_run.binding = active_pool, binding
        yield
    finally:
        annual_signals.active_pool, annual_run.binding = original_pool, original_binding
        _SCOPE.release()
