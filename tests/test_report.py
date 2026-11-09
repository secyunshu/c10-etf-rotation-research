import hashlib
import json
import shutil
from datetime import date
from math import prod
from pathlib import Path

import pytest

from c10share.report import (
    Account,
    NavPoint,
    annual_stats,
    drawdown_detail,
    monthly_stats,
    period_stats,
    read_accounts,
    verify_account,
)


def sample() -> Account:
    return Account(
        name="C10_5bp",
        rule="C10",
        cost_bps=5,
        initial_cash=50_000,
        split_year=2022,
        daily=(
            NavPoint(day=date(2022, 1, 3), equity=45_000),
            NavPoint(day=date(2022, 1, 31), equity=55_000),
            NavPoint(day=date(2022, 12, 30), equity=60_000),
            NavPoint(day=date(2023, 1, 3), equity=54_000),
            NavPoint(day=date(2023, 1, 31), equity=66_000),
            NavPoint(day=date(2023, 12, 29), equity=72_000),
        ),
    )


def test_annual_return_uses_prior_close_and_initial_capital() -> None:
    annual = annual_stats(sample())
    assert [row.opening_equity for row in annual] == [50_000, 60_000]
    assert [row.total_return for row in annual] == pytest.approx([0.2, 0.2])
    assert [row.profit for row in annual] == [10_000, 12_000]
    assert annual[0].total_return != annual[0].cagr


def test_annual_returns_compound_to_continuous_account_total() -> None:
    account = sample()
    full = period_stats(account, 0, len(account.daily) - 1, "full")
    assert prod(1 + row.total_return for row in annual_stats(account)) - 1 == pytest.approx(
        full.total_return
    )
    assert full.end_equity == 72_000


def test_annual_drawdown_includes_loss_from_opening_capital() -> None:
    assert [row.max_dd for row in annual_stats(sample())] == pytest.approx([0.1, 0.1])


def test_monthly_returns_use_previous_month_close_and_compound() -> None:
    account = sample()
    months = monthly_stats(account)
    assert [row.month for row in months] == ["2022-01", "2022-12", "2023-01", "2023-12"]
    assert [row.opening_equity for row in months] == [50_000, 55_000, 60_000, 66_000]
    assert [row.total_return for row in months] == pytest.approx(
        [0.1, 60 / 55 - 1, 0.1, 72 / 66 - 1]
    )
    assert prod(1 + row.total_return for row in months) == pytest.approx(72_000 / 50_000)


def test_drawdown_peak_trough_recovery_includes_initial_loss() -> None:
    detail = drawdown_detail(sample())
    assert detail.peak_date is None  # Capital before the first closing NAV.
    assert detail.trough_date == date(2022, 1, 3)
    assert detail.recovery_date == date(2022, 1, 31)
    assert detail.max_dd == pytest.approx(0.1)


def test_invalid_nav_order_and_future_holdout_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sample().model_copy(update={"daily": tuple(reversed(sample().daily))}).validate_history()
    with pytest.raises(ValueError, match="2025"):
        sample().model_copy(
            update={"daily": (NavPoint(day=date(2026, 1, 2), equity=1.0),)}
        ).validate_history()


def test_all_published_accounts_match_archived_metrics_and_common_period() -> None:
    accounts, hash_checks = read_accounts(Path(__file__).resolve().parents[1])
    assert len(accounts) == 12
    assert hash_checks == 108
    assert sum(verify_account(account) for account in accounts) == 840
    assert sum(len(annual_stats(account)) for account in accounts) == 48
    assert sum(len(monthly_stats(account)) for account in accounts) == 576
    assert all(account.initial_cash == 50_000 for account in accounts)


def test_changed_reference_return_fails_independent_verification() -> None:
    accounts, _ = read_accounts(Path(__file__).resolve().parents[1])
    account = accounts[0]
    changed = account.reference[0].model_copy(
        update={"total_return": account.reference[0].total_return + 0.01}
    )
    corrupt = account.model_copy(update={"reference": (changed,) + account.reference[1:]})
    with pytest.raises(ValueError, match="metric mismatch.*total_return"):
        verify_account(corrupt)


@pytest.mark.parametrize("mutation", ("rule", "research_period"))
def test_relabelled_account_or_changed_window_cannot_publish_valid_books(
    tmp_path: Path, mutation: str
) -> None:
    source = Path(__file__).resolve().parents[1] / "data"
    target = tmp_path / "data"
    shutil.copytree(source / "reference", target / "reference")
    manifest = json.loads((source / "ACCOUNTS.json").read_text())
    account = next(row for row in manifest if row["name"] == "C11_5bp")
    if mutation == "rule":
        account["rule"] = "C10"
    else:
        task_path = target / "reference/C11_5bp/task.json"
        task = json.loads(task_path.read_text())
        task["settings"]["research_period"]["first_year"] = 2021
        task_path.write_text(json.dumps(task))
        account["reference_files"]["task.json"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
    (target / "ACCOUNTS.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="account identity mismatch|frozen task changed"):
        read_accounts(tmp_path)
