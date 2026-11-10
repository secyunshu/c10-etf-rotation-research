"""The user's top2 policy must reach the real CLI and generated report."""

import shutil
import sys
from pathlib import Path

import pytest

from c10standard import run
from c10standard.io import load_market, load_strategy
from c10standard.models import Contract, Receipt
from c10standard.report import (
    Verification,
    annual_rows,
    load_books,
    summary_row,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_and_python_entry_select_the_authorized_top_two_policy() -> None:
    assert load_strategy(ROOT).default_rule == "C11"
    assert Contract().rule == "C11"


@pytest.mark.parametrize("explicit_rule,expected", [(None, "C11"), ("C10", "C10")])
def test_cli_default_is_top_two_but_explicit_top_one_remains_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_rule: str | None, expected: str
) -> None:
    selected: list[str] = []

    def capture(contract, *args, **kwargs):
        selected.append(contract.rule)
        return Receipt(account=contract.key, hashes={}, max_differences={})

    market = load_market(ROOT)
    monkeypatch.setattr(run, "load_market", lambda *args: market)
    monkeypatch.setattr(run, "execute", capture)
    monkeypatch.setattr(run.native, "simulate", lambda *args: pytest.fail("pure CLI test"))
    argv = ["c10standard.run", "--root", str(ROOT), "--output", str(tmp_path / "output")]
    if explicit_rule:
        argv += ["--rule", explicit_rule]
    monkeypatch.setattr(sys, "argv", argv)
    run.main()
    assert selected == [expected]


def test_generated_report_focuses_top_two_and_preserves_all_adverse_references(
    tmp_path: Path,
) -> None:
    (tmp_path / "standard/results").mkdir(parents=True)
    shutil.copyfile(ROOT / "standard/STRATEGY.json", tmp_path / "standard/STRATEGY.json")
    books, hashes = load_books(ROOT)
    annual = [row for book in books for row in annual_rows(book)]
    summaries = [summary_row(book) for book in books]
    write_report(
        tmp_path,
        summaries,
        annual,
        Verification(
            accounts=36,
            source_hashes_checked=hashes,
            archived_metric_values_checked=1260,
            annual_rows=len(annual),
            summary_rows=len(summaries),
            monthly_rows=1416,
        ),
    )
    text = (tmp_path / "standard/results/RESULTS.md").read_text()
    assert "默认" in text and "C11 / K=2" in text
    assert "9.39%" in text and "25.25%" in text
    assert "C10" in text and "-4.43%" in text and "37.95%" in text
