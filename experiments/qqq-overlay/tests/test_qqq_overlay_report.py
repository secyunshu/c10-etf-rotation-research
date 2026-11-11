"""Equal sample windows and actual strategy identity are essential to comparison."""

from pathlib import Path

import pytest

from c10share.report import read_csv
from c10standard.io import ROOT
from c10standard.models import Contract
from rotbt.models import DailyRecord
from scripts.qqq_overlay_report import validate_history


def test_historical_report_rejects_missing_or_duplicate_sessions_and_nonflat_terminal() -> None:
    contract = Contract()
    rows = read_csv(ROOT / "standard/reference" / contract.key / "daily.csv.gz", DailyRecord)
    validate_history(contract, rows)
    for corrupt in (rows[1:], rows + rows[-1:], rows[:-1]):
        with pytest.raises(ValueError, match="calendar"):
            validate_history(contract, corrupt)
    corrupt = rows[:-1] + (rows[-1].model_copy(update={"positions": 1}),)
    with pytest.raises(ValueError, match="liquidat"):
        validate_history(contract, corrupt)


def test_2026_is_not_a_permitted_reporting_cohort() -> None:
    with pytest.raises(ValueError, match="historical"):
        validate_history(Contract(cohort="reset2026"), ())


def test_other_signal_cannot_be_relabelled_as_overlay(tmp_path: Path) -> None:
    from scripts.qqq_overlay_report import read_book
    from scripts.qqq_overlay_run import Binding, Trial

    binding = Binding(trial=Trial(variant="BASE", contract=Contract()), source_lock_sha256="a" * 64)
    (tmp_path / "BINDING.json").write_text(binding.model_dump_json())
    with pytest.raises(ValueError, match="identity"):
        read_book("OVERLAY", Contract(), tmp_path, ROOT, science=True)
