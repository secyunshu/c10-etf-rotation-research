"""YTD must use the same continuous account's prior close and include firstday losses."""

from datetime import date
from types import SimpleNamespace

import pytest

from rotbt.models import DailyRecord
from scripts.ytd_annual_report import summarize


def test_science_report_accepts_its_own_pool_binding_not_the_historical_bridge(
    tmp_path, monkeypatch
) -> None:
    from scripts import ytd_annual_report as report
    from scripts.ytd_annual_models import Binding, Contract

    science, bridge, source = "a" * 64, "b" * 64, "c" * 64
    contract = Contract(role="ANNUAL", cost_bps=5, stage="science", schedule_sha256=science)
    folder = tmp_path / "native" / contract.key
    folder.mkdir(parents=True)
    binding = Binding(contract=contract, source_sha256=source)
    (folder / "BINDING.json").write_text(binding.model_dump_json())
    (folder / "REFERENCE.json").write_text('{"hashes":{"BINDING.json":"unused"}}')
    monkeypatch.setattr(report, "verify_lock", lambda root: source)
    monkeypatch.setattr(report, "load_prepared", lambda role, stage, root:
                        SimpleNamespace(schedule=SimpleNamespace(
                            sha256=science if stage == "science" else bridge)))
    monkeypatch.setattr(report, "verify_hashes", lambda folder, hashes: None)

    class AcceptedBinding(Exception):
        pass

    def stop_before_reading_account(*args):
        raise AcceptedBinding

    monkeypatch.setattr(report, "read_csv", stop_before_reading_account)
    # A valid continuous account must reach accounting, despite a different bridge pool.
    with pytest.raises(AcceptedBinding):
        report.build(tmp_path, tmp_path / "native", tmp_path / "report")


def row(day: date, value: float) -> DailyRecord:
    return DailyRecord(
        day=day, equity=value, cash=value, available=value, locked=0, positions=0, gross_exposure=0
    )


def test_ytd_uses_own_december31_value_and_includes_initial_drawdown() -> None:
    days = [
        row(date(2025, 12, 31), 150000),
        row(date(2026, 1, 2), 120000),
        row(date(2026, 11, 10), 165000),
    ]
    stats = summarize(days, [], [], "2026YTD")
    assert stats.opening_equity == 150000
    assert stats.total_return == pytest.approx(0.1)
    assert stats.max_drawdown == pytest.approx(0.2)
    assert stats.cagr is None
    with pytest.raises(ValueError, match="December31"):
        summarize(days[1:], [], [], "2026YTD")


def test_annual_metrics_do_not_reset_capital_or_extrapolate_ytd() -> None:
    days = [
        row(date(2022, 1, 3), 45000),
        row(date(2022, 12, 30), 60000),
        row(date(2023, 1, 3), 54000),
        row(date(2023, 12, 29), 66000),
    ]
    first = summarize(days, [], [], "2022")
    second = summarize(days, [], [], "2023")
    assert first.total_return == pytest.approx(0.2)
    assert first.max_drawdown == pytest.approx(0.1)
    assert second.opening_equity == 60000
    assert second.total_return == pytest.approx(0.1)
