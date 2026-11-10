"""Cash and split evidence must conserve investor wealth without source rewrites."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rotbt.models import CashEvent, Dataset
from scripts import etf_group_cap_inputs as inputs
from scripts.etf_c10_2026_inputs import FIRST, OLD_LAST, sha


def detail(ex: str, pay: str, amount: str) -> dict[str, Any]:
    return {"ex_date": ex, "dividend_payable_date": pay, "statement": f"1股派息{amount}USD"}


def test_split_units_preserve_full_historical_cash_entitlement_and_gross_wealth() -> None:
    frame = pd.DataFrame(
        {
            "open": [120.0, 119.0, 39.666666666666664],
            "high": [121.0, 120.0, 40.0],
            "low": [119.0, 118.0, 39.0],
            "close": [120.0, 119.0, 119 / 3],
            "volume": 1000000.0,
            "turnover": 100000000.0,
        },
        index=pd.to_datetime(["2026-11-03", "2026-11-04", "2026-11-05"]),
    )
    actions = inputs.parse_actions(
        "SOXX",
        [
            {"ex_div_date": "2026-11-04", "per_cash_div": 1.0},
            {"ex_div_date": "2026-11-05", "split_base": 1, "split_ert": 3, "split_ratio": 1 / 3},
        ],
        [detail("2026/11/04", "2026/11/06", "0.3333333333333333")],
        first=date(2026, 11, 3),
        last=date(2026, 11, 5),
    )
    dataset = inputs.with_total_return("SOXX", frame, actions.cash, actions.splits)
    assert dataset.dividends[0].amount == 1
    assert actions.matches[0]["later_share_factor"] == 3
    assert dataset.frame.tr.to_list() == pytest.approx([100, 100, 100])
    assert 300 * dataset.frame.close.iloc[-1] == pytest.approx(100 * 119)


def test_consistent_ordinary_components_sum_but_ambiguous_duplicates_do_not() -> None:
    rehab = [{"ex_div_date": "2025-12-19", "per_cash_div": 0.7}]
    rows = [detail("2025/12/19", "2025/12/23", "0.5"), detail("2025/12/19", "2025/12/23", "0.2")]
    result = inputs.parse_actions("AGG", rehab, rows)
    assert result.cash[0].amount == pytest.approx(0.7)
    rows[1]["dividend_payable_date"] = "2025/12/24"
    with pytest.raises(ValueError, match="payment"):
        inputs.parse_actions("AGG", rehab, rows)
    with pytest.raises(ValueError, match="duplicate"):
        inputs.parse_actions("AGG", rehab, [rows[0], rows[0]])


def test_qual_conflict_requires_issuer_adjudication_before_retaining_old_cash() -> None:
    approved = CashEvent(
        symbol="QUAL",
        ex_date=date(2023, 12, 28),
        pay_date=date(2024, 1, 4),
        amount=0.09652,
        source="user-approved original canonical event",
    )
    with pytest.raises(ValueError, match="historical cash revision"):
        inputs.parse_actions(
            "QUAL",
            [
                {"ex_div_date": "2023-12-28", "per_cash_div": 0.19302},
                {"ex_div_date": "2026-11-05", "split_base": 1, "split_ert": 4, "split_ratio": 0.25},
            ],
            [detail("2023/12/28", "2024/01/04", "0.02413")],
            retained=[approved],
        )
    result = inputs.parse_actions(
        "QUAL",
        [
            {"ex_div_date": "2023-12-28", "per_cash_div": 0.19302},
            {"ex_div_date": "2026-11-05", "split_base": 1, "split_ert": 4, "split_ratio": 0.25},
        ],
        [detail("2023/12/28", "2024/01/04", "0.02413")],
        retained=[approved],
        approved_revisions={date(2023, 12, 28): 0.09652},
    )
    assert result.cash == [approved]
    assert result.matches[0]["raw_rehab_amount"] == 0.19302


def test_bil_user_approved_zero_ignores_only_exact_conflicting_event_and_keeps_other_income() -> (
    None
):
    ignored = {date(2022, 3, 1): (0.0215, date(2022, 3, 7))}
    details = [
        detail("2022/03/01", "2022/03/07", "0.0215"),
        detail("2022/06/01", "2022/06/07", "0.01"),
    ]
    rehab = [{"ex_div_date": "2022-06-01", "per_cash_div": 0.01}]
    with pytest.raises(ValueError, match="coverage"):
        inputs.parse_actions("BIL", rehab, details)
    actions = inputs.parse_actions("BIL", rehab, details, ignored_details=ignored)
    assert [e.ex_date for e in actions.cash] == [date(2022, 6, 1)]
    assert actions.cash[0].amount == 0.01
    details[0]["statement"] = "1股派息0.022USD"
    with pytest.raises(ValueError, match="approved zero"):
        inputs.parse_actions("BIL", rehab, details, ignored_details=ignored)
    details[0]["statement"] = "1股派息0.0215USD"
    rehab.append({"ex_div_date": "2022-03-01", "per_cash_div": 0.0215})
    with pytest.raises(ValueError, match="approved zero"):
        inputs.parse_actions("BIL", rehab, details, ignored_details=ignored)


@pytest.mark.parametrize("mutation", ["missingpay", "special", "spinoff", "split", "conflict"])
def test_unresolved_financial_evidence_blocks_instead_of_guessing(mutation: str) -> None:
    rehab: list[dict[str, Any]] = [{"ex_div_date": "2025-12-19", "per_cash_div": 0.7}]
    rows = [detail("2025/12/19", "2025/12/23", "0.7")]
    if mutation == "missingpay":
        rows[0]["dividend_payable_date"] = None
    elif mutation == "special":
        rehab[0]["special_dividend"] = 0.1
    elif mutation == "spinoff":
        rehab[0]["spin_off_ratio"] = 0.2
    elif mutation == "split":
        rehab[0]["split_base"] = 1
    else:
        rows[0]["statement"] = "1股派息0.72USD"
    with pytest.raises(ValueError, match="DATA_BLOCKED"):
        inputs.parse_actions("AGG", rehab, rows)


def test_physical_2025_cohort_removes_all_future_events_and_feature_rows() -> None:
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000000.0,
            "turnover": 100000000.0,
            "tr": 100.0,
        },
        index=pd.bdate_range("2024-01-01", "2026-11-06"),
    )
    event = CashEvent(
        symbol="AGG",
        ex_date=date(2026, 5, 1),
        pay_date=date(2026, 5, 6),
        amount=0.3,
        source="actual future cash",
    )
    data = Dataset(symbol="AGG", frame=frame, dividends=[event])
    clipped = inputs.clip_dataset(data, date(2025, 12, 31))
    assert clipped.frame.index.max() == pd.Timestamp("2025-12-31")
    assert not clipped.dividends and not clipped.splits
    assert inputs.features_for([clipped]).frames["AGG"].index.max().year == 2025


def test_raw_calendar_or_changed_overlap_cannot_be_silently_filled() -> None:
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000000.0,
            "turnover": 100000000.0,
        },
        index=pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05"]),
    )
    inputs.validate_frame("AGG", frame, date(2020, 12, 31), date(2021, 1, 5))
    with pytest.raises(ValueError, match="calendar"):
        inputs.validate_frame("AGG", frame.iloc[[0, 2]], date(2020, 12, 31), date(2021, 1, 5))
    old = Dataset(symbol="AGG", frame=frame.assign(tr=100.0), dividends=[])
    changed = old.model_copy(deep=True)
    changed.frame.loc[pd.Timestamp("2021-01-04"), "turnover"] = 100000001.0
    with pytest.raises(ValueError, match="overlap"):
        inputs.retain_history(old, changed)


def test_legacy_execution_source_retains_actual_turnover_lost_by_old_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "inputs_etf_legacy_resolved_20261107"
    folder.mkdir()
    rows = [
        {
            "time_key": "2020-12-31",
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
            "volume": 1000,
            "turnover": 99999.75,
        }
    ]
    (folder / "HYG_manifest.json").write_text(json.dumps({"pages": 1}))
    (folder / "HYG_raw_page0.json").write_text(json.dumps(rows))
    frame = pd.DataFrame(rows).set_index("time_key").drop(columns="turnover").assign(tr=100.0)
    frame.index = pd.DatetimeIndex(frame.index)

    def trusted_loader(root: Path, symbol: str) -> Dataset:
        return Dataset(symbol=symbol, frame=frame.copy(), dividends=[])

    monkeypatch.setattr(inputs, "load_legacy", trusted_loader)
    result = inputs.load_legacy_evidence(tmp_path, "HYG")
    assert result.frame.turnover.iloc[0] == 99999.75
    assert result.frame.turnover.iloc[0] != result.frame.close.iloc[0] * result.frame.volume.iloc[0]


def test_historical_loader_reads_only_physically_bounded_prices_and_refuses_revisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import exchange_calendars as xcals

    monkeypatch.setattr(inputs, "NAMES", ("AGG",))
    output = tmp_path / inputs.OUTPUT
    folder = output / "2025"
    folder.mkdir(parents=True)
    source = tmp_path / "frozen_source.txt"
    source.write_text("unchanged actual source")
    provenance = output / "SOURCE_PROVENANCE.json"
    provenance.write_text(json.dumps({"hashes": {str(source): sha(source)}}))
    days = xcals.get_calendar("XNYS").sessions_in_range(FIRST, OLD_LAST)
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000000.0,
            "turnover": 100000000.0,
            "tr": 100.0,
        },
        index=days.tz_localize(None),
    )
    frame.to_csv(folder / "AGG.csv.gz", index_label="date", compression="gzip")
    for name in ("CASH.json", "SPLITS.json"):
        (folder / name).write_text(json.dumps({"AGG": []}))
    manifest = inputs.Manifest(
        cohort="2025",
        first=FIRST,
        last=OLD_LAST,
        names=("AGG",),
        index_names={"AGG": "day"},
        provenance_sha256=sha(provenance),
        output_hashes={p.name: sha(p) for p in folder.iterdir()},
    )
    (folder / "MANIFEST.json").write_text(manifest.model_dump_json())
    future = output / "2026"
    future.mkdir()
    (future / "AGG.csv.gz").write_text("poison future data must never be parsed")
    prepared = inputs.load(tmp_path, "2025")
    assert prepared.last == date(2025, 12, 31)
    assert prepared.datasets[0].frame.index.name == "day"
    assert prepared.features.frames["AGG"].index.max() == pd.Timestamp("2025-12-31")
    source.write_text("source was revised")
    with pytest.raises(ValueError, match="source hash mismatch"):
        inputs.load(tmp_path, "2025")


def test_failed_build_keeps_distinct_failure_receipts_without_touching_old_inputs(
    tmp_path: Path,
) -> None:
    old = tmp_path / "immutable_old_source.txt"
    old.write_text("must stay identical")
    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            inputs.build(tmp_path)
    failures = list((tmp_path / inputs.RESEARCH / "data_qa").glob("*/FAILURE.json"))
    assert len(failures) == 2
    assert all(json.loads(p.read_text())["status"] == "DATA_BLOCKED" for p in failures)
    assert old.read_text() == "must stay identical"
    assert not (tmp_path / inputs.OUTPUT).exists()
