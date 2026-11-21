import gzip
from pathlib import Path

import pytest

from c10standard import native
from c10standard.io import ROOT
from c10standard.models import Artifact, Contract, Prepared
from rotbt.models import Settings
from scripts.current_standard_ranked import verify_account
from scripts.v23_rank_weights import Context, context


@pytest.fixture(scope="module")
def history() -> tuple[Context, Artifact]:
    source = ROOT / "standard/v2_3_1/reference/HIST/C11/ACCOUNT.json.gz"
    return context("HIST"), Artifact.model_validate_json(gzip.decompress(source.read_bytes()))


def test_retry_resumes_an_account_directory_left_empty_by_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: tuple[Context, Artifact],
) -> None:
    ctx, book = history
    folder = tmp_path / "HIST/C11"
    folder.mkdir(parents=True)

    def simulate(_market: Prepared, _contract: Contract, _settings: Settings, out: Path) -> None:
        native.write(out / "ARTIFACT.json", book)

    monkeypatch.setattr(native, "simulate", simulate)
    verify_account(ctx, "HIST", "C11", tmp_path, True)
    assert Artifact.model_validate_json((folder / "ARTIFACT.json").read_bytes()) == book


def test_native_fault_keeps_attempt_for_diagnosis_and_allows_successful_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: tuple[Context, Artifact],
) -> None:
    ctx, book = history
    completed = tmp_path / "HIST/C11/ARTIFACT.json"
    attempts: list[Path] = []

    def simulate(_market: Prepared, _contract: Contract, _settings: Settings, out: Path) -> None:
        attempts.append(out)
        native.write(out / "ARTIFACT.json", book)
        if len(attempts) == 1:
            raise ValueError("late native fault")

    monkeypatch.setattr(native, "simulate", simulate)
    with pytest.raises(ValueError, match="late native fault"):
        verify_account(ctx, "HIST", "C11", tmp_path, True)
    assert not completed.exists()
    assert (attempts[0] / "ARTIFACT.json").is_file()

    verify_account(ctx, "HIST", "C11", tmp_path, True)
    cached = (completed.read_bytes(), completed.stat().st_mtime_ns)
    verify_account(ctx, "HIST", "C11", tmp_path, True)
    assert len(attempts) == 2
    assert (completed.read_bytes(), completed.stat().st_mtime_ns) == cached
    assert (attempts[0] / "ARTIFACT.json").is_file()


def test_existing_success_from_a_second_writer_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: tuple[Context, Artifact],
) -> None:
    ctx, book = history
    completed = tmp_path / "HIST/C11/ARTIFACT.json"
    published: list[tuple[bytes, int]] = []

    def simulate(_market: Prepared, _contract: Contract, _settings: Settings, out: Path) -> None:
        native.write(completed, book)
        published.append((completed.read_bytes(), completed.stat().st_mtime_ns))
        native.write(out / "ARTIFACT.json", book)

    monkeypatch.setattr(native, "simulate", simulate)
    verify_account(ctx, "HIST", "C11", tmp_path, True)
    assert (completed.read_bytes(), completed.stat().st_mtime_ns) == published[0]
