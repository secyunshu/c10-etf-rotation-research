"""Adapter safety contracts; every native invocation is replaced by a failing stub."""

import json
from datetime import date
from pathlib import Path

import pytest

from rotbt.phase2.models import FeatureStore
from scripts import etf_c10_2026_run as native
from scripts import etf_group_cap_run as run
from scripts.etf_group_cap_targets import GroupMap, load_roles

ROOT = Path(__file__).resolve().parents[1]


def test_execution_shell_uses_finite_dates_and_only_historical_accounts_liquidate() -> None:
    for role in load_roles(ROOT):
        contract = run.contract_for(role)
        assert (contract.start, contract.through) == (role.start, role.through)
        assert contract.through < date.max
        assert (contract.mode == "bridge") == (role.cohort == "historical2022_2025")


def test_temporary_native_hooks_restore_even_when_the_account_fails() -> None:
    role = next(r for r in load_roles(ROOT) if r.mode == "new_native")
    originals = (native.target, native.oracle, native.feed_names)
    with pytest.raises(RuntimeError, match="interrupted"):
        with run.scoped_hooks(role, GroupMap(groups={"group": ("A",)})):
            assert native.target is not originals[0]
            assert native.oracle is not originals[1]
            raise RuntimeError("interrupted")
    assert (native.target, native.oracle, native.feed_names) == originals


def test_group_feed_order_matches_actual_pool_without_unselected_joint_assets() -> None:
    pools, _ = run.configuration(ROOT)
    assert len(run.pool_feeds(pools, "legacy36")) == 37
    assert len(run.pool_feeds(pools, "historical34")) == 35
    assert run.pool_feeds(pools, "historical34") == tuple(sorted((*pools.historical34, "SHY")))
    assert "SGOV" not in run.pool_feeds(pools, "historical34")


def test_frozen_native_cap_counts_all_started_attempts_including_failures(tmp_path: Path) -> None:
    folder = tmp_path / run.OUT
    folder.mkdir(parents=True)
    for number in range(64):
        run.append_attempt(tmp_path, str(number), "started")
        run.append_attempt(tmp_path, str(number), "failed", "expected fixture")
    with pytest.raises(ValueError, match="64"):
        run.require_budget(tmp_path)
    assert run.native_count(tmp_path) == 64


def test_existing_native_artifact_is_revalidated_without_spending_another_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role = next(r for r in load_roles(ROOT) if r.mode == "new_native")
    folder = tmp_path / run.OUT / role.key
    folder.mkdir(parents=True)
    (folder / "ARTIFACT.json").write_text("saved account")
    monkeypatch.setattr(run, "validate_saved", lambda *args: None)
    monkeypatch.setattr(native, "_native", lambda *args: pytest.fail("native rerun forbidden"))
    run.execute(tmp_path, role, None, GroupMap(groups={}), bridge=False)
    assert run.native_count(tmp_path) == 0
    assert (folder / "ARTIFACT.json").read_text() == "saved account"
    states = [json.loads(line)["state"] for line in run.journal(tmp_path).read_text().splitlines()]
    assert states == ["revalidated"]


def test_partial_failed_native_folder_requires_explicit_new_attempt_not_silent_rerun(
    tmp_path: Path,
) -> None:
    role = next(r for r in load_roles(ROOT) if r.mode == "new_native")
    folder = tmp_path / run.OUT / role.key
    folder.mkdir(parents=True)
    (folder / "PARTIAL_NATIVE.json").write_text("evidence")
    with pytest.raises(ValueError, match="preserved"):
        run.execute(tmp_path, role, None, GroupMap(groups={}), bridge=False)
    assert (folder / "PARTIAL_NATIVE.json").read_text() == "evidence"
    assert run.native_count(tmp_path) == 0


def test_reuse_binding_checks_actual_rule_instead_of_legacy_top_three_shell() -> None:
    role = next(r for r in load_roles(ROOT) if r.mode == "reuse" and r.top_k == 1)
    from scripts.etf_group_cap_audit import validate_reference_identity

    with pytest.raises(ValueError, match="identity"):
        validate_reference_identity(role, {"key": "C11_5bp", "spec": {"label": "C11"}})


def test_joint_acceptance_requires_the_immutable_successful_pass_and_canonical_binding(
    tmp_path: Path,
) -> None:
    output = tmp_path / "inputs_etf_group_cap_20261109"
    output.mkdir()
    source = tmp_path / "frozen-source.json"
    source.write_text("source")
    hashes = {str(source): run.sha(source)}
    (output / "SOURCE_PROVENANCE.json").write_text(json.dumps({"hashes": hashes}))
    for cutoff in ("2025", "2026"):
        (output / cutoff).mkdir()
        (output / cutoff / "MANIFEST.json").write_text(json.dumps({"output_hashes": {}}))
    qa = tmp_path / run.RESEARCH / "data_qa/success"
    qa.mkdir(parents=True)
    passed = qa / "PASS.json"
    passed.write_text(json.dumps({"status": "PASS", "old38_exact": True, "source_hashes": hashes}))
    binding = {
        "status": "PASS",
        "source_pass_path": str(passed),
        "source_pass_sha256": run.sha(passed),
        "canonical_hashes": {
            str(p): run.sha(p)
            for p in (
                output / "SOURCE_PROVENANCE.json",
                output / "2025/MANIFEST.json",
                output / "2026/MANIFEST.json",
            )
        },
        "source_hashes": hashes,
    }
    (qa / "PASS_CANONICAL_BINDING.json").write_text(json.dumps(binding))
    assert str(passed) in run.verify_joint_acceptance(tmp_path)
    passed.write_text(json.dumps({"status": "PASS", "old38_exact": False, "source_hashes": hashes}))
    with pytest.raises(ValueError, match="binding|source"):
        run.verify_joint_acceptance(tmp_path)


def test_native_failure_is_metered_before_invocation_and_retained_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role = next(r for r in load_roles(ROOT) if r.mode == "new_native")
    prepared = native.Prepared(
        datasets=[],
        features=FeatureStore(frames={}),
        risk_names=(),
        groups=(),
        first=date(2020, 12, 31),
        last=role.through,
    )
    from scripts.etf_group_cap_audit import settings_for

    monkeypatch.setattr(run, "settings_for", lambda *args: settings_for(ROOT, role))

    def fail(*args: object) -> None:
        assert run.native_count(tmp_path) == 1
        assert native.target is not original
        raise RuntimeError("expected native fixture failure")

    original = native.target
    monkeypatch.setattr(native, "_native", fail)
    with pytest.raises(RuntimeError, match="expected native fixture failure"):
        run.execute(tmp_path, role, prepared, GroupMap(groups={}), bridge=False)
    assert native.target is original
    assert run.native_count(tmp_path) == 1
    assert (tmp_path / run.OUT / role.key / "FAILURE.txt").exists()
    with pytest.raises(ValueError, match="preserved"):
        run.execute(tmp_path, role, prepared, GroupMap(groups={}), bridge=False)
    assert run.native_count(tmp_path) == 1


def test_incomplete_science_never_claims_full_seventy_two_role_acceptance(tmp_path: Path) -> None:
    from scripts.etf_group_cap_targets import Pools

    monkey_roles = load_roles(ROOT)
    # A missing native VERIFIED is decisive before any source/data/finance read.
    import unittest.mock

    with unittest.mock.patch.object(run, "load_roles", return_value=monkey_roles):
        assert (
            run.finalize(tmp_path, Pools(legacy36=(), historical34=()), GroupMap(groups={}))
            is False
        )
    assert not (tmp_path / run.OUT / "FULL_AUDIT.json").exists()


def test_approved_old_empty_trade_amendment_cannot_change_any_frozen_input(tmp_path: Path) -> None:
    from scripts.etf_group_cap_audit import verify_reference_lock

    folder = tmp_path / "c10_2026_outputs/20261109_v1"
    folder.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    runner = tmp_path / "scripts/etf_c10_2026_run.py"
    runner.write_text("old source")
    frozen = tmp_path / "frozen.csv"
    frozen.write_text("price data")
    prior = {str(runner): run.sha(runner), str(frozen): run.sha(frozen)}
    (folder / "SOURCE_BEFORE_EMPTY_TRADE_FIX.py").write_bytes(runner.read_bytes())
    lock = folder / "RUN_LOCK.json"
    lock.write_text(json.dumps({"hashes": prior}))
    amendment = folder / "EMPTY_TRADES_AMENDMENT.json"
    amendment.write_text(
        json.dumps(
            {
                "reason": "empty_closed_trade_dtype_only",
                "prior_lock_sha256": run.sha(lock),
                "failed_key": "continuous2022_QQQ_BH_5bp",
            }
        )
    )
    resume = tmp_path / "scripts/etf_c10_2026_resume.py"
    resume.write_text("approved recovery source")
    runner.write_text("approved empty trade dtype fix")
    current = {
        **prior,
        str(runner): run.sha(runner),
        str(resume): run.sha(resume),
        str(amendment): run.sha(amendment),
    }
    (folder / "RUN_LOCK_V2.json").write_text(json.dumps({"hashes": current}))
    assert str(lock) in verify_reference_lock(tmp_path)
    frozen.write_text("changed price data")
    current[str(frozen)] = run.sha(frozen)
    (folder / "RUN_LOCK_V2.json").write_text(json.dumps({"hashes": current}))
    with pytest.raises(ValueError, match="unrelated|input"):
        verify_reference_lock(tmp_path)


def test_native_summary_preserves_a_failed_extra_attempt_within_the_hard_budget() -> None:
    planned = {f"path{n}" for n in range(44)}
    attempts = [{"key": key, "state": "started"} for key in sorted(planned)]
    attempts += [{"key": "path0", "state": "failed"}, {"key": "path0", "state": "started"}]
    summary = run.attempt_summary(attempts, planned)
    assert summary["native_started"] == 45
    assert summary["native_failed"] == 1
    with pytest.raises(ValueError, match="budget|paths"):
        run.attempt_summary(attempts + [{"key": "rogue", "state": "started"}], planned)
