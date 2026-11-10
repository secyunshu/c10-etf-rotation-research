"""Reproduce54 fixed experiments through the unchanged standard native runner."""

import argparse
import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import TypeAdapter

import c10standard.audit as financial
import c10standard.native as native
import c10standard.run as standard
from c10share.verify import compare_frames, compare_json
from c10standard.io import ROOT
from c10standard.models import BOOKS, Artifact, Contract, NativeScalar, Prepared, TargetRecord

from . import io, signals
from .models import Binding, Experiment, ExperimentReceipt


def native_folder(output: Path, experiment: Experiment) -> Path:
    return output / experiment.variant / experiment.contract.key


@contextmanager
def release_budget() -> Iterator[None]:
    """Use the same new72-start journal service for default3 and experiment54."""
    previous = standard.journal_event
    standard.journal_event = io.journal_event
    try:
        yield
    finally:
        standard.journal_event = previous


def compare_experiment(
    experiment: Experiment, artifact: Artifact, folder: Path, root: Path = ROOT
) -> dict[str, float]:
    if artifact.contract != experiment.contract:
        raise ValueError("experiment artifact has a different execution contract")
    reference = io.reference_folder(experiment, root)
    differences = {
        name: compare_frames(
            pd.read_csv(reference / f"{name}.csv.gz", float_precision="round_trip"),
            pd.read_csv(folder / f"{name}.csv.gz", float_precision="round_trip"),
            name,
        )
        for name in BOOKS
    }
    targets = TypeAdapter(list[TargetRecord]).validate_json(
        (reference / "TARGETS.json").read_text()
    )
    differences["targets"] = compare_json(
        [r.model_dump(mode="json") for r in targets],
        [r.model_dump(mode="json") for r in artifact.targets],
        "experiment targets",
    )
    scalar = NativeScalar.model_validate_json((reference / "NATIVE_TOTAL_RETURN.json").read_text())
    financial.equal(
        artifact.account.native_total_return,
        scalar.native_total_return,
        "experiment actual native scalar",
        1e-10,
    )
    differences["native_total_return"] = abs(
        artifact.account.native_total_return - scalar.native_total_return
    )
    saved_name, actual_name = (
        ("METRICS.json", "metrics.json")
        if experiment.contract.terminal
        else ("YTD.json", "ytd.json")
    )
    differences[actual_name] = compare_json(
        json.loads((reference / saved_name).read_text()),
        json.loads((folder / actual_name).read_text()),
        "experiment metrics",
    )
    return differences


@contextmanager
def scoped_hooks(experiment: Experiment, root: Path = ROOT) -> Iterator[None]:
    native_namespace, audit_namespace = native.__dict__, financial.__dict__
    previous = (
        native_namespace["target"],
        audit_namespace["oracle"],
        standard.compare_reference,
        standard.journal_event,
    )

    def target(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
        if contract != experiment.contract:
            raise ValueError("target shell differs from outer experiment")
        return signals.target(experiment, prepared, day)

    def oracle(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
        if contract != experiment.contract:
            raise ValueError("oracle shell differs from outer experiment")
        return signals.oracle(experiment, prepared, day)

    def compare(
        contract: Contract, artifact: Artifact, folder: Path, root: Path = ROOT
    ) -> dict[str, float]:
        if contract != experiment.contract:
            raise ValueError("comparison shell differs from outer experiment")
        return compare_experiment(experiment, artifact, folder, root)

    def journal(
        journal: Path,
        event: Literal["started", "native_completed", "finished", "failed", "revalidated"],
        account: str,
        attempt_id: str,
        output: Path,
        error: str = "",
    ) -> None:
        io.journal_event(journal, event, experiment.key, attempt_id, output, error)

    native_namespace["target"] = target
    audit_namespace["oracle"] = oracle
    standard.compare_reference, standard.journal_event = compare, journal
    try:
        yield
    finally:
        native_namespace["target"], audit_namespace["oracle"] = previous[:2]
        standard.compare_reference, standard.journal_event = previous[2:]


def bind(experiment: Experiment, output: Path, root: Path = ROOT) -> tuple[Binding, Path]:
    expected = Binding(
        experiment=experiment,
        contract=experiment.contract,
        native_path=native_folder(output, experiment).relative_to(output).as_posix(),
        source_hashes=io.source_hashes(experiment, root),
    )
    path = output / "EXPERIMENTS" / f"{experiment.key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = Binding.model_validate_json(path.read_text())
        io.verify_source_hashes(root, saved.source_hashes)
        if saved != expected:
            raise ValueError("immutable outer experiment/source binding differs")
        return saved, path
    native.write(path, expected)
    return expected, path


def verify_outer(experiment: Experiment, output: Path, root: Path = ROOT) -> ExperimentReceipt:
    binding_path = output / "EXPERIMENTS" / f"{experiment.key}.json"
    binding = Binding.model_validate_json(binding_path.read_text())
    if (
        binding.experiment != experiment
        or binding.contract != experiment.contract
        or binding.native_path != native_folder(output, experiment).relative_to(output).as_posix()
    ):
        raise ValueError("outer experiment identity/native namespace mismatch")
    io.verify_source_hashes(root, binding.source_hashes)
    folder = output / binding.native_path
    path = output / "EXPERIMENT_VERIFIED" / f"{experiment.key}.json"
    hashes = {
        p.relative_to(output).as_posix(): io.sha(p)
        for p in folder.iterdir()
        if p.is_file() and p.name != "FAILURE.txt"
    }
    hashes[binding_path.relative_to(output).as_posix()] = io.sha(binding_path)
    expected = ExperimentReceipt(
        experiment=experiment,
        native_path=binding.native_path,
        binding_sha256=io.sha(binding_path),
        artifact_sha256=io.sha(folder / "ARTIFACT.json"),
        standard_receipt_sha256=io.sha(folder / "VERIFIED.json"),
        hashes=hashes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = ExperimentReceipt.model_validate_json(path.read_text())
        io.verify_source_hashes(output, saved.hashes)
        if saved != expected:
            raise ValueError("immutable experiment verification changed")
        return saved
    native.write(path, expected)
    return expected


def execute(
    experiment: Experiment, prepared: Prepared, output: Path, journal: Path, root: Path = ROOT
) -> ExperimentReceipt:
    registered = next((a for a in io.load_experiments(root) if a.key == experiment.key), None)
    if registered != experiment:
        raise ValueError("unregistered experiment identity")
    bind(experiment, output, root)
    with scoped_hooks(experiment, root):
        standard.execute(experiment.contract, prepared, output / experiment.variant, journal, root)
    return verify_outer(experiment, output, root)


def verify_saved(
    experiment: Experiment, prepared: Prepared, output: Path, root: Path = ROOT
) -> ExperimentReceipt:
    with scoped_hooks(experiment, root):
        standard.validate_saved(
            experiment.contract, prepared, native_folder(output, experiment), root
        )
    return verify_outer(experiment, output, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--journal", type=Path, help="shared new-release72-start journal including default3"
    )
    parser.add_argument(
        "--all54", action="store_true", help="explicitly run all54 frozen native experiments"
    )
    parser.add_argument("--case", help="run one exact published experiment key")
    parser.add_argument(
        "--verify-only", action="store_true", help="revalidate existing outputs without native"
    )
    args = parser.parse_args()
    if args.all54 and args.case:
        parser.error("choose --all54 or one --case")
    experiments = io.load_experiments(args.root)
    if args.case:
        experiments = [e for e in experiments if e.key == args.case]
        if not experiments:
            parser.error("--case must be one exact frozen experiment key")
    if not args.all54 and not args.case and not args.verify_only:
        print(f"PASS {len(experiments)} frozen experiment references; no native execution")
        return
    if args.output is None:
        parser.error("--output is required for running or revalidating outputs")
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.journal or args.output / "release_attempts.jsonl"
    with (args.output / ".native.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache: dict[str, Prepared] = {}
        for experiment in experiments:
            cutoff = "2025" if experiment.contract.terminal else "2026"
            if cutoff not in cache:
                cache[cutoff] = io.load_market(args.root, cutoff)
            receipt = (
                verify_saved(experiment, cache[cutoff], args.output, args.root)
                if args.verify_only
                else execute(experiment, cache[cutoff], args.output, journal, args.root)
            )
            print(f"{receipt.status} {experiment.key}", flush=True)


if __name__ == "__main__":
    main()
