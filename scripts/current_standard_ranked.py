"""Current V2.3.1 ranked60/30 standard; frozen V2.3.0 helpers remain unchanged."""
import argparse
import gzip
from datetime import date
from pathlib import Path
from typing import Annotated, Literal
from unittest.mock import patch

from beartype import beartype
from pydantic import Field

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import audit, native
from c10standard.io import ROOT, settings_for
from c10standard.models import Artifact, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Model
from rotbt.phase2.models import FeatureStore
from scripts import annual_io, ytd_annual_io
from scripts.annual_models import Pool
from scripts.current_standard import Policy as PriorPolicy
from scripts.current_standard import select_target as prior_select
from scripts.v23_rank_weights import Context, Window, allocation, context, contract_for, describe
from scripts.ytd_annual_models import Pool as YtdPool
from scripts.ytd_annual_report import Statistics

RELEASE = Path("standard/v2_3_1")
RULES = ("C11", "SPY_BH", "QQQ_BH")


class Policy(Model):
    policy_version: Literal["2.3.1"]
    prior_policy: Literal["2.3.0"]
    id: Literal["ANNUAL_TOP2_RANKED_CONDITIONAL_SLEEVE"]
    adopted_date: date
    ranked_slot_weights: tuple[Annotated[float, Field(ge=.6, le=.6)],
                               Annotated[float, Field(ge=.3, le=.3)]]
    sleeve_weight: float = Field(ge=.1, le=.1)
    shy_momentum_gate: Literal[False]
    base_policy_path: Literal["standard/v2_3/STRATEGY.json"]
    base_policy_sha256: str


class Pointer(Model):
    policy_version: Literal["2.3.1"]
    strategy_path: Literal["standard/v2_3_1/STRATEGY.json"]
    strategy_sha256: str
    entrypoint: Literal["scripts.current_standard_ranked"]
    prior_policy: Literal["2.3.0"]


class ReportRow(Model):
    window: Window
    strategy: str
    stats: Statistics


@beartype
def load_policy(root: Path) -> Policy:
    pointer = Pointer.model_validate_json((root / "standard/CURRENT.json").read_bytes())
    p = root / pointer.strategy_path
    if sha(p) != pointer.strategy_sha256:
        raise ValueError("current policy hash mismatch")
    policy = Policy.model_validate_json(p.read_bytes())
    base = root / policy.base_policy_path
    if sha(base) != policy.base_policy_sha256:
        raise ValueError("base policy hash mismatch")
    PriorPolicy.model_validate_json(base.read_bytes())
    return policy


@beartype
def select_target(pool: Pool | YtdPool, features: FeatureStore, day: date) -> TargetRecord:
    prior = prior_select(pool, features, day)
    picks = [s for s, weight in prior.q.items() if weight == .45]
    if len(picks) != 2:
        raise ValueError("prior ordered selection missing two slots")
    # Prior selector inserts assets in score order; weights never follow code order.
    sleeve = "SHY" if {"QQQ", "SPY"}.intersection(picks) else "QQQ"
    return TargetRecord(day=day, q={picks[0]: .6, picks[1]: .3, sleeve: .1}, reason="oracle")


@beartype
def verify_release(root: Path) -> None:
    load_policy(root)
    annual_io.verify_lock(root)
    ytd_annual_io.verify_lock(root)
    proof = Fingerprints.model_validate_json((root / RELEASE / "RELEASE_LOCK.json").read_bytes())
    for name, expected in proof.hashes.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or sha(p) != expected:
            raise ValueError(f"release file mismatch: {name}")


@beartype
def verify_account(ctx: Context, window: Window, rule: str, output: Path,
                   execute: bool) -> Artifact:
    reference = ROOT / RELEASE / "reference" / window / rule
    proof = Fingerprints.model_validate_json((reference / "VERIFIED.json").read_bytes())
    verify_hashes(reference, proof.hashes)
    expected = Artifact.model_validate_json(gzip.decompress(
        (reference / "ACCOUNT.json.gz").read_bytes()))
    contract = contract_for(window, 5).model_copy(update={"rule": rule})
    settings = settings_for(ctx.market.strategy, contract)
    settings.invested = 1.
    schedule = {p.day: allocation(p, "60_30") for p in ctx.picks}

    def target(actual: Contract, market: Prepared, day: date) -> TargetRecord:
        if actual != contract or market is not ctx.market:
            raise ValueError("callback identity mismatch")
        return (schedule[day] if rule == "C11" else TargetRecord(
            day=day, q={rule.removesuffix("_BH"): 1.}, reason="oracle"))

    artifact = expected
    if execute:
        folder = output / window / rule
        p = folder / "ARTIFACT.json"
        if p.exists():
            artifact = Artifact.model_validate_json(p.read_bytes())
        else:
            folder.mkdir(parents=True)
            with patch.object(native, "target", target):
                artifact = native.simulate(ctx.market, contract, settings, folder)
        if Artifact.model_validate_json(artifact.model_dump_json()) != expected:
            raise ValueError("native standard differs from reference")
    artifact.contract = contract
    with patch.object(audit, "oracle", target):
        audit.audit(artifact, ctx.market, settings)
    return artifact


@beartype
def run(output: Path, execute: bool = False) -> None:
    verify_release(ROOT)
    rows = []
    for window in ("HIST", "YTD2026"):
        ctx = context(window)
        for rule in RULES:
            artifact = verify_account(ctx, window, rule, output, execute)
            for r in describe(artifact, window, "60_30", 5):
                rows.append(ReportRow(window=window, strategy=rule, stats=r.stats))
            print(window, rule, "PASS", flush=True)
    from pydantic import TypeAdapter
    archived = (ROOT / RELEASE / "RESULTS.json").read_bytes()
    if TypeAdapter(list[ReportRow]).validate_json(archived) != rows:
        raise ValueError("standard report differs from native accounts")
    print("Current V2.3.1: six60/30-standard and benchmark accounts verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/current-standard-v2-3-1"))
    args = parser.parse_args()
    run(args.output.resolve(), args.execute)


if __name__ == "__main__":
    main()
