"""Rebuild all real monthly signals with physical prefixes; never run a broker."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from c10experiments.io import prefix_market
from c10share.verify import compare_json
from c10standard.io import ROOT, load_market
from c10standard.models import Contract
from c10standard.native import decision_days

from .models import VARIANTS
from .signals import oracle, oracle_scores, score_table, target


def verify(root: Path, output: Path) -> None:
    full = load_market(root, cutoff="2026")
    days = (date(2021, 12, 31), *decision_days(Contract(cohort="continuous2022_2026")))
    observations = []
    largest = 0.0
    for day in days:
        prefix = prefix_market(full, day)
        for variant in VARIANTS:
            scored = score_table(variant, full, day)
            expected = scored.model_dump(mode="json")
            for actual in (score_table(variant, prefix, day), oracle_scores(variant, prefix, day)):
                largest = max(
                    largest,
                    compare_json(
                        expected, actual.model_dump(mode="json"), f"{variant}/{day}/scores"
                    ),
                )
            planned = target(variant, full, day)
            for actual_target in (target(variant, prefix, day), oracle(variant, prefix, day)):
                largest = max(
                    largest,
                    compare_json(
                        planned.model_dump(mode="json"),
                        actual_target.model_dump(mode="json"),
                        "targets",
                    ),
                )
            observations.append(
                {
                    "variant": variant,
                    "day": str(day),
                    "score_sha256": hashlib.sha256(scored.model_dump_json().encode()).hexdigest(),
                    "target": planned.model_dump(mode="json"),
                }
            )
    receipt = {
        "status": "PASS",
        "monthly_dates": len(days),
        "variant_dates": len(observations),
        "largest_numeric_difference": largest,
        "native_attempts": 0,
        "method": "full as-of vs physically cut raw prefix/rebuilt features vs raw oracle",
        "observations": observations,
    }
    if output.exists():
        if json.loads(output.read_text()) != receipt:
            raise ValueError("immutable causal receipt changed")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        f"PASS: {len(days)} monthly dates, {len(observations)} variant dates; "
        f"largest difference={largest:.3g}; native=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.output)


if __name__ == "__main__":
    main()
