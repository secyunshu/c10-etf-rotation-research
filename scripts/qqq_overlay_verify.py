"""Real historical signal prefix verification without native execution."""

import argparse
import json
from datetime import date
from pathlib import Path

from c10experiments.io import prefix_market
from c10share.verify import compare_json
from c10standard.io import ROOT, load_market
from c10standard.models import Contract
from c10standard.native import decision_days
from scripts.qqq_overlay_signals import oracle, target


def verify(root: Path, output: Path) -> None:
    market = load_market(root, "2025")
    days = (date(2021, 12, 31), *decision_days(Contract()))
    records = []
    maximum = 0.0
    for day in days:
        prefix = prefix_market(market, day)
        actual = target(market, day)
        for other in (target(prefix, day), oracle(prefix, day)):
            maximum = max(
                maximum,
                compare_json(
                    actual.model_dump(mode="json"), other.model_dump(mode="json"), "QQQ targets"
                ),
            )
        records.append(actual.model_dump(mode="json"))
    result = {
        "status": "PASS",
        "monthends": len(days),
        "max_difference": maximum,
        "native_attempts": 0,
        "physical_cutoff": "2025-12-31",
        "targets": records,
    }
    if len(days) != 48:
        raise ValueError("fixed48 historical signal dates required")
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("immutable prefix receipt differs")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS:48 historical monthends; max difference={maximum}; native=0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.output)


if __name__ == "__main__":
    main()
