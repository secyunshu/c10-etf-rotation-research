"""Rebuild the public tables and plots from the archived account books."""

from pathlib import Path

from c10share.report import build_report

if __name__ == "__main__":
    result = build_report(Path(__file__).resolve().parents[1])
    print(result.model_dump_json(indent=2))
