#!/usr/bin/env python3
"""Aggregate completed manual annotations. No automatic faithfulness judge."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from product_qa.data import read_jsonl
from product_qa.evaluation import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="Optional metrics JSON path")
    args = parser.parse_args()
    metrics = score(read_jsonl(args.input))
    payload = json.dumps(metrics, indent=2)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
