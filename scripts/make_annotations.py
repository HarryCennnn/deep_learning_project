#!/usr/bin/env python3
"""Prepare a human annotation sheet from generated responses."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from product_qa.data import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def rows():
        for row in read_jsonl(args.predictions):
            if not isinstance(row.get("answer"), str):
                raise ValueError(f"{row.get('id')}: no generated answer to annotate")
            yield {
                "id": row["id"],
                "mode": row["mode"],
                "question": row["question"],
                "evidence": row["evidence"],
                "answer": row["answer"],
                "evidence_label": row["evidence_label"],
                "claims": [],
                "answerable_units": [],
                "complete": False,
                "notes": "",
            }

    write_jsonl(args.output, rows())
    print(f"Annotation sheet written to {args.output}; complete each row by hand.")


if __name__ == "__main__":
    main()
