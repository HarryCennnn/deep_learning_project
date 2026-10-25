#!/usr/bin/env python3
"""Convert the official tab-delimited ePQA split to the experiment format."""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from product_qa.data import evidence_item, write_jsonl


def convert(path, split):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"ASIN", "question", "candidate", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"ePQA TSV requires columns: {', '.join(sorted(required))}")
        for index, row in enumerate(reader, 1):
            try:
                label = int(row["label"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Row {index}: invalid label") from exc
            if label not in (0, 1, 2):
                raise ValueError(f"Row {index}: label must be 0, 1, or 2")
            if not row["ASIN"] or not row["question"] or not row["candidate"]:
                raise ValueError(f"Row {index}: missing ASIN, question, or candidate")
            answer = (row.get("answer") or "").strip() if label == 2 else ""
            yield {
                "id": f"{split}:{index}",
                "split": split,
                "product_id": row["ASIN"].strip(),
                "question_id": (row.get("qid") or "").strip(),
                "question": row["question"].strip(),
                "evidence": [evidence_item(row["candidate"], row.get("source"))],
                "evidence_label": label,
                "reference_answer": answer or None,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Official ePQA .csv (tab-delimited)")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=57)
    args = parser.parse_args()

    if args.max_per_label is not None and args.max_per_label < 1:
        parser.error("--max-per-label must be positive")
    rows = list(convert(args.input, args.split))
    full_counts = Counter(row["evidence_label"] for row in rows)
    if args.max_per_label is not None:
        rng = random.Random(args.seed)
        rows = [
            row
            for label in (0, 1, 2)
            for row in rng.sample(
                [item for item in rows if item["evidence_label"] == label],
                min(args.max_per_label, full_counts[label]),
            )
        ]
        rows.sort(key=lambda item: int(item["id"].split(":")[-1]))
    write_jsonl(args.output, rows)
    counts = Counter(row["evidence_label"] for row in rows)
    print(f"Wrote {len(rows)} {args.split} candidates to {args.output}")
    print(f"Source counts by label: {dict(sorted(full_counts.items()))}")
    print(f"Selected counts by label: {dict(sorted(counts.items()))}")
    print(f"Unique products: {len({row['product_id'] for row in rows})}")
    print("Labels indicate candidate relevance, not claim-level support annotations.")


if __name__ == "__main__":
    main()
