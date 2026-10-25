#!/usr/bin/env python3
"""Measure whether top-k same-product passages contain an ePQA label-2 candidate."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from product_qa.data import read_jsonl
from product_qa.retrieval import ProductRetriever


def evaluate(rows, top_k):
    rows = list(rows)
    retriever = ProductRetriever(rows)
    questions = defaultdict(list)
    for row in rows:
        key = (
            row["split"],
            row["product_id"],
            row.get("question_id") or row["question"],
        )
        questions[key].append(row)

    eligible = hits = 0
    for (_, product_id, _), candidates in questions.items():
        gold = {
            (item["text"], item.get("source", "unknown"))
            for row in candidates
            if row["evidence_label"] == 2
            for item in row["evidence"]
        }
        if not gold:
            continue
        eligible += 1
        retrieved = retriever.search(product_id, candidates[0]["question"], top_k)
        if any((item["text"], item.get("source", "unknown")) in gold for item in retrieved):
            hits += 1
    return {
        "questions": len(questions),
        "questions_with_label_2": eligible,
        "questions_without_label_2": len(questions) - eligible,
        "top_k": top_k,
        "label_2_hit_count": hits,
        "label_2_hit_at_k": hits / eligible if eligible else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Use a full official split for evaluation")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    print(json.dumps(evaluate(read_jsonl(args.input), args.top_k), indent=2))


if __name__ == "__main__":
    main()
