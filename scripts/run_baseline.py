#!/usr/bin/env python3
"""Generate fixed-evidence or same-product retrieved answers with one LM."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from product_qa.data import read_jsonl, write_jsonl
from product_qa.prompting import make_messages
from product_qa.retrieval import ProductRetriever


def load_generator(model_name, max_input_tokens, max_new_tokens):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements.txt to run model inference") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.chat_template:
        raise ValueError("Choose an instruction model with a tokenizer chat template")
    tokenizer.truncation_side = "left"  # Preserve the question at the end of the prompt.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    def generate(messages):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True).strip()

    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared JSONL query set")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("fixed", "retrieved"), default="fixed")
    parser.add_argument("--retrieval-pool", help="Full prepared split; required in retrieved mode")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", help="Hugging Face instruction model ID")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, help="Optional small pilot run")
    parser.add_argument(
        "--prompts-only", action="store_true", help="Inspect prompt assembly without model inference"
    )
    args = parser.parse_args()
    if args.mode == "retrieved" and not args.retrieval_pool:
        parser.error("--retrieval-pool is required in retrieved mode")
    if not args.prompts_only and not args.model:
        parser.error("--model is required unless --prompts-only is set")
    if args.top_k < 1 or args.max_input_tokens < 1 or args.max_new_tokens < 1:
        parser.error("Token limits and --top-k must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    retriever = (
        ProductRetriever(read_jsonl(args.retrieval_pool))
        if args.mode == "retrieved"
        else None
    )
    generate = None if args.prompts_only else load_generator(
        args.model, args.max_input_tokens, args.max_new_tokens
    )
    rows = read_jsonl(args.input)
    if args.limit is not None:
        from itertools import islice

        rows = islice(rows, args.limit)

    def predictions():
        for row in rows:
            evidence = (
                row["evidence"]
                if retriever is None
                else retriever.search(row["product_id"], row["question"], args.top_k)
            )
            messages = make_messages(row["question"], evidence)
            result = {
                "id": row["id"],
                "split": row["split"],
                "product_id": row["product_id"],
                "question": row["question"],
                "evidence_label": row["evidence_label"],
                "mode": args.mode,
                "model": args.model,
                "evidence": evidence,
                "answer": generate(messages) if generate else None,
            }
            if args.prompts_only:
                result["messages"] = messages
            yield result

    write_jsonl(args.output, predictions())
    print(f"Wrote {'prompts' if args.prompts_only else 'predictions'} to {args.output}")


if __name__ == "__main__":
    main()
