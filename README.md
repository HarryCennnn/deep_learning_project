# Reliable Product Question Answering under Limited Supervision and Incomplete Evidence

**Group 57 · Carnegie Mellon University**

Siyu Fang · Yitian Pan · Harry Cen

This project studies how to answer product questions when the available product
evidence supports only part of an answer. For example, "requires two AA
batteries" supports the battery type and quantity but does not establish
whether batteries are included. We aim to reduce unsupported product claims
while retaining useful, supported details.

## Research design and status

We separate **generation with a fixed candidate passage** from **generation
after retrieval**. This lets us examine evidence use before attributing an
error to a retriever. The starter code currently contains:

| Component | Status |
| --- | --- |
| ePQA tab-delimited data adapter and balanced pilot sampling | Implemented |
| Prompted instruction-model baseline with fixed evidence | Implemented; model run pending |
| Same-product lexical retrieval + shared generator | Implemented; model run pending |
| Manual claim/unit annotation template and metric scorer | Implemented; real annotation pending |
| Conventional and evidence-aware LoRA adaptations | Planned |
| Empirical results and conclusions | Pending |

No benchmark scores are presented yet. The included toy data are synthetic
and only demonstrate the file format and software path. The initial retrieval
baseline uses simple lexical matching and is not a reimplementation of RAFT,
AGREE, or CRAG.

Our candidate dataset is [ePQA](https://github.com/amazon-science/contextual-product-qa/tree/main/ePQA).
Its label 2 marks fully answering candidates, label 1 helpful but not fully
answering candidates, and label 0 irrelevant candidates. The answer field is
available for fully answering candidates. We use the candidate text as the
fixed evidence and do not put the reference answer or relevance label into
the model prompt. Label 1 is **not** a gold partial-answer target.

## Repository layout

| Path | Purpose |
| --- | --- |
| scripts/prepare_epqa.py | Convert an official ePQA split into JSONL; optional balanced sampling |
| scripts/run_baseline.py | Fixed candidate or same-product retrieved evidence, with the same prompt and model |
| scripts/evaluate_retrieval.py | Label-2 passage hit@k on questions with a fully answering candidate |
| scripts/make_annotations.py | Create blank manual evaluation sheets |
| scripts/score_annotations.py | Aggregate claim-level faithfulness and coverage |
| src/product_qa/ | Reusable data, lexical retrieval, prompting, and scoring code |
| examples/toy_epqa.tsv | Synthetic data for a local smoke test |
| docs/annotation_guidelines.md | Claim and answerable-unit annotation instructions |
| docs/midterm_progress.md | Honest midterm status and next experimental steps |

## Quick smoke test (no model download)

Use Python 3.10+ from the repository root:

~~~bash
python -m unittest discover -s tests -v
python scripts/prepare_epqa.py --input examples/toy_epqa.tsv --split dev --output data/processed/toy.jsonl
python scripts/run_baseline.py --input data/processed/toy.jsonl --mode fixed --prompts-only --output runs/toy_prompts.jsonl
~~~

The prompts-only output has null answers. It checks conversion and prompt
assembly; it does not provide evaluation results.

## Run a prompted pilot on ePQA

Download the official ePQA splits from the dataset link above and place them
in data/raw/ePQA/. Despite the .csv extension, these files are
tab-delimited. Do not commit the dataset, generated outputs, or model weights.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_epqa.py --input data/raw/ePQA/dev.csv --split dev --output data/processed/dev_full.jsonl
python scripts/prepare_epqa.py --input data/raw/ePQA/dev.csv --split dev --max-per-label 30 --seed 57 --output data/processed/dev_pilot.jsonl
python scripts/evaluate_retrieval.py --input data/processed/dev_full.jsonl --top-k 3

python scripts/run_baseline.py --input data/processed/dev_pilot.jsonl --mode fixed --model Qwen/Qwen2.5-0.5B-Instruct --output runs/fixed_dev.jsonl
python scripts/run_baseline.py --input data/processed/dev_pilot.jsonl --mode retrieved --retrieval-pool data/processed/dev_full.jsonl --top-k 3 --model Qwen/Qwen2.5-0.5B-Instruct --output runs/retrieved_dev.jsonl
~~~

The model ID is an **illustrative small baseline**, not a finalized selection.
It requires downloading weights and enough local CPU/GPU memory. Both runs use
the same prompt, model, and greedy decoding. The fixed setting supplies one
row's candidate; the retrieved setting searches candidates for the same ASIN
from the supplied split. The retriever sees candidate text and source only,
never reference answers or relevance labels. It returns no evidence when
there is no lexical overlap. The optional --limit flag can make a shorter
inference smoke test.

The retrieval diagnostic reports label-2 hit@k only for questions that have
at least one label-2 candidate in the supplied split. It is a candidate
ranking measure, not answer faithfulness or a test of unanswerable questions.

For evaluation, make a sheet from each prediction file:

~~~bash
python scripts/make_annotations.py --predictions runs/fixed_dev.jsonl --output annotations/fixed_dev.jsonl
# Fill claims, answerable_units, and complete by hand as described in docs/annotation_guidelines.md.
python scripts/score_annotations.py --input annotations/fixed_dev.jsonl --output runs/fixed_dev_metrics.json
~~~

**Unsupported Claim Rate** is the number of unsupported factual product
claims divided by all factual product claims. **Answerable Coverage** is the
number of correctly covered evidence-supported units divided by all such
units. Both use micro totals. A missing denominator is reported as null;
responses with no factual claims are counted separately. An absence of
evidence does not imply the opposite product claim.

## Planned comparisons

1. Prompted base model and conventional LoRA on fully answering examples.
2. Evidence-aware LoRA with explicitly authored or validated targets for
   partial and insufficient evidence. We will not reuse a full answer as a
   target for a label-1 candidate.
3. The same generator with supplied versus retrieved evidence. Report
   retrieval failures separately and inspect product overlap across splits.
4. Compare both metrics and qualitative cases, especially useful partial
   answers, unsupported claims, and excessive refusal.

We will record the sample, seed, model revision, decoding settings, and
hardware when actual experiments are run. See [midterm progress](docs/midterm_progress.md)
for a report-ready status outline with unfilled results.

## Related work

- [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131)
  (Zhang et al., 2024) motivates training in the presence of relevant and
  distractor documents. We may adapt its evidence-exposure idea later.
- [Effective Large Language Model Adaptation for Improved Grounding and Citation Generation](https://aclanthology.org/2024.naacl-long.346/)
  (Ye et al., 2024; AGREE) informs grounding and attribution analysis. We do
  not claim citation generation in this first baseline.
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
  (Yan et al., 2024; CRAG) highlights sensitivity to retrieval quality.
  Retrieval evaluation and correction remain future extensions.
- [Contextual Product Question Answering datasets](https://github.com/amazon-science/contextual-product-qa)
  (Amazon Science) provides ePQA and its candidate-level label definitions.
