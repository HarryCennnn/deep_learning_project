# Group 57 — midterm progress notes (draft)

## Research question

How do adaptation and retrieval affect evidence faithfulness and the coverage of
answerable details in product question answering with limited supervision?

## Implemented in this starter repository

- An adapter for the official ePQA tab-delimited splits. It preserves the
  candidate relevance label and uses reference answers only for label 2.
- A shared prompted generator interface for supplied candidate evidence and
  same-product lexical retrieval, using a configurable instruction model.
- A claim/unit annotation sheet and a scorer for Unsupported Claim Rate and
  Answerable Coverage, plus unit tests and a synthetic workflow fixture.

**No model inference, LoRA adaptation, human evaluation, or measured comparison
is claimed by this commit.** Fill this section only after running experiments:

| Condition | Sample size | Unsupported Claim Rate | Answerable Coverage |
| --- | ---: | ---: | ---: |
| Prompted model + fixed candidate | pending | pending | pending |
| Prompted model + retrieved candidates | pending | pending | pending |
| Conventional LoRA + fixed candidate | planned | planned | planned |
| Evidence-aware LoRA + fixed candidate | planned | planned | planned |

## Next experiments

1. Inspect label counts, product overlap between official splits, and candidate
   quality. Use a fixed evaluation sample and record seed, model revision,
   inference parameters, and hardware.
2. Run the prompted model on fixed evidence. Manually decompose answerable units
   and generated claims; compare label 2, 1, and 0 settings. The label is a
   coarse candidate relevance signal, not a claim-level annotation.
3. Train a conventional LoRA baseline on fully answering label-2 pairs only,
   using evidence/question/answer triples. Keep product-disjoint validation if
   the official splits overlap at product level.
4. Author or validate targets for partial and insufficient evidence, then train
   an evidence-aware adaptation. A label-1 row does not supply a gold partial
   answer, so do not turn the full answer into a target for that row.
5. Evaluate the same generator with a fixed evidence pool and a retrieved pool;
   audit retrieval errors separately. Compare both faithfulness and coverage,
   and report annotation disagreement and qualitative failure cases.

RAFT motivates training in the presence of distractor passages; AGREE motivates
claim-level grounding and attribution; CRAG motivates inspecting retrieval
quality. The current lexical retriever is an intentionally simple baseline and
does not implement those papers' full methods.
