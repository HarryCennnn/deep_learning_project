# Manual evaluation protocol (draft)

Annotate a small, fixed sample of generated answers for each model and evidence
setting. Keep the question and *the exact evidence supplied to the generator*
visible. Do not use the reference answer or external product knowledge to decide
whether a claim is supported.

1. Split the generated answer into atomic factual product claims. Exclude statements
   such as "the provided information does not specify whether batteries are
   included" when they only describe the evidence gap. Mark each claim
   supported true only if the supplied evidence entails it; otherwise false.
   A plausible claim without evidence counts as unsupported, even if it might
   happen to be true.
2. Separately write the atomic facts about the question that *can be answered
   from the supplied evidence*. Mark covered true only if the answer correctly
   communicates the fact. Do not create units for missing information.
3. Set complete to true after both lists have been reviewed. Empty lists are
   valid for genuinely empty denominators. Explain borderline decisions in notes.
   Resolve disagreements on a subset with a second annotator before scoring.

Example: question "How many batteries are needed, and are they included?",
evidence "Requires two AA batteries.", answer "It requires two AA batteries;
the supplied information does not say whether they are included."

The answer has one product claim ("requires two AA batteries", supported true)
and one answerable unit ("requires two AA batteries", covered true). It should
not be marked as an answer to the inclusion part. If the answer says "batteries
are included", add a second claim with supported false.

Metrics are **micro-averaged** over completed rows:

- Unsupported Claim Rate = unsupported product claims / all factual product claims.
- Answerable Coverage = correctly covered answerable units / all answerable units.
- A zero denominator yields null, not 0%. Report total counts and the number of
  responses with no factual product claims to expose excessive abstention.

Use the same annotated examples and evidence when comparing models. For retrieved
mode, the ePQA row label still describes the original candidate; it does not
label the quality of the retrieved set. Inspect the retrieved passages manually.
The automatic scorer trusts annotation decisions; it does not verify entailment.
