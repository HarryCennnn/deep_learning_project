"""Shared question and evidence prompt for fixed and retrieved settings."""

SYSTEM_PROMPT = (
    "Answer the product question using only the supplied product evidence. "
    "State supported details even when another part of the question is unknown. "
    "For any missing detail, say that the supplied information does not specify it. "
    "Do not guess, use outside product knowledge, or claim that absence of a fact "
    "proves its opposite. Keep the answer concise."
)


def make_messages(question, evidence):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty")
    passages = [
        f"[{index}] ({item.get('source', 'unknown')}) {item['text']}"
        for index, item in enumerate(evidence, 1)
    ]
    evidence_text = "\n".join(passages) if passages else "[No product evidence available]"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Product evidence:\n{evidence_text}\n\nQuestion: {question.strip()}",
        },
    ]
