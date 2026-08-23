"""
pipeline/context_builder.py

WHAT THIS FILE DOES
--------------------
Takes the Records returned by the retrievers (a list of structured
objects) and turns them into ONE plain-text block that gets inserted
into the prompt sent to the LLM.

WHY IT EXISTS AS ITS OWN STEP
--------------------------------
Retrieval (finding the right records) and context formatting
(turning records into LLM-readable text) are two different concerns.
Keeping them separate means:
  - You can change HOW retrieved data is displayed to the LLM
    (e.g. add source labels, reorder, truncate long records) without
    touching any retriever.
  - Phase 2's re-ranker/CRAG steps can filter or reorder the list of
    Records BEFORE it reaches this function, and this function
    doesn't need to know or care that happened.

INPUTS  : list[Record] -- already-retrieved, already-ranked records
OUTPUTS : a single string, formatted with clear source labels,
          ready to drop into the user prompt
"""

from loaders.record import Record
from config.settings import MAX_RECORD_CHARS

# Human-readable labels for each source, used as section headers in
# the context block so the LLM (and you, when debugging) can see
# where each piece of information came from.
SOURCE_LABELS = {
    "cards": "CREDIT CARD PRODUCTS",
    "offers_installment": "MERCHANT INSTALLMENT OFFERS",
    "offers_discount": "MERCHANT DISCOUNT OFFERS",
    "campaigns": "CAMPAIGNS",
}


def _truncate(text: str, limit: int) -> str:
    """
    Cap one record's rendered text.

    WHY THIS EXISTS
    ---------------
    Card records are enormous: a single one is ~42,000 characters
    (~10,600 tokens) because the product catalogue has 366 columns and
    every one of them is rendered. Two card records alone therefore
    overflow any sane generator context window, and an overflowing
    prompt is silently truncated by Ollama -- which is how the model
    came to report an annual fee of "EGP 4,500" for a card whose record
    says no such thing.

    Cutting here is better than relying on a bigger window for two
    reasons: it bounds the prompt no matter how many records arrive,
    and it keeps the answer-bearing head of each record (name, type,
    fees, limits) rather than letting the tail push it out of view.

    The cut is marked, so a truncated record is visible when reading
    the prompt back during debugging, and the generator can tell that
    the record continues rather than assuming it has seen everything.
    """
    if limit <= 0 or len(text) <= limit:
        return text

    return (
        text[:limit].rstrip()
        + f"\n[... record truncated at {limit} characters ...]"
    )


def build_context(
    records: list[Record],
    max_record_chars: int = MAX_RECORD_CHARS,
) -> str:
    """
    Group records by source and render each as a labeled section,
    e.g.:

        === CREDIT CARD PRODUCTS ===
        Card: VISA INFINITE
        - Minimum limit: ...
        ...

        === CAMPAIGNS ===
        Campaign: Egypt Air July 2026 Installment Campaign
        ...

    Returns an empty string if no records were retrieved.
    """
    if not records:
        return ""

    grouped: dict[str, list[Record]] = {}
    for r in records:
        grouped.setdefault(r.source, []).append(r)

    sections = []
    for source, source_records in grouped.items():
        label = SOURCE_LABELS.get(source, source.upper())
        block_lines = [f"=== {label} ==="]
        for r in source_records:
            block_lines.append(
                _truncate(r.display_text, max_record_chars)
            )
            block_lines.append("")  # blank line between records
        sections.append("\n".join(block_lines).strip())

    return "\n\n".join(sections)
