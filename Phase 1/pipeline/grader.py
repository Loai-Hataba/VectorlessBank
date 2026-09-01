"""
pipeline/grader.py

WHAT THIS FILE DOES
-------------------
Judges the records that retrieval returned, in one LLM call, and
produces two things from that single judgement:

    1. a relevance label per record   -> Corrective RAG's filter
    2. an ordering over the records   -> Re-ranking

    retrieved records
        |
        v
    RecordGrader.grade(question, records)
        |
        +-- label each record: relevant / partially_relevant / irrelevant
        +-- rank each record:  1 = most useful
        |
        v
    GradeResult
        .kept      -> relevant+partial, rank-ordered, capped at top_k
        .verdict   -> "correct" | "incorrect"
        .needs_retry

WHY GRADING AND RE-RANKING SHARE ONE CALL
-----------------------------------------
Both need the model to answer the same underlying question -- "how
useful is this record for answering this question?" -- and merely
consume the answer differently: a categorical filter versus an
ordering. Asking twice would double the latency of the slowest part of
the pipeline for no accuracy gain. This is deliberately one
implementation satisfying two separate requirements.

WHY THE LABELS ARE CATEGORICAL AND NOT A NUMERIC SCORE
------------------------------------------------------
A small local model produces a reliable three-way label far more
readily than a well-calibrated 0..1 score. The branch logic only needs
"retry or don't", and the finer ordering is what `rank` is for.

MODEL PROPOSES, PYTHON DISPOSES
-------------------------------
Nothing the model returns is trusted as-is. IDs are whitelisted against
the records actually retrieved, labels are clamped to the enum, ranks
are coerced to integers, and the branch decision is computed in Python
from the cleaned labels -- never read from the model's own words. This
is the same discipline the traverser uses for node IDs.

INPUTS  : question (str), list[Record]
OUTPUTS : GradeResult

REQUIRES: Ollama running locally with the grader model pulled.
"""

import json
from dataclasses import dataclass, field

from config.settings import (
    RERANK_TOP_K,
    GRADER_MAX_RECORD_CHARS,
    CRAG_MIN_RELEVANT,
)
from llm.llm_client import LLMClient


RELEVANT = "relevant"
PARTIALLY_RELEVANT = "partially_relevant"
IRRELEVANT = "irrelevant"

VALID_RELEVANCE = frozenset({RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT})

# Where an unusable label or a missing judgement lands.
#
# Deliberately NOT "irrelevant": a record the model failed to mention,
# or labelled with a word we don't recognise, has not been judged
# irrelevant -- it has not been judged at all. Dropping it would let a
# lazy or malformed reply silently delete the record that holds the
# answer. Treating it as partial keeps it available but ranks it last,
# so a confidently-relevant record always outranks it.
FALLBACK_RELEVANCE = PARTIALLY_RELEVANT

# Rank assigned when the model gives none. Sorts after any real rank.
UNRANKED = 10_000


GRADER_SYSTEM_PROMPT = """
You are a retrieval evaluator for a retail banking assistant.

You are given a customer question and a numbered list of records that
were retrieved for it. For EACH record you must decide two things:

1. relevance -- one of exactly these three strings:
   - "relevant"            the record can answer the question, or a
                           substantial part of it
   - "partially_relevant"  the record is about the right subject but
                           does not directly answer the question
   - "irrelevant"          the record is about something else

2. rank -- an integer starting at 1, where 1 is the record most useful
   for answering the question. Every record gets a distinct rank.

RULES:

1. Return ONLY valid JSON. No prose, no explanation outside the JSON.
2. Judge every record you are given, exactly once.
3. Use the record's "id" exactly as it was given to you.
4. Do not invent records or IDs.
5. Do not answer the customer's question. You are only grading.
6. A record that merely mentions a word from the question is not
   relevant. Ask whether it actually answers what was asked.

THE NAMED-ENTITY RULE (the most common grading mistake):

If the question names a specific merchant, card, or campaign, then a
record about a DIFFERENT merchant, card, or campaign is "irrelevant" --
even when it is the same KIND of product.

Being the same kind of thing is not the same as being the thing asked
about. "Same category" is not relevance.

Worked example. Question: "What installment offers are available at
Carrefour?" Retrieved records:

  - Vevian installment offer   -> "irrelevant"  (different merchant)
  - Mahgoub installment offer  -> "irrelevant"  (different merchant)
  - Carrefour installment offer-> "relevant"    (this is the merchant
                                                 that was asked about)

Marking the Vevian record "relevant" there would be wrong: the customer
asked about Carrefour, and nothing in the Vevian record tells them
anything about Carrefour.

It is correct and useful to mark EVERY record "irrelevant" when none of
them is about the thing that was asked about. Do not stretch to find
something relevant.

Required output shape:

{
  "records": [
    {"id": "card_303", "relevance": "relevant", "rank": 1},
    {"id": "offer_installment_4_Carrefour", "relevance": "irrelevant", "rank": 2}
  ]
}
"""


@dataclass
class GradedRecord:
    """One record plus the judgement made about it."""

    record: object
    relevance: str
    rank: int

    @property
    def is_useful(self) -> bool:
        return self.relevance in (RELEVANT, PARTIALLY_RELEVANT)


@dataclass
class GradeResult:
    """
    Outcome of grading one batch of retrieved records.

    kept     : the records worth generating from, best first, capped
    graded   : every candidate with its final (cleaned) judgement
    verdict  : "correct" if anything useful survived, else "incorrect"
    warnings : anything the model got wrong that Python had to repair,
               worth logging and worth reading when tuning the prompt
    """

    kept: list = field(default_factory=list)
    graded: list = field(default_factory=list)
    verdict: str = "incorrect"
    warnings: list = field(default_factory=list)

    @property
    def needs_retry(self) -> bool:
        """
        CRAG's 'Incorrect' branch: nothing useful came back, so the
        caller should widen the search once. The retry cap itself lives
        with the caller, not here -- this object only reports what it
        found.
        """
        return self.verdict == "incorrect"

    @property
    def relevant_count(self) -> int:
        return sum(1 for g in self.graded if g.relevance == RELEVANT)


class RecordGrader:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        top_k: int = RERANK_TOP_K,
        max_record_chars: int = GRADER_MAX_RECORD_CHARS,
        min_relevant: int = CRAG_MIN_RELEVANT,
    ):
        self.llm_client = (
            llm_client
            or LLMClient(role="generator", temperature=0.0)
        )

        self.top_k = top_k
        self.max_record_chars = max_record_chars
        self.min_relevant = min_relevant

    def grade(self, question: str, records: list) -> GradeResult:

        records = list(records)

        # Zero records retrieved is already CRAG's "Incorrect" branch.
        # There is nothing to ask the model about, so don't spend a call
        # discovering that.
        if not records:
            return GradeResult(
                kept=[],
                graded=[],
                verdict="incorrect",
                warnings=["no records were retrieved"],
            )

        by_id = {str(r.id): r for r in records}

        raw = self.llm_client.generate(
            system_prompt=GRADER_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(question, records),
            response_format="json",
        )

        judgements, warnings = self._parse(raw, by_id)

        graded = self._apply_judgements(by_id, judgements, warnings)

        # Sort by the model's ordering, but only among records that
        # survived the filter. Python decides the branch, not the model.
        useful = [g for g in graded if g.is_useful]
        useful.sort(key=lambda g: g.rank)

        kept = [g.record for g in useful[: self.top_k]]

        # The verdict asks a stricter question than "did anything
        # survive the filter". Partially-relevant records are worth
        # generating from if that is all there is, but they are not
        # evidence that retrieval succeeded -- a local model will label
        # almost anything on-topic as partial. Requiring records it
        # called outright relevant is what makes the corrective branch
        # fire on a genuine miss.
        #
        # Note that `kept` stays populated either way: if the retry also
        # comes back empty, the caller still has these to fall back on
        # rather than nothing at all.
        relevant_count = sum(
            1 for g in graded if g.relevance == RELEVANT
        )

        verdict = (
            "correct"
            if relevant_count >= self.min_relevant
            else "incorrect"
        )

        return GradeResult(
            kept=kept,
            graded=graded,
            verdict=verdict,
            warnings=warnings,
        )

    def _build_user_prompt(self, question: str, records: list) -> str:

        blocks = []

        for record in records:

            text = str(record.display_text or "").strip()

            if len(text) > self.max_record_chars:
                text = text[: self.max_record_chars].rstrip() + " [...]"

            blocks.append(
                f"- id: {record.id}\n"
                f"  source: {record.source}\n"
                f"  title: {record.title}\n"
                f"  content: {text}"
            )

        records_text = "\n\n".join(blocks)

        return (
            f"CUSTOMER QUESTION:\n{question}\n\n"
            f"RETRIEVED RECORDS ({len(records)}):\n{records_text}\n\n"
            f"Grade every record above. Return ONLY the JSON object."
        )

    @staticmethod
    def _parse(raw: str, by_id: dict) -> tuple[dict, list]:
        """
        Turn the model's reply into {record_id: (relevance, rank)}.

        Every failure mode here is expected rather than exceptional:
        small models wrap JSON in prose, invent IDs, and use labels they
        were not offered. Each one is repaired and recorded instead of
        raising, because a malformed grade should degrade retrieval
        quality, not take down the request.
        """

        warnings = []

        text = (raw or "").strip()

        if not text:
            return {}, ["grader returned an empty response"]

        try:
            data = json.loads(text)

        except json.JSONDecodeError:
            # Last resort: pull the outermost {...} out of surrounding
            # prose. Cheaper than a retry and usually enough.
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                return {}, ["grader returned unparseable JSON"]

            try:
                data = json.loads(text[start : end + 1])
                warnings.append("grader JSON needed prose stripped")

            except json.JSONDecodeError:
                return {}, ["grader returned unparseable JSON"]

        if not isinstance(data, dict):
            return {}, ["grader JSON was not an object"]

        rows = data.get("records")

        if not isinstance(rows, list):
            return {}, ["grader JSON had no 'records' list"]

        judgements = {}

        for row in rows:

            if not isinstance(row, dict):
                warnings.append("skipped a non-object entry in 'records'")
                continue

            record_id = str(row.get("id", "")).strip()

            if record_id not in by_id:
                # The whitelist. An ID we did not retrieve cannot be
                # graded into the answer, however confident the model is.
                warnings.append(
                    f"discarded unknown record id '{record_id}'"
                )
                continue

            if record_id in judgements:
                warnings.append(
                    f"duplicate judgement for '{record_id}', kept first"
                )
                continue

            relevance = str(row.get("relevance", "")).strip().lower()

            if relevance not in VALID_RELEVANCE:
                warnings.append(
                    f"record '{record_id}' had unusable relevance "
                    f"'{relevance}', treated as {FALLBACK_RELEVANCE}"
                )
                relevance = FALLBACK_RELEVANCE

            try:
                rank = int(row.get("rank", UNRANKED))

            except (TypeError, ValueError):
                warnings.append(
                    f"record '{record_id}' had a non-integer rank"
                )
                rank = UNRANKED

            judgements[record_id] = (relevance, rank)

        return judgements, warnings

    @staticmethod
    def _apply_judgements(
        by_id: dict,
        judgements: dict,
        warnings: list,
    ) -> list:
        """
        Attach a judgement to every retrieved record, including the ones
        the model never mentioned.
        """

        graded = []

        for record_id, record in by_id.items():

            if record_id in judgements:
                relevance, rank = judgements[record_id]

            else:
                warnings.append(
                    f"grader did not judge '{record_id}', "
                    f"treated as {FALLBACK_RELEVANCE}"
                )
                relevance, rank = FALLBACK_RELEVANCE, UNRANKED

            graded.append(
                GradedRecord(
                    record=record,
                    relevance=relevance,
                    rank=rank,
                )
            )

        return graded
