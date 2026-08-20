"""
indexing/record_categorizer.py

WHAT THIS FILE DOES
-------------------
Asks the local indexing LLM to assign ONE short category label to each
Record, e.g.

    campaign_merchant_egypt_air_2026_07  ->  "Travel And Airlines"
    campaign_merchant_el_araby_2026_07   ->  "Home Appliances"

WHY IT EXISTS
-------------
PageIndex builds its tree from markdown heading levels, deterministically.
It does NOT reason about how records should be grouped -- whatever
headings we emit ARE the tree, one for one.

So the semantic grouping that the old TreeBuilder asked the indexer LLM
to invent now happens here instead, one record at a time. A per-record
question ("what is this?") is a much easier task for a 1B model than
"design a whole hierarchy and emit valid nested JSON", which is what the
old indexer prompt asked for.

The category labels produced here become the '##' headings in
record_markdown.py, which become the middle layer of the PageIndex tree.

INPUTS  : list[Record]
OUTPUTS : dict mapping record.id -> category label (str)

REQUIRES: Ollama running locally with the indexer model pulled.
"""

import json
import re

from llm.llm_client import LLMClient


CATEGORY_SYSTEM_PROMPT = """
You are a record classification system for a retail bank.

You are given ONE record. Assign it a single short category label.

Rules:

1. Return ONLY valid JSON.
2. Do not return markdown.
3. Do not explain your answer.
4. The label must be 1 to 3 words.
5. The label must describe what the record IS about, not its ID.
6. Reuse an existing category if one of them fits the record.
7. Only invent a new category when no existing category fits.
8. Do not put the record's own name in the label.

Required output:

{"category": "Travel And Airlines"}
"""

# A category label is a markdown heading, so keep it to plain text.
_UNSAFE_LABEL_CHARS = re.compile(r"[\[\]#*`_\r\n]")

# Metadata values can be enormous (the cards sheet has ~366 columns).
# The classifier only needs a flavour of the record, not all of it.
MAX_METADATA_FIELDS = 12
MAX_VALUE_LENGTH = 120


class RecordCategorizer:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        fallback_category: str = "Other",
    ):
        self.llm_client = (
            llm_client
            or LLMClient(role="indexer")
        )

        self.fallback_category = fallback_category

    def categorize(self, records: list) -> dict:
        """
        Return {record_id: category_label} for every supplied record.

        Categories already assigned are fed back into the prompt so the
        model converges on a small, reusable set of labels instead of
        inventing a brand new one for every single record.
        """

        categories = {}
        known_labels = []

        for index, record in enumerate(records, start=1):

            label = self._categorize_one(
                record=record,
                known_labels=known_labels,
            )

            print(
                f"[CATEGORIZER] "
                f"{index}/{len(records)} "
                f"{record.id} -> {label}"
            )

            categories[str(record.id)] = label

            if label not in known_labels:
                known_labels.append(label)

        return categories

    def _categorize_one(
        self,
        record,
        known_labels: list[str],
    ) -> str:

        user_prompt = self._build_user_prompt(
            record=record,
            known_labels=known_labels,
        )

        try:

            raw_response = self.llm_client.generate(
                system_prompt=CATEGORY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_format="json",
            )

            data = json.loads(raw_response)

            label = data.get("category", "")

        except (RuntimeError, ValueError, TypeError, AttributeError) as error:

            # One flaky classification must not kill a whole index build.
            # The record still ends up in the tree, just under "Other".
            print(
                f"[CATEGORIZER] "
                f"Falling back for {record.id}: {error}"
            )

            return self.fallback_category

        return self._normalize_label(
            label=label,
            known_labels=known_labels,
        )

    def _build_user_prompt(
        self,
        record,
        known_labels: list[str],
    ) -> str:

        if known_labels:
            existing = "\n".join(
                f"- {label}"
                for label in known_labels
            )
        else:
            existing = "(none yet -- you are classifying the first record)"

        return f"""
Classify this record.

EXISTING CATEGORIES (reuse one of these if it fits):
{existing}

RECORD:
Title: {record.title}
Source: {record.source}
{self._summarize_metadata(record)}

Return ONLY:

{{"category": "..."}}
"""

    @staticmethod
    def _summarize_metadata(record) -> str:

        if not record.metadata:
            return ""

        lines = []

        for key, value in record.metadata.items():

            if value is None or value == "":
                continue

            text = str(value).replace("\n", " ").strip()

            if len(text) > MAX_VALUE_LENGTH:
                text = text[:MAX_VALUE_LENGTH] + "..."

            lines.append(f"{key}: {text}")

            if len(lines) >= MAX_METADATA_FIELDS:
                break

        if not lines:
            return ""

        return "Details:\n" + "\n".join(lines)

    def _normalize_label(
        self,
        label,
        known_labels: list[str],
    ) -> str:
        """
        Turn whatever the model said into a safe, stable heading label.

        Matching case-insensitively against labels we have already used
        stops "Travel", "travel" and "TRAVEL" from becoming three
        separate branches of the tree.
        """

        if not isinstance(label, str):
            return self.fallback_category

        label = _UNSAFE_LABEL_CHARS.sub(" ", label)
        label = re.sub(r"\s+", " ", label).strip()

        if not label:
            return self.fallback_category

        # Keep labels short enough to read as a heading.
        words = label.split(" ")

        if len(words) > 4:
            label = " ".join(words[:4])

        label = label.title()

        for existing in known_labels:
            if existing.lower() == label.lower():
                return existing

        return label
