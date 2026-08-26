"""
pipeline/querier.py

WHAT THIS FILE DOES
--------------------
Second retrieval technique, built for flat/columnar sources (cards,
offers) where forcing a tree hierarchy onto the data doesn't reflect
any real structure in it. Given a question and a DataFrame's schema,
an LLM decides which column filters would find the answering rows.
Built specifically to be compared against tree-based retrieval on the
same questions -- see evaluation/compare_retrieval.py.

WHY STRUCTURED FILTERS, NEVER GENERATED CODE
---------------------------------------------
The tempting version of this lets the model write literal pandas code
and exec() it. That is a real code-execution surface sitting on top of
bank data: prompt injection hidden inside a record's own text (a
merchant name, a benefit description) could attempt to smuggle
instructions into what looks like an innocent filter request, and
exec() would run whatever it produced.

Instead, the model's ONLY output is a JSON list of
{column, op, value} triples -- never a string of code. Python
validates every column name against the DataFrame's real, current
columns and every operator against a fixed whitelist, and Python
alone builds and executes the actual boolean mask. The model never
gets anything resembling execution. This is the same "model proposes,
Python disposes" discipline used everywhere else in this project
(tree traversal's node-id whitelist, the router's source whitelist,
CRAG's relevance-label whitelist), applied here because this is the
one place a naive implementation would have broken that discipline
outright rather than just risking a wrong answer.

WHY THE SCHEMA DESCRIPTION CALLS OUT SPARSITY EXPLICITLY
------------------------------------------------------------
indexing/dataframe_builder.py flattens each record's metadata into
columns, which produces a wide table where most cells are NaN (a
disambiguated column like "Benefits #30" only has a value for the
handful of cards with 30+ benefits). Without being told this, a model
asked to filter on "Benefits" might reasonably expect one column and
be confused to find forty sparsely-populated ones. The schema text
says this outright rather than letting the model guess.

INPUTS  : question (str), pandas DataFrame
OUTPUTS : list[dict] -- validated {column, op, value} filters, safe
          to hand directly to DataFrameRetriever's mask builder.
          Empty list means "no filter could be built", not an error.
"""

import json

from config.settings import VALID_FILTER_OPS, QUERIER_MAX_SAMPLE_ROWS
from llm.llm_client import LLMClient
from pipeline.pipeline_logger import log_stage


QUERIER_SYSTEM_PROMPT = """You write data filters. You do not write code.

You are given a table's column names, their data types, and a few
sample rows, plus a customer question. Decide which filter(s) would
find the rows that answer it.

Many columns are sparsely populated -- for example, several numbered
columns like "Benefits #1", "Benefits #2" exist because different
products list different numbers of benefits, so most of those columns
are empty for most rows. This is expected. Prefer filtering on columns
that are clearly populated for the concept the question asks about
(e.g. a fee column for fee questions, a segment column for tier
questions) over guessing which numbered variant might hold a value.

Return ONLY JSON in exactly this shape, and nothing else:
{"filters": [{"column": "Segment", "op": "==", "value": "Premium"}]}

Valid "op" values, exactly as written, nothing else:
== != contains > < >= <=

Use "contains" for partial text matches such as merchant or product
names. Use the EXACT column name as given to you -- never invent one,
never guess a close spelling. If no column in the table plausibly
answers the question, return {"filters": []}.
"""


class DataFrameQuerier:

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient(role="querier")

    def build_filters(
        self,
        question: str,
        df,
        session_id: str = "",
        turn_id: str = "",
    ) -> list[dict]:

        schema_text = self._describe_schema(df)

        raw = self.llm_client.generate(
            system_prompt=QUERIER_SYSTEM_PROMPT,
            user_prompt=(
                f"TABLE SCHEMA:\n{schema_text}\n\n"
                f"QUESTION: {question}"
            ),
            response_format="json",
        )

        filters, warnings = self._parse_and_validate(raw, df)

        log_stage(
            stage="querier",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "question": question,
                "raw_llm_output": raw,
                "filters": filters,
                "warnings": warnings,
            },
        )

        return filters

    def _describe_schema(self, df) -> str:

        columns_text = "\n".join(
            f"- {col} ({df[col].dtype}, "
            f"{df[col].notna().sum()}/{len(df)} rows populated)"
            for col in df.columns
        )

        sample = df.head(QUERIER_MAX_SAMPLE_ROWS).to_string(index=False)

        return (
            f"Columns (name, type, how many rows actually have a "
            f"value):\n{columns_text}\n\n"
            f"Sample rows:\n{sample}"
        )

    def _parse_and_validate(self, raw: str, df) -> tuple[list[dict], list[str]]:
        """
        Every failure mode here degrades to an empty filter list
        rather than raising: an unusable querier response should mean
        "this technique found nothing", exactly like a real no-match
        case, not a crash. The caller (DataFrameRetriever) treats an
        empty list as a normal, valid outcome.
        """

        warnings = []

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [], ["querier returned unparseable JSON"]

        if not isinstance(data, dict):
            return [], ["querier JSON was not an object"]

        raw_filters = data.get("filters")

        if not isinstance(raw_filters, list):
            return [], ["querier JSON had no 'filters' list"]

        valid_columns = set(df.columns)
        valid = []

        for f in raw_filters:

            if not isinstance(f, dict):
                warnings.append("skipped a non-object filter entry")
                continue

            column = f.get("column")
            op = f.get("op")
            value = f.get("value")

            if column not in valid_columns:
                warnings.append(
                    f"discarded filter on unknown column '{column}'"
                )
                continue

            if op not in VALID_FILTER_OPS:
                warnings.append(
                    f"discarded filter with invalid op '{op}' "
                    f"on column '{column}'"
                )
                continue

            valid.append({"column": column, "op": op, "value": value})

        return valid, warnings