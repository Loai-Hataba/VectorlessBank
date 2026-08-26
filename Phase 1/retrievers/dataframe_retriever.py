"""
retrievers/dataframe_retriever.py

WHAT THIS FILE DOES
--------------------
The second retrieval technique's query-time half: given a question,
asks pipeline/querier.py to decide which column filters apply, builds
the corresponding pandas boolean mask itself (never the model), and
resolves the matching rows back to real Record objects.

Deliberately matches TreeRetriever's retrieve(question, top_k) ->
list[Record] interface exactly, so RagPipeline or a comparison
harness can hold both retrievers side by side and call either one
identically -- neither needs to know the other exists.

WHY PYTHON BUILDS THE MASK, NOT THE MODEL
--------------------------------------------
Every filter DataFrameQuerier returns has already been validated
against real column names and a fixed operator whitelist. This file
is where those validated pieces are turned into an actual pandas
operation -- one explicit branch per whitelisted operator, nothing
dynamic, nothing eval()'d. A filter that survived validation can only
ever produce one of these seven exact operations.

INPUTS  : source (str), df (pandas DataFrame), records (list[Record])
OUTPUTS : retrieve(question, top_k) -> list[Record]
"""

import pandas as pd

from pipeline.querier import DataFrameQuerier


class DataFrameRetriever:

    def __init__(
        self,
        source: str,
        df: pd.DataFrame,
        records: list,
        querier: DataFrameQuerier | None = None,
    ):
        self.source = source
        self.df = df
        self.records_by_id = {r.id: r for r in records}
        self.querier = querier or DataFrameQuerier()

    def retrieve(
        self,
        question: str,
        top_k: int,
        session_id: str = "",
        turn_id: str = "",
    ) -> list:

        filters = self.querier.build_filters(
            question,
            self.df,
            session_id=session_id,
            turn_id=turn_id,
        )

        if not filters:
            # No filter the querier proposed survived validation, or it
            # explicitly found nothing plausible. This is a legitimate
            # "no match" result, identical in meaning to an empty list
            # from TreeRetriever -- not an error, nothing to fall back to.
            return []

        mask = self._build_mask(filters)

        matched_ids = self.df.loc[mask, "id"].tolist()

        records = [
            self.records_by_id[record_id]
            for record_id in matched_ids
            if record_id in self.records_by_id
        ]

        return records[:top_k]

    def _build_mask(self, filters: list[dict]) -> pd.Series:
        """
        Builds the combined boolean mask from validated filters only.
        Every branch here corresponds exactly to one entry in
        config.settings.VALID_FILTER_OPS -- there is no path from a
        filter dict to anything other than one of these seven
        operations, and querier.py has already rejected anything with
        an op outside this set before it reaches here.
        """

        mask = pd.Series(True, index=self.df.index)

        for f in filters:

            column, op, value = f["column"], f["op"], f["value"]
            series = self.df[column]

            if op == "==":
                mask &= (series == value)
            elif op == "!=":
                mask &= (series != value)
            elif op == "contains":
                mask &= series.astype(str).str.contains(
                    str(value), case=False, na=False
                )
            elif op == ">":
                mask &= (series > value)
            elif op == "<":
                mask &= (series < value)
            elif op == ">=":
                mask &= (series >= value)
            elif op == "<=":
                mask &= (series <= value)

        return mask