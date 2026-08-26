"""
indexing/dataframe_builder.py

WHAT THIS FILE DOES
--------------------
Converts a list of Records into ONE pandas DataFrame and persists it,
for sources where a flat table is a more natural fit than a tree (see
config.settings.DATAFRAME_SOURCES). This is the second of two indexing
techniques built over the same source data, specifically so the two
can be compared on identical questions (evaluation/compare_retrieval.py)
rather than one being assumed better than the other.

WHY METADATA IS FLATTENED, NOT COPIED AS-IS
--------------------------------------------
Record.to_dict() is correct for its own purpose (caching, logging) but
wrong for this one: it puts every card's ~50-300 business attributes
inside one nested "metadata" dict field. Building a DataFrame straight
from that gives a single "metadata" column of dtype object -- pandas
cannot filter, compare, or type-check the inside of that dict, which
defeats the entire point of this technique. The querier LLM needs real
columns (Annual fee, Segment, Minimum limit, ...) to write filters
against, so metadata is expanded into top-level columns here, one
column per distinct attribute key seen across all records in the
source.

WHY THE RESULT IS WIDE AND SPARSE, AND WHY THAT'S LEFT AS-IS
--------------------------------------------------------------------
Cards disambiguates repeated attribute names with a counter
("Benefits #1".."Benefits #40"), and different cards have different
numbers of benefits, minimum limits, etc. Flattening across all
records therefore produces a wide table where most cells for any
given row are NaN (a card with 2 benefits has nothing in "Benefits
#15"..."Benefits #40"). This is not cleaned up or renamed here,
because doing so would mean this file is silently making editorial
decisions about the source data. The querier is told explicitly, in
its schema description, that sparsity is expected -- see querier.py.

INPUTS  : source (str), list[Record]
OUTPUTS : a pandas DataFrame, saved to
          config.settings.DATAFRAME_INDEX_DIR / "<source>.parquet"
"""

from pathlib import Path

import pandas as pd

from config.settings import DATAFRAME_INDEX_DIR


class DataFrameBuilder:

    def build(self, source: str, records: list) -> pd.DataFrame:

        if not records:
            raise ValueError(
                f"Cannot build a dataframe from zero records for "
                f"source '{source}'."
            )

        rows = []

        for record in records:

            row = {
                "id": record.id,
                "source": record.source,
                "title": record.title,
            }

            # Flatten: every metadata key becomes its own column. Later
            # records introduce new columns automatically; pandas fills
            # NaN for records that didn't have that key, which is the
            # correct representation of "this card has no Benefits #12",
            # not a bug to paper over.
            row.update(record.metadata)

            rows.append(row)

        df = pd.DataFrame(rows)
        df.attrs["source"] = source

        return df

    def save(self, source: str, df: pd.DataFrame) -> Path:

        DATAFRAME_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        path = DATAFRAME_INDEX_DIR / f"{source}.parquet"

        # Parquet over pickle: it's a standard columnar format any
        # teammate can inspect with any tool, not a Python-version-
        # specific pickle blob. The dtype inference from build() above
        # is preserved on reload.
        df.to_parquet(path)

        return path

    def load(self, source: str) -> pd.DataFrame:

        path = DATAFRAME_INDEX_DIR / f"{source}.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"No dataframe index found for '{source}' at {path}. "
                f"Run the dataframe build step for this source first."
            )

        return pd.read_parquet(path)