"""
indexing/record_batcher.py

WHAT THIS FILE DOES
-------------------
Splits a collection of Records into smaller batches before they are
sent to the indexing LLM.

WHY IT EXISTS
-------------
A local LLM should not be forced to understand hundreds of records
in a single prompt.

The batcher keeps the indexing module independent from the LLM itself.

INPUT:
    list[Record]

OUTPUT:
    list[list[Record]]

Example:

    23 records
        ↓
    batch size = 10
        ↓
    [
        [record 1 ... record 10],
        [record 11 ... record 20],
        [record 21 ... record 23]
    ]
"""

from typing import Iterable


class RecordBatcher:

    def __init__(self, batch_size: int = 10):

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.batch_size = batch_size

    def batch(
        self,
        records: Iterable,
    ) -> list[list]:

        records = list(records)

        return [
            records[start:start + self.batch_size]
            for start in range(
                0,
                len(records),
                self.batch_size,
            )
        ]