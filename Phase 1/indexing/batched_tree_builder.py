"""
indexing/batched_tree_builder.py

WHAT THIS FILE DOES
-------------------
Builds a complete tree index for a large source by:

    records
        ↓
    batches
        ↓
    local LLM-generated trees
        ↓
    tree merge
        ↓
    global TreeIndex

WHY IT EXISTS
-------------
A local 1B model should not receive hundreds of records in one prompt.

TreeBuilder handles ONE manageable group.

BatchedTreeBuilder handles the complete dataset.
"""

import json

from config.settings import (
    TREE_INDEX_BATCH_SIZE,
)

from indexing.record_batcher import RecordBatcher
from indexing.tree_builder import TreeBuilder
from indexing.tree_schema import TreeIndex
from llm.llm_client import LLMClient


MERGE_SYSTEM_PROMPT = """
You are a hierarchical index merging system.

You are given several partial trees describing different batches of
records from the same data source.

Your task is to merge them into ONE coherent hierarchical index.

Rules:

1. Return ONLY valid JSON.
2. Do not return markdown.
3. Do not explain your answer.
4. Preserve every valid Record ID.
5. Never invent Record IDs.
6. Do not generate node IDs.
7. The application will assign node IDs.
8. Every node must have a non-empty title.
9. Merge nodes that represent the same concept.
10. Do not create duplicate categories when they can be combined.
11. Prefer a meaningful hierarchy from broad concepts to specific ones.
12. Records should normally live in specific leaf/category nodes.
13. Do not put every record directly under the root.
14. Do not lose records during the merge.
15. Do not invent information that does not exist in the partial trees.

Required structure:

{
  "title": "...",
  "summary": "...",
  "record_ids": [],
  "children": []
}
"""


class BatchedTreeBuilder:

    def __init__(
        self,
        indexer: TreeBuilder | None = None,
        merger: LLMClient | None = None,
        batch_size: int = TREE_INDEX_BATCH_SIZE,
    ):

        self.indexer = (
            indexer
            or TreeBuilder()
        )

        self.merger = (
            merger
            or LLMClient(role="indexer")
        )

        self.batcher = RecordBatcher(
            batch_size=batch_size
        )

    def build(self, source: str, records: list,) -> TreeIndex:

        records = list(records)

        if not records:
            raise ValueError(
                "Cannot build a tree from zero records."
            )

        initial_batches = self.batcher.batch(
            records
        )

        print(
            f"[BATCH INDEXER] "
            f"Split {len(records)} records into "
            f"{len(initial_batches)} initial batches."
        )

        partial_trees = []

        for index, batch in enumerate(
            initial_batches,
            start=1,
        ):

            print(
                f"[BATCH INDEXER] "
                f"Processing initial batch "
                f"{index}/{len(initial_batches)} "
                f"({len(batch)} records)"
            )

            trees = self._build_adaptive_batch(
                source=source,
                records=batch,
            )

            partial_trees.extend(
                trees
            )

        print(
            f"[BATCH INDEXER] "
            f"Successfully built "
            f"{len(partial_trees)} partial trees."
        )

        if len(partial_trees) == 1:
            return partial_trees[0]

        return self._merge_trees(
            source=source,
            partial_trees=partial_trees,
            all_records=records,
        )

    def _merge_trees(
        self,
        source: str,
        partial_trees: list[TreeIndex],
        all_records: list,
    ) -> TreeIndex:

        trees_text = []

        for index, tree in enumerate(
            partial_trees,
            start=1,
        ):

            trees_text.append(
                f"PARTIAL TREE {index}:\n"
                + json.dumps(
                    tree.to_dict()["root"],
                    indent=2,
                    ensure_ascii=False,
                )
            )

        user_prompt = f"""
Merge the following partial trees into one global tree.

SOURCE:
{source}

PARTIAL TREES:

\n\n{trees_text}

IMPORTANT:

The complete dataset contains these Record IDs:

{
    [str(record.id) for record in all_records]
}

The final tree MUST preserve all of these records.

Do not invent IDs.

Do not generate node IDs.

Return ONLY the root tree JSON object.
"""

        raw_response = self.merger.generate(
            system_prompt=MERGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format="json",
        )

        try:

            root = json.loads(
                raw_response
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Tree merger returned invalid JSON."
            ) from error

        if not isinstance(root, dict):

            raise ValueError(
                "Tree merger must return "
                "a JSON object."
            )

        tree_data = {
            "version": "1.0",
            "source": source,
            "metadata": {
                "builder": "batched_llm",
                "model": self.merger.model,
                "batch_count": len(
                    partial_trees
                ),
            },
            "root": root,
        }

        tree_data = (
            self.indexer._normalize_root(
                source=source,
                tree_data=tree_data,
            )
        )

        tree_data = (
            self.indexer._validate_llm_tree(
                tree_data=tree_data,
                valid_record_ids={
                    str(record.id)
                    for record in all_records
                },
            )
        )

        tree_data = (
            self.indexer._assign_node_ids(
                tree_data
            )
        )

        tree = TreeIndex.from_dict(
            tree_data
        )

        tree.validate(
            {
                str(record.id)
                for record in all_records
            }
        )

        return tree

    def _build_adaptive_batch(self, source: str, records: list,) -> list[TreeIndex]:
        """
        Try to index a batch.

        If the local LLM cannot successfully process the batch, split it
        into smaller batches and try again.

        This is recursive so the system can automatically find a workable
        batch size for different record shapes.
        """

        try:

            tree = self.indexer.build(
                source=source,
                records=records,
            )

            return [tree]

        except (
            RuntimeError,
            ValueError,
            TypeError,
        ) as error:

            print(
                f"[BATCH INDEXER] "
                f"Batch of {len(records)} records failed:"
            )

            print(
                f"[BATCH INDEXER] {error}"
            )

            if len(records) <= 1:

                raise RuntimeError(
                    "A single record could not be indexed "
                    f"for source '{source}'."
                ) from error

            midpoint = len(records) // 2

            left = records[:midpoint]
            right = records[midpoint:]

            print(
                f"[BATCH INDEXER] "
                f"Splitting {len(records)} records "
                f"into {len(left)} + {len(right)}"
            )

            left_trees = self._build_adaptive_batch(
                source=source,
                records=left,
            )

            right_trees = self._build_adaptive_batch(
                source=source,
                records=right,
            )

            return (
                left_trees
                + right_trees
            )