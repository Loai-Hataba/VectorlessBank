"""
indexing/pageindex_tree_builder.py

WHAT THIS FILE DOES
-------------------
Builds a TreeIndex using the PageIndex library instead of asking a local
LLM to emit the whole hierarchy as JSON:

    records
        |
        v
    RecordCategorizer      (LLM: one short category per record)
        |
        v
    record_markdown        (headings: source / category / record)
        |
        v
    pageindex.md_to_tree   (headings -> tree, + optional node summaries)
        |
        v
    TreeIndex              (our existing schema, unchanged)

WHY IT EXISTS
-------------
It is a drop-in replacement for BatchedTreeBuilder: same build(source,
records) signature, same validated TreeIndex out. Everything downstream
-- TreeStorage, TreeRetriever, RagPipeline -- is untouched, because the
conversion at the end of this file lands on the existing schema.

WHAT CHANGED IN BEHAVIOUR
-------------------------
The hierarchy is now deterministic. PageIndex derives it from markdown
heading depth, so the two rules the old 1B indexer kept breaking --
"category nodes must have empty record_ids" and "attach records to the
most specific leaf" -- are now structurally impossible to violate.
Batching, adaptive splitting and tree merging are all gone with it: the
document is processed whole.

The LLM is still used, in two narrower places: choosing each record's
category, and (optionally) summarizing each node.

INPUTS  : source (str), list[Record]
OUTPUTS : a validated TreeIndex

REQUIRES: pip install pageindex; Ollama running locally.
"""

import asyncio
import os
import re

from config.settings import (
    MARKDOWN_DIR,
    OLLAMA_BASE_URL,
    PAGEINDEX_ADD_NODE_SUMMARY,
    PAGEINDEX_SUMMARY_MODEL,
    PAGEINDEX_SUMMARY_TOKEN_THRESHOLD,
    PAGEINDEX_MAX_SUMMARY_CHARS,
)

from indexing.record_categorizer import RecordCategorizer
from indexing.record_markdown import write_markdown
from indexing.tree_schema import TreeIndex


# "Some Card Name [card_1234]" -> ("Some Card Name", "card_1234").
# Greedy, so it binds to the LAST bracketed group: record IDs built from
# merchant names may themselves contain a bracket.
_RECORD_ID_IN_TITLE = re.compile(r"^(.*)\[(.+)\]\s*$")


class PageIndexTreeBuilder:

    def __init__(
        self,
        categorizer: RecordCategorizer | None = None,
        add_node_summary: str = PAGEINDEX_ADD_NODE_SUMMARY,
        summary_model: str = PAGEINDEX_SUMMARY_MODEL,
        summary_token_threshold: int = PAGEINDEX_SUMMARY_TOKEN_THRESHOLD,
        markdown_dir=MARKDOWN_DIR,
        max_summary_chars: int = PAGEINDEX_MAX_SUMMARY_CHARS,
    ):
        self.categorizer = (
            categorizer
            or RecordCategorizer()
        )

        self.add_node_summary = add_node_summary
        self.summary_model = summary_model
        self.summary_token_threshold = summary_token_threshold
        self.markdown_dir = markdown_dir
        self.max_summary_chars = max_summary_chars

    def build(self, source: str, records: list) -> TreeIndex:

        records = list(records)

        if not records:
            raise ValueError(
                "Cannot build a tree from zero records."
            )

        valid_record_ids = {
            str(record.id)
            for record in records
        }

        print(
            f"[PAGEINDEX] Categorizing {len(records)} records..."
        )

        categories = self.categorizer.categorize(records)

        markdown_path = write_markdown(
            source=source,
            records=records,
            categories=categories,
            markdown_dir=self.markdown_dir,
        )

        print(
            f"[PAGEINDEX] Wrote markdown: {markdown_path}"
        )

        result = self._run_md_to_tree(markdown_path)

        structure = result.get("structure") or []

        if not structure:
            raise ValueError(
                "PageIndex returned an empty structure for "
                f"source '{source}'."
            )

        root = self._build_root(
            source=source,
            structure=structure,
            valid_record_ids=valid_record_ids,
        )

        tree_data = {
            "version": "1.0",
            "source": source,
            "metadata": {
                "builder": "pageindex",
                "doc_name": result.get("doc_name", source),
                "line_count": result.get("line_count"),
                "markdown_path": str(markdown_path),
                "node_summaries": self.add_node_summary,
                "summary_model": (
                    self.summary_model
                    if self.add_node_summary == "yes"
                    else None
                ),
            },
            "root": root,
        }

        tree = TreeIndex.from_dict(tree_data)

        tree.validate(valid_record_ids)

        self._warn_about_missing_records(
            tree=tree,
            valid_record_ids=valid_record_ids,
        )

        return tree

    def _run_md_to_tree(self, markdown_path) -> dict:
        """
        Call PageIndex. Imported lazily so the rest of the project still
        works if the library is not installed.
        """

        try:
            from pageindex import md_to_tree

        except ImportError as error:
            raise RuntimeError(
                "PageIndex is not installed. "
                "Run: pip install pageindex"
            ) from error

        if self.add_node_summary == "yes":

            # PageIndex routes summaries through LiteLLM, which reads the
            # Ollama host from this variable rather than our settings.
            os.environ.setdefault(
                "OLLAMA_API_BASE",
                OLLAMA_BASE_URL,
            )

            print(
                f"[PAGEINDEX] Generating node summaries with "
                f"{self.summary_model}..."
            )

        return asyncio.run(
            md_to_tree(
                str(markdown_path),
                if_add_node_summary=self.add_node_summary,
                summary_token_threshold=self.summary_token_threshold,
                model=self.summary_model,
                if_add_node_text="no",
                if_add_node_id="yes",
            )
        )

    def _build_root(
        self,
        source: str,
        structure: list,
        valid_record_ids: set,
    ) -> dict:
        """
        PageIndex returns a FOREST (a list of top-level headings), but
        TreeIndex requires a single root.

        record_markdown emits exactly one '#' heading, so there is
        normally one entry -- but a synthetic root is used otherwise so
        the conversion never depends on that.
        """

        converted = [
            self._convert_node(node, valid_record_ids)
            for node in structure
        ]

        if len(converted) == 1:
            return converted[0]

        return {
            "node_id": "node_root",
            "title": source.replace("_", " ").title(),
            "summary": f"All indexed records for {source}.",
            "record_ids": [],
            "children": converted,
        }

    def _convert_node(
        self,
        node: dict,
        valid_record_ids: set,
    ) -> dict:
        """
        Convert one PageIndex node into our TreeNode dict shape.

        PageIndex node: {title, node_id, line_num, summary?,
                         prefix_summary?, nodes?}
        Our node:       {node_id, title, summary, record_ids, children}
        """

        title = str(node.get("title", "")).strip()

        record_ids = []

        match = _RECORD_ID_IN_TITLE.match(title)

        if match:

            candidate_title = match.group(1).strip()
            candidate_id = match.group(2).strip()

            if candidate_id in valid_record_ids:

                record_ids = [candidate_id]

                title = candidate_title or candidate_id

            else:
                # Not one of ours -- a category that happened to end in
                # brackets. Leave the title exactly as written.
                print(
                    f"[PAGEINDEX] Ignoring unknown record ID "
                    f"'{candidate_id}' in heading."
                )

        # Leaves get 'summary'; nodes with children get 'prefix_summary'.
        summary = (
            node.get("summary")
            or node.get("prefix_summary")
            or ""
        )

        summary = str(summary).replace("\n", " ").strip()

        if len(summary) > self.max_summary_chars:
            summary = summary[: self.max_summary_chars].rstrip() + "..."

        children = [
            self._convert_node(child, valid_record_ids)
            for child in (node.get("nodes") or [])
        ]

        return {
            "node_id": f"node_{node.get('node_id', '')}",
            "title": title or "Untitled",
            "summary": summary,
            "record_ids": record_ids,
            "children": children,
        }

    @staticmethod
    def _warn_about_missing_records(
        tree: TreeIndex,
        valid_record_ids: set,
    ) -> None:
        """
        tree.validate() rejects INVENTED record IDs. This catches the
        opposite failure: records that silently never made it in.
        """

        indexed = set(tree.get_record_ids())

        missing = valid_record_ids - indexed

        if missing:
            print(
                f"[PAGEINDEX] WARNING: {len(missing)} record(s) "
                f"are not referenced by any node: "
                f"{', '.join(sorted(missing))}"
            )
