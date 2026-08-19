"""
LLM-powered hierarchical tree index builder.
"""

import json
from typing import Iterable

from config.settings import (
    TREE_INDEX_MAX_RETRIES,
)

from indexing.tree_schema import TreeIndex
from indexing.tree_prompt import (
    TREE_INDEX_SYSTEM_PROMPT,
    build_tree_user_prompt,
)
from llm.llm_client import LLMClient

class TreeBuilder:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ):
        self.llm_client = (
            llm_client
            or LLMClient(role="indexer")
        )

    def build(self, source: str,records: Iterable,) -> TreeIndex:
        """
        Build a validated TreeIndex from records using the indexing LLM.

        The LLM is responsible for semantic organization.

        Python is responsible for:
            - JSON parsing
            - structural validation
            - node ID assignment
            - record ID validation
        """

        records = list(records)

        if not records:
            raise ValueError(
                f"Cannot build tree for '{source}': "
                "no records were provided."
            )

        records_text = self._build_records_text(
            records
        )

        user_prompt = build_tree_user_prompt(
            source=source,
            records_text=records_text,
        )

        valid_record_ids = {
            str(record.id)
            for record in records
        }

        last_error = None

        for attempt in range(
            TREE_INDEX_MAX_RETRIES + 1
        ):

            print(
                f"[TREE INDEXER] "
                f"Attempt {attempt + 1}/"
                f"{TREE_INDEX_MAX_RETRIES + 1}"
            )

            try:

                raw_response = self.llm_client.generate(
                    system_prompt=TREE_INDEX_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response_format="json",
                )

                tree_data = self._parse_json(
                    raw_response
                )

                tree_data = self._normalize_root(
                    source=source,
                    tree_data=tree_data,
                )
                tree_data = (
                    self._remove_redundant_parent_records(
                        tree_data
                    )
                )

                self._validate_llm_tree(
                    tree_data=tree_data,
                    valid_record_ids=valid_record_ids,
                )

                tree_data = self._assign_node_ids(
                    tree_data
                )

                tree = TreeIndex.from_dict(
                    tree_data
                )

                tree.validate(
                    valid_record_ids
                )

                return tree

            except (
                ValueError,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ) as error:

                last_error = error

                print(
                    "[TREE INDEXER] "
                    f"Invalid output: {error}"
                )

                if attempt < TREE_INDEX_MAX_RETRIES:

                    print(
                        "[TREE INDEXER] "
                        "Retrying..."
                    )

                    user_prompt = (
                        build_tree_user_prompt(
                            source=source,
                            records_text=records_text,
                        )
                        + "\n\n"
                        + "IMPORTANT: Your previous output "
                        "was invalid.\n"
                        + f"Validation error: {error}\n"
                        + "Return a corrected JSON object."
                    )

        raise RuntimeError(
            f"Tree indexing failed for source "
            f"'{source}' after "
            f"{TREE_INDEX_MAX_RETRIES + 1} attempts. "
            f"Last error: {last_error}"
        )

    def _build_records_text(
        self,
        records: list,
    ) -> str:

        lines = []

        for record in records:

            lines.append(
                f"Record ID: {record.id}"
            )

            lines.append(
                f"Title: {record.title}"
            )

            lines.append(
                f"Source: {record.source}"
            )

            if record.metadata:

                metadata_items = []

                for key, value in record.metadata.items():

                    if value is None:
                        continue

                    metadata_items.append(
                        f"{key}={value}"
                    )

                if metadata_items:
                    lines.append(
                        "Metadata: "
                        + " | ".join(metadata_items)
                    )

            lines.append("---")

        return "\n".join(lines)

    def _normalize_root(
        self,
        source: str,
        tree_data: dict,
    ) -> dict:

        if not isinstance(tree_data, dict):
            raise ValueError(
                "Indexing LLM response must be a JSON object."
            )

        return {
            "version": "1.0",
            "source": source,
            "metadata": {
                "builder": "llm",
                "model": self.llm_client.model,
                "role": self.llm_client.role,
            },
            "root": tree_data,
        }

    def _assign_node_ids(
            
        self,
        tree_data: dict,
    ) -> dict:
        """
        Assign stable application-generated IDs to every node.

        The LLM is responsible for the hierarchy and node meaning.
        The application is responsible for node identity.

        This prevents malformed or missing LLM-generated node IDs
        from breaking the index.
        """

        counter = 0

        def visit(node: dict, path: str):

            nonlocal counter

            if not isinstance(node, dict):
                raise ValueError(
                    "Every tree node must be a JSON object."
                )

            counter += 1

            node["node_id"] = (
                f"node_{counter:04d}"
            )

            children = node.get(
                "children",
                [],
            )

            if children is None:
                children = []

            if not isinstance(children, list):
                raise ValueError(
                    f"Tree node '{node.get('title', 'unknown')}' "
                    "has invalid 'children'. "
                    "Expected a list."
                )

            node["children"] = children

            for index, child in enumerate(children):

                visit(
                    child,
                    f"{path}.{index}",
                )

        root = tree_data.get("root")

        if not isinstance(root, dict):
            raise ValueError(
                "Tree index must contain a valid root node."
            )

        visit(
            root,
            "root",
        )

        return tree_data


    def _parse_json(self, raw_response: str,) -> dict:
        """
        Parse the raw LLM response as JSON.
        """

        if not raw_response.strip():
            raise ValueError(
                "Indexer returned an empty response."
            )

        data = json.loads(
            raw_response
        )

        if not isinstance(data, dict):
            raise ValueError(
                "Indexer JSON root must be an object."
            )

        return data

    def _validate_llm_tree(self, tree_data: dict, valid_record_ids: set[str],) -> None:
        """
        Validate the LLM-generated tree BEFORE assigning application
        node IDs.

        This checks semantic structure rather than application IDs.
        """

        root = tree_data.get("root")

        if not isinstance(root, dict):
            raise ValueError(
                "Tree must contain a root object."
            )

        def visit(node: dict):

            if not isinstance(node, dict):
                raise ValueError(
                    "Every tree node must be an object."
                )

            title = node.get("title")

            if not isinstance(title, str):
                raise ValueError(
                    "Every tree node must have a title."
                )

            if not title.strip():
                raise ValueError(
                    "Every tree node must have a non-empty title."
                )

            summary = node.get(
                "summary",
                "",
            )

            if not isinstance(summary, str):
                raise ValueError(
                    f"Node '{title}' has invalid summary."
                )

            record_ids = node.get(
                "record_ids",
                [],
            )

            if not isinstance(record_ids, list):
                raise ValueError(
                    f"Node '{title}' must have "
                    "record_ids as a list."
                )

            for record_id in record_ids:

                if not isinstance(
                    record_id,
                    str,
                ):
                    raise ValueError(
                        f"Node '{title}' contains "
                        "a non-string record ID."
                    )

                if record_id not in valid_record_ids:

                    raise ValueError(
                        f"Node '{title}' references "
                        f"unknown record ID "
                        f"'{record_id}'."
                    )

            children = node.get(
                "children",
                [],
            )

            if not isinstance(
                children,
                list,
            ):
                raise ValueError(
                    f"Node '{title}' must have "
                    "children as a list."
                )

            for child in children:
                visit(child)

        visit(root)

    def _remove_redundant_parent_records(    self,    tree_data: dict,) -> dict:
        """
        Remove record IDs from parent nodes when the same records are
        already represented by descendant nodes.

        This keeps the tree traversal-oriented instead of turning parent
        categories into giant record buckets.
        """

        root = tree_data["root"]

        def process(node):

            children = node.get(
                "children",
                [],
            )

            for child in children:
                process(child)

            if not children:
                return

            descendant_record_ids = set()

            for child in children:
                descendant_record_ids.update(
                    child.get(
                        "record_ids",
                        [],
                    )
                )

            node_record_ids = node.get(
                "record_ids",
                [],
            )

            node["record_ids"] = [
                record_id
                for record_id in node_record_ids
                if record_id not in descendant_record_ids
            ]

        process(root)

        return tree_data