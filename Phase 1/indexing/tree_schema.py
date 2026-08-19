"""
indexing/tree_schema.py

WHAT THIS FILE DOES
-------------------
Defines the data structures used by the LLM-generated hierarchical
tree index.

The tree is NOT the source data itself.

Instead, the tree is an index that organizes Records into a hierarchy
and points to the original records using their IDs.

Example:

    Banking Products
        |
        +-- Credit Cards
        |      |
        |      +-- Premium Cards
        |             |
        |             +-- record_ids: ["card_001", "card_007"]
        |
        +-- Cashback
               |
               +-- record_ids: ["campaign_003"]

WHY THIS FILE EXISTS
--------------------
The LLM will eventually generate JSON representing this structure.

We do not want the rest of the application working directly with
unvalidated dictionaries.

Instead:

    LLM JSON
        ↓
    TreeIndex
        ↓
    validation
        ↓
    TreeRetriever

This gives the tree system a stable contract.

IMPORTANT
---------
The tree contains references to Records.

It does NOT replace the original Record data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    node_id: str
    title: str
    summary: str = ""
    children: list["TreeNode"] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "record_ids": self.record_ids,
            "children": [
                child.to_dict()
                for child in self.children
            ],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TreeNode":
        if not isinstance(data, dict):
            raise ValueError("Tree node must be a dictionary.")

        children_data = data.get("children", [])

        if not isinstance(children_data, list):
            raise ValueError(
                f"Node '{data.get('node_id', '<unknown>')}' "
                f"has invalid 'children'. Expected a list."
            )

        record_ids = data.get("record_ids", [])

        if not isinstance(record_ids, list):
            raise ValueError(
                f"Node '{data.get('node_id', '<unknown>')}' "
                f"has invalid 'record_ids'. Expected a list."
            )

        return TreeNode(
            node_id=str(data.get("node_id", "")),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            record_ids=[
                str(record_id)
                for record_id in record_ids
            ],
            children=[
                TreeNode.from_dict(child)
                for child in children_data
            ],
        )


@dataclass
class TreeIndex:
    source: str
    root: TreeNode
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "metadata": self.metadata,
            "root": self.root.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TreeIndex":
        if not isinstance(data, dict):
            raise ValueError("Tree index must be a dictionary.")

        if "root" not in data:
            raise ValueError("Tree index is missing 'root'.")

        return TreeIndex(
            version=str(data.get("version", "1.0")),
            source=str(data.get("source", "")),
            metadata=data.get("metadata", {}),
            root=TreeNode.from_dict(data["root"]),
        )

    def validate(
        self,
        valid_record_ids: set[str] | None = None,
    ) -> None:
 
        if not self.source.strip():
            raise ValueError("Tree index source cannot be empty.")

        if not self.version.strip():
            raise ValueError("Tree index version cannot be empty.")

        if not isinstance(self.root, TreeNode):
            raise ValueError("Tree index root must be a TreeNode.")

        node_ids: set[str] = set()
        referenced_record_ids: list[str] = []

        def visit(node: TreeNode) -> None:

            if not isinstance(node, TreeNode):
                raise ValueError("Every child must be a TreeNode.")

            if not node.node_id.strip():
                raise ValueError(
                    "Every tree node must have a non-empty node_id."
                )

            if not node.title.strip():
                raise ValueError(
                    f"Tree node '{node.node_id}' "
                    f"must have a non-empty title."
                )

            if node.node_id in node_ids:
                raise ValueError(
                    f"Duplicate tree node_id detected: "
                    f"'{node.node_id}'."
                )

            node_ids.add(node.node_id)

            if not isinstance(node.record_ids, list):
                raise ValueError(
                    f"Node '{node.node_id}' "
                    f"record_ids must be a list."
                )

            for record_id in node.record_ids:

                if not isinstance(record_id, str):
                    raise ValueError(
                        f"Node '{node.node_id}' contains "
                        f"a non-string record ID."
                    )

                if not record_id.strip():
                    raise ValueError(
                        f"Node '{node.node_id}' contains "
                        f"an empty record ID."
                    )

                referenced_record_ids.append(record_id)

            if not isinstance(node.children, list):
                raise ValueError(
                    f"Node '{node.node_id}' children must be a list."
                )

            for child in node.children:
                visit(child)

        visit(self.root)

        if valid_record_ids is not None:

            unknown_records = (
                set(referenced_record_ids)
                - valid_record_ids
            )

            if unknown_records:
                raise ValueError(
                    "Tree contains references to unknown records: "
                    + ", ".join(sorted(unknown_records))
                )

    def get_all_nodes(self) -> list[TreeNode]:
    
        nodes: list[TreeNode] = []

        def visit(node: TreeNode) -> None:
            nodes.append(node)

            for child in node.children:
                visit(child)

        visit(self.root)

        return nodes

    def get_record_ids(self) -> list[str]:

        seen: set[str] = set()
        record_ids: list[str] = []

        for node in self.get_all_nodes():

            for record_id in node.record_ids:

                if record_id not in seen:
                    seen.add(record_id)
                    record_ids.append(record_id)

        return record_ids