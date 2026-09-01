"""
LLM-based hierarchical tree retriever.

The retriever:
    1. Loads a persistent tree.
    2. Gives the tree structure to the traversal LLM.
    3. Asks the LLM which nodes are relevant.
    4. Resolves those nodes into Record IDs.
    5. Returns the original Records.

The LLM is used to navigate the index.

It is NOT used to generate the final answer here.
"""

import json

from indexing.tree_storage import TreeStorage
from llm.llm_client import LLMClient


TRAVERSAL_SYSTEM_PROMPT = """
You are a hierarchical tree index traversal system.

Your job is to navigate the supplied tree index and select the
smallest set of nodes that contain records useful for answering the
user's question.

IMPORTANT:

1. Return ONLY valid JSON.
2. Do not answer the question.
3. Do not invent node IDs.
4. Only select node IDs present in the tree.
5. Prefer specific nodes over broad parent nodes.
6. Do not select the root unless the question genuinely requires
   information spread across the entire source.
7. Do not select unrelated sibling branches.
8. If a specific leaf answers the question, select that leaf instead
   of its parent.
9. A node's record_ids identify the records represented by that node.
10. Selecting a broad node can retrieve many records, so avoid broad
    nodes when a more specific node is available.
11. If nothing is relevant, return an empty list.

The goal is HIGH PRECISION retrieval.

Required output:

{
  "selected_node_ids": [
    "node_0001"
  ]
}
"""


class TreeRetriever:

    def __init__(
        self,
        source: str,
        records: list,
        llm_client: LLMClient | None = None,
        storage: TreeStorage | None = None,
    ):
        self.source = source
        self.records = records

        self.llm_client = (
            llm_client
            or LLMClient(role="traverser")
        )

        self.storage = (
            storage
            or TreeStorage()
        )

        self.tree = self.storage.load(source)

        self.records_by_id = {
            str(record.id): record
            for record in records
        }

        self.nodes_by_id = {}
        self.node_depth = {}

        self._index_tree_nodes(
            self.tree.root
        )

        self.last_trace = None

        self.tree.validate(
            set(self.records_by_id.keys())
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list:

        if not query.strip():
            return []

        tree_text = self._build_tree_text()

        user_prompt = f"""
USER QUESTION:
{query}

TREE INDEX:
{tree_text}

TASK:

Navigate the tree and select ONLY the most specific nodes that are
relevant to the question.

Think of the tree as folders:

root
  -> category
      -> subcategory
          -> specific leaf
              -> records

Prefer the deepest relevant branch.

For example:

If the question asks about cashback, do NOT select a general
"Campaigns" node if a "Cashback" node exists.

If the question asks about Egypt Air, do NOT select the entire
"Installment Campaigns" branch if an "Egypt Air" node exists.

Return ONLY:

{{
  "selected_node_ids": ["..."]
}}
"""

        raw_response = self.llm_client.generate(
            system_prompt=TRAVERSAL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format="json",
        )

        selected_node_ids = self._parse_selected_nodes(
            raw_response
        )

        selected_nodes = [
            self.nodes_by_id[node_id]
            for node_id in selected_node_ids
            if node_id in self.nodes_by_id
        ]

        # Prefer selected leaf/specific nodes over their ancestors.
        #
        # If the LLM selected both:
        #
        #     node_0001 -> Campaigns
        #     node_0004 -> Cashback
        #
        # and node_0004 is a descendant of node_0001,
        # the broad parent does not add useful specificity.

        selected_nodes = self._remove_redundant_ancestors(
            selected_nodes
        )

        record_ids = self._merge_selected_records(selected_nodes, top_k)

        records = [
            self.records_by_id[record_id]
            for record_id in record_ids
            if record_id in self.records_by_id
        ]

        records = records[:top_k]

        self.last_trace = {
            "query": query,
            "selected_node_ids": selected_node_ids,
            "selected_nodes": [
                {
                    "node_id": node.node_id,
                    "title": node.title,
                }
                for node in selected_nodes
            ],
            "record_ids": [
                record.id
                for record in records
            ],
            "selected_node_titles": [
                node.title
                for node in selected_nodes
            ],
            "record_count": len(records),
        }

        return records

    def _merge_selected_records(self, selected_nodes: list, top_k: int) -> list:
        """
        Take records from EVERY selected node, most specific node first.

        THE BUG THIS REPLACES
        ---------------------
        The old version walked the selected nodes in the order the LLM
        happened to name them, appended each node's whole subtree, and
        only then cut to top_k. When the model selected one broad node
        and one specific node -- which the traversal prompt explicitly
        invites, and which is usually the RIGHT answer -- the broad
        node's subtree filled top_k before the specific node was
        reached at all.

        Asked "tell me about the egypt air campaign", the traversal
        correctly selected BOTH "Installment Offers" (186 records) and
        "Egypt air - Installment Offer" (1 record). Egypt Air sat at
        position 117 of the merged list, top_k was 5, and so the one
        record the customer actually asked about was the one record
        guaranteed to be dropped. What surfaced instead was the first
        five offers in tree order -- Mahgoub, Vevian, Carrefour --
        which is not a ranking of anything, just an alphabetical
        accident.

        Note _remove_redundant_ancestors could not help here: the two
        nodes were in different branches ("Travel And Airlines" vs
        "Installment Offers"), so neither was the other's ancestor.
        Both selections were legitimate; the merge was what lost one.

        THE RULE
        --------
        Round-robin across the selected nodes, deepest node first. Every
        node the model chose contributes its best record before any node
        contributes its second, so a specific selection can no longer be
        crowded out by a broad one. Within a node the original
        parents-before-children order is preserved.
        """

        # Deepest first: depth is the only signal available here for
        # "how specific was this selection", and it is exactly the
        # signal the traversal prompt asks the model to optimise for.
        ordered = sorted(
            selected_nodes,
            key=lambda node: self.node_depth.get(node.node_id, 0),
            reverse=True,
        )

        # A selected node stands for everything beneath it, not just the
        # records pinned to it directly. Category nodes carry no
        # record_ids of their own, so selecting "Installment" would
        # otherwise retrieve nothing at all.
        queues = [self._collect_record_ids(node) for node in ordered]

        merged = []
        seen = set()
        position = 0

        while any(position < len(q) for q in queues):

            for queue in queues:

                if position >= len(queue):
                    continue

                record_id = queue[position]

                if record_id in seen:
                    continue

                seen.add(record_id)
                merged.append(record_id)

                if len(merged) >= top_k:
                    return merged

            position += 1

        return merged

    def _build_tree_text(self) -> str:
        """
        Convert the tree into a compact representation for the
        traversal LLM.
        """

        lines = []

        def visit(node, depth: int):

            indent = "  " * depth

            lines.append(
                f"{indent}- node_id: {node.node_id}"
            )

            lines.append(
                f"{indent}  title: {node.title}"
            )

            if node.summary:

                lines.append(
                    f"{indent}  summary: {node.summary}"
                )

            if node.record_ids:

                lines.append(
                    f"{indent}  record_ids: "
                    f"{node.record_ids}"
                )

            for child in node.children:
                visit(child, depth + 1)

        visit(self.tree.root, 0)

        return "\n".join(lines)

    def _parse_selected_nodes(
        self,
        raw_response: str,
    ) -> list[str]:
        """
        Parse and validate the traversal LLM response.
        """

        try:
            data = json.loads(raw_response)

        except json.JSONDecodeError as e:
            raise ValueError(
                "Traversal LLM returned invalid JSON."
            ) from e

        selected = data.get(
            "selected_node_ids",
            [],
        )

        if not isinstance(selected, list):
            raise ValueError(
                "'selected_node_ids' must be a list."
            )

        valid_ids = []

        for node_id in selected:

            if not isinstance(node_id, str):
                continue

            if node_id in self.nodes_by_id:
                valid_ids.append(node_id)

        return valid_ids

    @staticmethod
    def _collect_record_ids(node) -> list[str]:
        """
        Every record ID in this node's subtree, parents before children,
        so broader context leads and detail follows.
        """

        record_ids = []

        def visit(current):

            for record_id in current.record_ids:

                if record_id not in record_ids:
                    record_ids.append(record_id)

            for child in current.children:
                visit(child)

        visit(node)

        return record_ids

    def _remove_redundant_ancestors(self, selected_nodes: list,) -> list:
        """
        Remove selected parent nodes when a selected descendant exists.

        This prevents the traversal model from unnecessarily retrieving
        broad branches when it has already selected a more specific branch.
        """

        selected_ids = {
            node.node_id
            for node in selected_nodes
        }

        # A selected node makes an ancestor redundant only when that
        # ancestor was ALSO selected. Walking down while carrying the
        # selected ancestors seen so far is what distinguishes "this
        # node has a selected parent" from "this node has any parent".
        redundant_ids = set()

        def visit(node, selected_ancestors: frozenset):

            if node.node_id in selected_ids and selected_ancestors:

                redundant_ids.update(
                    selected_ancestors
                )

            if node.node_id in selected_ids:

                selected_ancestors = (
                    selected_ancestors
                    | {node.node_id}
                )

            for child in node.children:
                visit(child, selected_ancestors)

        visit(
            self.tree.root,
            frozenset(),
        )

        return [
            node
            for node in selected_nodes
            if node.node_id not in redundant_ids
        ]

    def _index_tree_nodes(self, node, depth: int = 0):
        """
        Recursively index every node in the tree.

        The retriever needs O(1)-style lookup from the node ID returned
        by the traversal LLM to the actual TreeNode object, and the
        depth of each node so that a specific selection can be ranked
        ahead of a broad one -- see _merge_selected_records.
        """

        self.nodes_by_id[node.node_id] = node
        self.node_depth[node.node_id] = depth

        for child in node.children:
            self._index_tree_nodes(child, depth + 1)