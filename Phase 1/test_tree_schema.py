from indexing.tree_schema import TreeIndex, TreeNode


def main():

    tree = TreeIndex(
        source="campaigns",
        root=TreeNode(
            node_id="campaigns",
            title="Campaigns",
            summary="All banking campaigns",
            children=[
                TreeNode(
                    node_id="cashback",
                    title="Cashback Campaigns",
                    summary="Campaigns that provide cashback",
                    record_ids=[
                        "campaign_001",
                        "campaign_002",
                    ],
                ),
                TreeNode(
                    node_id="travel",
                    title="Travel Campaigns",
                    summary="Campaigns related to travel",
                    record_ids=[
                        "campaign_003",
                    ],
                ),
            ],
        ),
    )

    valid_record_ids = {
        "campaign_001",
        "campaign_002",
        "campaign_003",
    }

    tree.validate(valid_record_ids)

    print("Tree validation: PASSED")

    print()
    print("All nodes:")

    for node in tree.get_all_nodes():
        print(
            f"- {node.node_id}: "
            f"{node.title} "
            f"(records={node.record_ids})"
        )

    print()
    print("Record IDs:")
    print(tree.get_record_ids())

    print()
    print("Serialized tree:")

    data = tree.to_dict()

    print(data)

    print()
    print("Deserializing...")

    rebuilt_tree = TreeIndex.from_dict(data)

    rebuilt_tree.validate(valid_record_ids)

    print("Serialization/deserialization: PASSED")


if __name__ == "__main__":
    main()