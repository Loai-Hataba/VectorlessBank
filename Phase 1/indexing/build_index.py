"""
Command-line entry point for building persistent tree indexes.

Examples:

    python -m indexing.build_index campaigns
    python -m indexing.build_index offers
    python -m indexing.build_index cards
"""

import sys

from indexing.tree_storage import TreeStorage
from indexing.batched_tree_builder import (
    BatchedTreeBuilder,
)

from loaders.campaigns_loader import CampaignsLoader
from loaders.offers_loader import OffersLoader
from loaders.cards_loader import CardsLoader


LOADERS = {
    "campaigns": CampaignsLoader,
    "offers": OffersLoader,
    "cards": CardsLoader,
}


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python -m indexing.build_index campaigns\n"
            "  python -m indexing.build_index offers\n"
            "  python -m indexing.build_index cards"
        )

        raise SystemExit(1)

    source = sys.argv[1].lower()

    if source not in LOADERS:

        print(
            f"Unknown source '{source}'.\n"
            f"Available sources: "
            f"{', '.join(LOADERS.keys())}"
        )

        raise SystemExit(1)

    loader_class = LOADERS[source]

    print("=" * 60)
    print("TREE INDEX BUILD")
    print("=" * 60)
    print(f"Source: {source}")
    print()

    loader = loader_class()

    print("Loading records...")

    records = loader.load()

    print(
        f"Loaded {len(records)} records."
    )

    print()
    print("Building tree with local LLM...")

    builder = BatchedTreeBuilder()

    tree = builder.build(
        source=source,
        records=records,
    )

    print(
        f"Generated {len(tree.get_all_nodes())} tree nodes."
    )

    print(
        f"Referenced {len(tree.get_record_ids())} records."
    )

    print()
    print("Saving tree index...")

    storage = TreeStorage()

    path = storage.save(tree)

    print(f"Saved: {path}")

    print()
    print("=" * 60)
    print("TREE INDEX BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()