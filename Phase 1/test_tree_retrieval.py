from loaders.campaigns_loader import CampaignsLoader
from retrievers.tree_retriever import TreeRetriever


def main():

    print("=" * 60)
    print("TREE RETRIEVER TEST")
    print("=" * 60)

    loader = CampaignsLoader()

    records = loader.load()

    print(
        f"Loaded {len(records)} campaign records."
    )

    retriever = TreeRetriever(
        source="campaigns",
        records=records,
    )

    questions = [
        "Which campaigns offer cashback?",
        "What campaigns are related to travel?",
        "Tell me about campaigns for cards.",
    ]

    for question in questions:

        print()
        print("-" * 60)
        print(f"QUESTION: {question}")
        print("-" * 60)

        results = retriever.retrieve(
            question,
            top_k=5,
        )

        print()
        print("TRACE:")
        print(retriever.last_trace)

        print()
        print("RETRIEVED RECORDS:")

        for record in results:

            print(
                f"- {record.id}: "
                f"{record.title}"
            )

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()