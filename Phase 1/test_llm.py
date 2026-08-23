# from llm.llm_client import LLMClient


# def main():

#     print("=" * 60)
#     print("TEST 1 — Generator")
#     print("=" * 60)

#     generator = LLMClient(role="generator")

#     print(f"Role: {generator.role}")
#     print(f"Model: {generator.model}")
#     print(f"Temperature: {generator.temperature}")
#     print()

#     response = generator.generate(
#         system_prompt="You are a helpful assistant.",
#         user_prompt="Say hello and explain in one short sentence that you are running locally.",
#     )

#     print("MODEL RESPONSE:")
#     print(response)

#     print()
#     print("=" * 60)
#     print("TEST 2 — Indexer")
#     print("=" * 60)

#     indexer = LLMClient(role="indexer")

#     print(f"Role: {indexer.role}")
#     print(f"Model: {indexer.model}")
#     print(f"Temperature: {indexer.temperature}")
#     print()

#     response = indexer.generate(
#         system_prompt="You are a document indexing assistant.",
#         user_prompt=(
#             "Given these three topics:\n"
#             "- Credit cards\n"
#             "- Installment offers\n"
#             "- Cashback campaigns\n\n"
#             "Return a short explanation of how you would organize them."
#         ),
#     )

#     print("MODEL RESPONSE:")
#     print(response)

#     print()
#     print("=" * 60)
#     print("TEST 3 — Traverser")
#     print("=" * 60)

#     traverser = LLMClient(role="traverser")

#     print(f"Role: {traverser.role}")
#     print(f"Model: {traverser.model}")
#     print(f"Temperature: {traverser.temperature}")
#     print()

#     response = traverser.generate(
#         system_prompt="You are a tree traversal assistant.",
#         user_prompt=(
#             "Here is a simple tree:\n\n"
#             "Banking Products\n"
#             "├── Cards\n"
#             "│   ├── Credit Cards\n"
#             "│   └── Debit Cards\n"
#             "├── Offers\n"
#             "│   ├── Installments\n"
#             "│   └── Discounts\n"
#             "└── Campaigns\n"
#             "    ├── Cashback\n"
#             "    └── Travel\n\n"
#             "Question: Which branch should I inspect to find cashback information?"
#         ),
#     )

#     print("MODEL RESPONSE:")
#     print(response)


#     print()
#     print("=" * 60)
#     print("TEST 4 — JSON OUTPUT")
#     print("=" * 60)

#     indexer = LLMClient(role="indexer")

#     response = indexer.generate(
#         system_prompt=(
#             "You are a document indexing system. "
#             "Return ONLY valid JSON. "
#             "Do not include markdown or explanations."
#         ),
#         user_prompt=(
#             "Create a simple hierarchical index for these topics:\n"
#             "1. Credit Cards\n"
#             "2. Installment Offers\n"
#             "3. Cashback Campaigns\n\n"
#             "Use this structure:\n"
#             "{\n"
#             '  "title": "...",\n'
#             '  "children": []\n'
#             "}"
#         ),
#         response_format="json",
#     )

#     print("RAW MODEL RESPONSE:")
#     print(response)

#     print()
#     print("Attempting to parse JSON...")

#     import json

#     try:
#         parsed = json.loads(response)

#         print("JSON PARSED SUCCESSFULLY!")
#         print(parsed)

#     except json.JSONDecodeError as e:
#         print("JSON PARSING FAILED!")
#         print(e)


#     print()
#     print("=" * 60)
#     print("ALL TESTS COMPLETED")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()



from pathlib import Path; import tempfile; from indexing.pageindex_tree_builder import PageIndexTreeBuilder; path = Path(tempfile.gettempdir()) / 'pageindex-compat-check.md'; path.write_text('# Check\n\n## Node\ntext\n', encoding='utf-8'); PageIndexTreeBuilder(add_node_summary='no')._run_md_to_tree(path); print('Python 3.10 PageIndex compatibility: OK')