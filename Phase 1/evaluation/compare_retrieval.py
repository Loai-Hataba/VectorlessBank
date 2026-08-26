"""
evaluation/compare_retrieval.py

TWO RETRIEVAL TECHNIQUES, SAME QUESTIONS, SAME SCORING.

WHAT THIS MEASURES
------------------
Runs the identical question set from evaluation/testset.json through
both retrieval techniques built for flat/columnar sources:

    "tree"       -- TreeRetriever, PageIndex-built hierarchy + LLM
                    traversal (evaluation/eval_retrieval.py's technique)
    "dataframe"  -- DataFrameRetriever, flattened pandas table + an LLM
                    that proposes structured column filters (never code)

Both are scored with the exact same precision/recall/F1 function
against the exact same hand-labelled gold record IDs, imported
directly from eval_retrieval.py rather than reimplemented, so the two
numbers are guaranteed to mean the same thing.

WHY THIS EXISTS
---------------
The two techniques were built specifically to answer a design question
with evidence instead of intuition: does a tree built from PageIndex's
heading-based hierarchy actually help on data that has no real
hierarchy in the first place (a flat product-catalog spreadsheet), or
does a technique built for that shape of data directly (a table +
column filters) do better? Running both and reporting real numbers
side by side is what makes that a testable claim rather than an
assertion.

WHY DISAGREEMENTS ARE CALLED OUT, NOT JUST AGGREGATE SCORES
--------------------------------------------------------------------
Two aggregate F1 numbers tell you which technique wins on average, but
say nothing about WHY. Every question where the two techniques
returned a different set of record IDs is printed as its own block --
question, what each technique returned, and which one (if either)
actually matched the gold set -- so a specific, concrete example is
available immediately rather than needing to be hunted for separately
afterward.

USAGE
-----
    python evaluation/compare_retrieval.py
    python evaluation/compare_retrieval.py --category single_source_cards
    python evaluation/compare_retrieval.py --limit 10
    python evaluation/compare_retrieval.py --json out.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MAX_RESULTS_PER_SOURCE, DATAFRAME_SOURCES  # noqa: E402
from loaders.cards_loader import CardsLoader                          # noqa: E402
from loaders.offers_loader import OffersLoader                        # noqa: E402
from retrievers.tree_retriever import TreeRetriever                   # noqa: E402
from retrievers.dataframe_retriever import DataFrameRetriever         # noqa: E402
from indexing.dataframe_builder import DataFrameBuilder                # noqa: E402

# Reused, not reimplemented -- see eval_retrieval.py's own docstring
# for why prf() handles the empty-gold case the way it does.
from evaluation.eval_retrieval import prf, load_cases                 # noqa: E402

LOADERS = {
    "cards": CardsLoader,
    "offers": OffersLoader,
}


def build_tree_retrievers() -> dict:
    return {
        source: TreeRetriever(source=source, records=loader_cls().load())
        for source, loader_cls in LOADERS.items()
        if source in DATAFRAME_SOURCES
    }


def build_dataframe_retrievers() -> dict:
    """
    Builds the dataframe index fresh, in-memory, for this comparison
    run rather than requiring a separate offline build step first --
    keeps this script runnable on its own. A real deployment would
    build and persist these once via DataFrameBuilder.save(), the same
    way build_index.py does for trees.
    """
    builder = DataFrameBuilder()
    retrievers = {}

    for source, loader_cls in LOADERS.items():
        if source not in DATAFRAME_SOURCES:
            continue
        records = loader_cls().load()
        df = builder.build(source, records)
        retrievers[source] = DataFrameRetriever(
            source=source, df=df, records=records
        )

    return retrievers


def retrieve_all(retrievers: dict, question: str) -> list:
    all_records = []
    seen = set()
    for retriever in retrievers.values():
        for record in retriever.retrieve(question, top_k=MAX_RESULTS_PER_SOURCE):
            if record.id not in seen:
                seen.add(record.id)
                all_records.append(record)
    return all_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--case")
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    cases = load_cases(args)

    # Only questions relevant to the sources both techniques actually
    # cover -- comparing on a campaigns-only question would silently
    # score "dataframe" as failing every time, for a reason that has
    # nothing to do with which technique works better.
    cases = [
        c for c in cases
        if any(
            gid.startswith(("card_", "offer_"))
            for gid in c.get("gold_record_ids", [])
        )
        or not c.get("gold_record_ids")
    ]

    if not cases:
        print("No comparable cases matched (need cards/offers gold IDs).")
        raise SystemExit(1)

    print("Building tree retrievers...")
    tree_retrievers = build_tree_retrievers()

    print("Building dataframe retrievers...")
    dataframe_retrievers = build_dataframe_retrievers()

    print(f"\nComparing {len(cases)} case(s) across both techniques.\n")

    results = []
    disagreements = []
    started = time.time()

    for index, case in enumerate(cases, start=1):

        question = case.get("standalone_question") or case["question"]
        gold = set(case.get("gold_record_ids", []))

        t0 = time.time()
        tree_records = retrieve_all(tree_retrievers, question)
        tree_seconds = time.time() - t0

        t0 = time.time()
        df_records = retrieve_all(dataframe_retrievers, question)
        df_seconds = time.time() - t0

        tree_ids = {r.id for r in tree_records}
        df_ids = {r.id for r in df_records}

        tp, tr, tf = prf(tree_ids, gold)
        dp, dr, df_f1 = prf(df_ids, gold)

        row = {
            "id": case["id"],
            "category": case.get("category", ""),
            "question": question,
            "gold": sorted(gold),
            "tree": {
                "retrieved": sorted(tree_ids), "precision": round(tp, 3),
                "recall": round(tr, 3), "f1": round(tf, 3),
                "seconds": round(tree_seconds, 1),
            },
            "dataframe": {
                "retrieved": sorted(df_ids), "precision": round(dp, 3),
                "recall": round(dr, 3), "f1": round(df_f1, 3),
                "seconds": round(df_seconds, 1),
            },
        }
        results.append(row)

        if tree_ids != df_ids:
            disagreements.append(row)

        status = "SAME" if tree_ids == df_ids else "DIFF"
        print(
            f"[{index:>2}/{len(cases)}] {status} {case['id']:<14} "
            f"tree F1={tf:.2f} ({tree_seconds:.1f}s)  "
            f"dataframe F1={df_f1:.2f} ({df_seconds:.1f}s)"
        )

    # -----------------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------------

    def mean(technique, key):
        values = [r[technique][key] for r in results]
        return sum(values) / len(values) if values else 0.0

    print()
    print("=" * 66)
    print(f"{len(results)} cases in {time.time() - started:.0f}s")
    print("-" * 66)
    print(
        f"  tree        P={mean('tree','precision'):.3f}  "
        f"R={mean('tree','recall'):.3f}  F1={mean('tree','f1'):.3f}  "
        f"avg {mean('tree','seconds'):.1f}s/case"
    )
    print(
        f"  dataframe   P={mean('dataframe','precision'):.3f}  "
        f"R={mean('dataframe','recall'):.3f}  F1={mean('dataframe','f1'):.3f}  "
        f"avg {mean('dataframe','seconds'):.1f}s/case"
    )
    print("-" * 66)
    print(f"  agreed on identical record sets: {len(results) - len(disagreements)}/{len(results)}")
    print(f"  disagreed: {len(disagreements)}/{len(results)}")
    print("=" * 66)

    # -----------------------------------------------------------
    # Disagreements, spelled out -- the presentation material
    # -----------------------------------------------------------

    if disagreements:
        print("\nDISAGREEMENTS (concrete examples for the presentation)\n")

        for row in disagreements:
            gold = set(row["gold"])
            tree_hit = set(row["tree"]["retrieved"]) >= gold and bool(gold)
            df_hit = set(row["dataframe"]["retrieved"]) >= gold and bool(gold)

            verdict = (
                "tree matched gold, dataframe did not" if tree_hit and not df_hit else
                "dataframe matched gold, tree did not" if df_hit and not tree_hit else
                "both matched gold" if tree_hit and df_hit else
                "neither matched gold"
            )

            print(f"[{row['id']}] {row['question']}")
            print(f"  gold:      {row['gold']}")
            print(f"  tree:      {row['tree']['retrieved']}  (F1={row['tree']['f1']})")
            print(f"  dataframe: {row['dataframe']['retrieved']}  (F1={row['dataframe']['f1']})")
            print(f"  -> {verdict}")
            print()

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"results": results, "disagreements": disagreements},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()