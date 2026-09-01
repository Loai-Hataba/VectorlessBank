"""
evaluation/eval_retrieval.py

TIER 1 EVALUATION -- retrieval correctness, deterministic.

WHAT THIS MEASURES
------------------
Whether retrieval returns the records a question actually needs, scored
as precision / recall / F1 of record IDs against a hand-labelled gold
set. No LLM judge is involved anywhere in this file.

WHY THIS IS POSSIBLE HERE
-------------------------
Records have stable, human-readable IDs (card_303,
offer_installment_4_Carrefour), so "the right answer" can be written
down by hand. A vector-database RAG system cannot do this cheaply --
you would need embeddings just to define which chunk was correct. It is
a direct benefit of the vectorless, structured-record architecture and
worth using rather than reaching for an LLM judge on a question that is
objectively checkable.

WHY IT DELIBERATELY DOES NOT GENERATE ANSWERS
---------------------------------------------
A full question takes roughly 45 seconds end to end, most of it
generation, so a 39-case suite with answers is around half an hour
before anything is judged. Retrieval-only keeps the everyday feedback
loop short enough to actually be run.

Answer quality is therefore NOT measured here, and is currently not
measured anywhere -- the RAGAS-based Tier 2 this line used to point at
was evaluated and dropped, because it costs ~60 langchain packages and
a local judge slow enough that nobody would run it. The in-pipeline
output guardrail checks grounding on every real question instead, which
is continuous rather than occasional. If a batch metric is wanted
later, evaluation/eval_input_guardrail.py is the pattern to copy:
labelled cases, no new dependencies, false positives and false
negatives reported separately.

USAGE
-----
    python evaluation/eval_retrieval.py                      # everything
    python evaluation/eval_retrieval.py --limit 6            # first 6 cases
    python evaluation/eval_retrieval.py --category single_source_cards
    python evaluation/eval_retrieval.py --case offers_001
    python evaluation/eval_retrieval.py --grade              # + CRAG grader
    python evaluation/eval_retrieval.py --json out.json      # machine readable

--grade runs the Step 1 grader over what retrieval returned, so the
same suite reports whether grading improves precision and whether the
corrective branch fires where it should.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as `python evaluation/eval_retrieval.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MAX_RESULTS_PER_SOURCE          # noqa: E402
from loaders.cards_loader import CardsLoader                # noqa: E402
from loaders.offers_loader import OffersLoader              # noqa: E402
from loaders.campaigns_loader import CampaignsLoader        # noqa: E402
from retrievers.tree_retriever import TreeRetriever         # noqa: E402

TESTSET_PATH = Path(__file__).resolve().parent / "testset.json"


def prf(retrieved: set, gold: set) -> tuple[float, float, float]:
    """
    Precision, recall, F1 for one case.

    The empty-gold case is not a division-by-zero nuisance, it is a real
    test: "unanswerable" cases assert that nothing relevant exists.
    Retrieving nothing there is a perfect score; retrieving something is
    a precision failure with recall undefined, reported as 1.0 so it
    does not drag the recall average down for a question that had
    nothing to recall.
    """
    if not gold:
        return (1.0 if not retrieved else 0.0), 1.0, (1.0 if not retrieved else 0.0)

    if not retrieved:
        return 0.0, 0.0, 0.0

    hits = len(retrieved & gold)
    precision = hits / len(retrieved)
    recall = hits / len(gold)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    return precision, recall, 2 * precision * recall / (precision + recall)


def load_cases(args) -> list:
    data = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]

    # Cases with needs_retrieval=false test the router, not retrieval,
    # and scoring them here would measure the wrong component.
    cases = [c for c in cases if c.get("needs_retrieval", True)]

    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]

    if args.case:
        cases = [c for c in cases if c.get("id") == args.case]

    if args.limit:
        cases = cases[: args.limit]

    return cases


def build_retrievers() -> dict:
    print("Loading records and tree indexes...")
    return {
        "cards": TreeRetriever(source="cards", records=CardsLoader().load()),
        "offers": TreeRetriever(source="offers", records=OffersLoader().load()),
        "campaigns": TreeRetriever(
            source="campaigns", records=CampaignsLoader().load()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--case")
    parser.add_argument("--grade", action="store_true",
                        help="also run the CRAG grader over the results")
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    cases = load_cases(args)

    if not cases:
        print("No cases matched.")
        raise SystemExit(1)

    retrievers = build_retrievers()

    grader = None
    if args.grade:
        from pipeline.grader import RecordGrader
        grader = RecordGrader()

    print(f"Scoring {len(cases)} case(s). "
          f"Grader: {'on' if grader else 'off'}\n")

    results = []
    started = time.time()

    for index, case in enumerate(cases, start=1):

        # Follow-ups are scored on the resolved question. Until the
        # contextualizer exists, using the raw text would measure a
        # component nobody has built yet.
        question = case.get("standalone_question") or case["question"]
        gold = set(case.get("gold_record_ids", []))

        # No router yet, so fan out to every source, exactly as the
        # pipeline does today. When the router lands, swap this for the
        # sources it selects -- the numbers here are the before.
        retrieved_records = []
        seen = set()

        t0 = time.time()

        for source, retriever in retrievers.items():
            for record in retriever.retrieve(
                question, top_k=MAX_RESULTS_PER_SOURCE
            ):
                if record.id not in seen:
                    seen.add(record.id)
                    retrieved_records.append(record)

        retrieval_seconds = time.time() - t0

        retrieved = {r.id for r in retrieved_records}
        precision, recall, f1 = prf(retrieved, gold)

        row = {
            "id": case["id"],
            "category": case["category"],
            "question": question,
            "gold": sorted(gold),
            "retrieved": sorted(retrieved),
            "missed": sorted(gold - retrieved),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "seconds": round(retrieval_seconds, 1),
        }

        if grader is not None:
            g0 = time.time()
            graded = grader.grade(question, retrieved_records)
            kept = {r.id for r in graded.kept}
            gp, gr, gf = prf(kept, gold)

            row["graded"] = {
                "kept": sorted(kept),
                "verdict": graded.verdict,
                "needs_retry": graded.needs_retry,
                "precision": round(gp, 3),
                "recall": round(gr, 3),
                "f1": round(gf, 3),
                "seconds": round(time.time() - g0, 1),
                "warnings": graded.warnings,
            }

            # An unanswerable case is only handled correctly if the
            # grader ALSO asks for a retry -- silently keeping junk is
            # the failure this component exists to prevent.
            if not gold:
                row["graded"]["correctly_flagged"] = graded.needs_retry

        results.append(row)

        status = "OK  " if recall == 1.0 else "MISS"
        line = (f"[{index:>2}/{len(cases)}] {status} {case['id']:<12} "
                f"P={precision:.2f} R={recall:.2f} "
                f"({retrieval_seconds:.0f}s)")

        if grader is not None:
            g = row["graded"]
            line += (f"  |  graded P={g['precision']:.2f} "
                     f"R={g['recall']:.2f} {g['verdict']}"
                     f"{' RETRY' if g['needs_retry'] else ''}")

        print(line)

        if row["missed"]:
            print(f"           missed: {', '.join(row['missed'])}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    def mean(key, source=None):
        values = [
            (r[source][key] if source else r[key])
            for r in results
            if (source is None or source in r)
        ]
        return sum(values) / len(values) if values else 0.0

    print()
    print("=" * 62)
    print(f"{len(results)} cases in {time.time() - started:.0f}s")
    print("-" * 62)
    print(f"  retrieval   P={mean('precision'):.3f}  "
          f"R={mean('recall'):.3f}  F1={mean('f1'):.3f}")

    if grader is not None:
        print(f"  + grading   P={mean('precision', 'graded'):.3f}  "
              f"R={mean('recall', 'graded'):.3f}  "
              f"F1={mean('f1', 'graded'):.3f}")
        retries = sum(1 for r in results if r["graded"]["needs_retry"])
        print(f"  retries requested: {retries}/{len(results)}")

    perfect = sum(1 for r in results if r["recall"] == 1.0)
    print(f"  full recall: {perfect}/{len(results)} cases")

    print("-" * 62)
    by_category: dict[str, list] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    for category, rows in sorted(by_category.items()):
        rec = sum(r["recall"] for r in rows) / len(rows)
        print(f"  {category:<26} R={rec:.3f}  ({len(rows)} cases)")
    print("=" * 62)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
