"""
evaluation/eval_generate.py

TIER 2, PHASE A -- produce the answers that Tier 2 will score.

WHAT THIS FILE DOES
-------------------
Runs the full RagPipeline over the testset and writes every answer, with
the context it was built from, to a JSON file. It scores nothing. That
is Phase B's job (evaluation/eval_answers.py).

    testset.json
        |
        v
    RagPipeline.answer()        <- this file, slow, needs Ollama
        |
        v
    generated_answers.json      <- the handoff
        |
        v
    RAGAS metrics               <- eval_answers.py, needs ragas

WHY GENERATION AND SCORING ARE SEPARATE PROGRAMS
------------------------------------------------
Two reasons, and both are about not paying for the same work twice.

1. TIME. A full question takes roughly 165-205 seconds end to end,
   nearly all of it generation. The 39-case testset is therefore about
   two hours of wall clock. If scoring lived in the same run, every
   tweak to a metric, a threshold or an output format would cost
   another two hours. With the JSON file in between, scoring a cached
   run takes minutes and can be repeated freely.

2. DEPENDENCIES. RAGAS pulls in a large langchain/datasets stack. The
   Phase 2 blueprint is explicit that the project must not quietly
   acquire heavy retrieval infrastructure, so that stack must not land
   in the environment the app itself runs in. Splitting the phases lets
   Phase A run on the project's own five dependencies and Phase B run
   in a separate environment that the application never imports. The
   JSON file is the only thing they share.

   This mirrors the split that already exists between the two halves of
   the project: indexing and answering also communicate only through a
   file on disk.

WHY IT RECORDS ONE CONTEXT STRING PER RECORD
--------------------------------------------
RAGAS wants `retrieved_contexts` as a list, and judges each entry
separately -- that is what makes context precision meaningful. But the
entries must be exactly what the generator was shown, truncated by the
same shared budget, or Tier 2 recreates the project's own guardrail bug
in a new place: a verifier shown less than the writer saw reports the
writer's legitimate content as unsupported. See the note on
GUARDRAIL_MAX_CONTEXT_CHARS in config/settings.py.

That is why this file imports context_builder's private helpers rather
than re-deriving the budget. The coupling is deliberate: if the budget
logic changes, this must change with it.

WHY EVERY CASE IS GENERATED, INCLUDING THE ONES TIER 2 CANNOT SCORE
--------------------------------------------------------------------
Guardrail cases produce a canned refusal and general-knowledge cases
produce an answer with no retrieved context. Neither can be scored for
faithfulness -- there is nothing to be faithful to. They are still run
and still recorded, with a `scorable` flag and a reason, because "did
the guardrail actually fire" and "did the router actually skip
retrieval" are worth seeing in the same artifact. Phase B decides what
to score; this file decides nothing.

USAGE
-----
    python evaluation/eval_generate.py                       # all 39 cases
    python evaluation/eval_generate.py --limit 5
    python evaluation/eval_generate.py --category follow_up
    python evaluation/eval_generate.py --case cards_001
    python evaluation/eval_generate.py --resume              # continue a killed run
    python evaluation/eval_generate.py --out somewhere.json

--resume matters more than it looks. A two-hour run that dies at case 30
should not start over, so completed cases are read back from the output
file and skipped.

INPUTS  : evaluation/testset.json
OUTPUTS : evaluation/generated_answers.json

REQUIRES: Ollama running, all three tree indexes built.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as `python evaluation/eval_generate.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (                              # noqa: E402
    MAX_CONTEXT_TOTAL_CHARS,
    OLLAMA_GENERATOR_MODEL,
    OLLAMA_GRADER_MODEL,
    OLLAMA_ROUTER_MODEL,
    RERANK_TOP_K,
    MAX_RESULTS_PER_SOURCE,
    CRAG_MIN_RELEVANT,
)
from config.settings import (                              # noqa: E402
    BLOCKED_DISCLOSURE_REPLY,
    BLOCKED_FABRICATION_REPLY,
)
from pipeline.context_builder import (                     # noqa: E402
    _budget_per_record,
    _truncate,
)
from pipeline.input_guardrail import (                     # noqa: E402
    CANNED_RESPONSE_BY_VERDICT,
)
from pipeline.rag_pipeline import RagPipeline              # noqa: E402

# A blocked turn is recognised by its reply being one of the fixed
# strings the guardrails hand back.
#
# WHY NOT JUST READ A FLAG OFF THE PIPELINE
# -----------------------------------------
# RagPipeline.answer() returns the answer, the records and the context,
# and says nothing about whether a guardrail intervened. Adding a flag
# would mean changing the pipeline to suit its test harness, which is
# the wrong direction of dependency. These replies are module constants,
# so matching them exactly is reliable without touching the pipeline.
BLOCK_REPLIES = {
    **{text: verdict for verdict, text in CANNED_RESPONSE_BY_VERDICT.items()},
    BLOCKED_DISCLOSURE_REPLY: "output_guardrail_disclosure",
    BLOCKED_FABRICATION_REPLY: "output_guardrail_fabrication",
}


def detect_block(answer: str) -> str | None:
    """
    Which guardrail verdict produced this reply, if any.

    Distinguishing "a guardrail stopped this" from "retrieval found
    nothing" is the difference between a safety component misfiring and
    a retrieval component missing -- two very different bugs that
    otherwise present identically as an answer with no context.
    """
    return BLOCK_REPLIES.get((answer or "").strip())

TESTSET_PATH = Path(__file__).resolve().parent / "testset.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "generated_answers.json"


def load_cases(args) -> list:
    """
    Every case, filtered only by the command-line flags.

    Unlike eval_retrieval.py this does NOT drop needs_retrieval=false
    cases. Tier 1 scores retrieved IDs, so a case that should retrieve
    nothing would measure the wrong component there. Tier 2 scores the
    ANSWER, and a general-knowledge answer is a real answer worth
    recording -- it simply cannot be scored for faithfulness later.
    """
    data = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]

    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]

    if args.case:
        cases = [c for c in cases if c.get("id") == args.case]

    if args.limit:
        cases = cases[: args.limit]

    return cases


def seed_history(pipeline: RagPipeline, session_id: str, history: list) -> None:
    """
    Replay a follow-up case's prior turns into conversation memory.

    Follow-up cases carry their own history rather than depending on the
    case before them, so each one is reproducible in isolation and can
    be run with --case. The turns are written straight into memory
    instead of being asked as real questions: re-generating them would
    triple the runtime of those cases and, worse, make the case depend
    on what the model happens to answer today rather than on the fixed
    history the case was labelled against.

    No summarizer call fires here. The longest history in the testset is
    two turns and MEMORY_RAW_TURNS_KEPT is three, so nothing is evicted.
    """
    for turn in history:
        pipeline.conversation_memory.add_turn(
            session_id=session_id,
            turn_id=pipeline.conversation_memory.next_turn_id(session_id),
            user_message=turn["user"],
            assistant_message=turn["assistant"],
        )


def build_contexts(records: list) -> list:
    """
    One string per record, truncated exactly as the generator saw it.

    See the module docstring: showing the scorer more or less than the
    generator was shown produces a verdict about the wrong text.
    """
    if not records:
        return []

    per_record = _budget_per_record(len(records), MAX_CONTEXT_TOTAL_CHARS)

    return [_truncate(r.display_text, per_record) for r in records]


def scorability(
    case: dict,
    records: list,
    answer: str,
    blocked: str | None,
) -> tuple[bool, str]:
    """
    Whether Phase B can meaningfully score this sample, and why not.

    Both reference-free RAGAS metrics need retrieved contexts. With no
    context there is nothing for a claim to be grounded in, so a
    perfectly correct general-knowledge answer would score zero
    faithfulness -- a number that would say something false about the
    system. Marking it unscorable is the honest outcome, not a gap.

    The three no-context cases are reported separately rather than as one
    bucket, because they mean completely different things:

        blocked     a guardrail fired -- correct on a guard_* case,
                    a false positive on any other
        not routed  the router decided no lookup was needed
        missed      retrieval ran and came back empty
    """
    if blocked:
        return False, f"blocked by guardrail ({blocked})"

    if not records:
        if not case.get("needs_retrieval", True):
            return False, "no retrieval expected (general knowledge case)"
        return False, "retrieval returned nothing"

    if not answer.strip():
        return False, "empty answer"

    return True, ""


def guardrail_outcome(case: dict, blocked: str | None) -> str:
    """
    Compare what the guardrail did against what the case expected.

    The testset already labels `expected_guardrail` on adversarial
    cases, so the blueprint's "guardrail pass/fail rate on the
    adversarial subset" falls out of a run that was happening anyway --
    no extra LLM calls, no second harness.
    """
    expected = case.get("expected_guardrail")

    if expected:
        return "correct" if blocked == expected else f"expected {expected}"

    # No label means the case is legitimate traffic, so any block is a
    # false positive -- the failure mode that makes a guardrail worse
    # than no guardrail.
    if blocked:
        return "FALSE POSITIVE"

    return "not blocked (correct)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--case")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cases already present in the output file",
    )
    args = parser.parse_args()

    cases = load_cases(args)

    if not cases:
        print("No cases matched.")
        raise SystemExit(1)

    out_path = Path(args.out)

    done: dict[str, dict] = {}

    if args.resume and out_path.exists():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        done = {
            s["id"]: s
            for s in previous.get("samples", [])
            if not s.get("error")
        }
        print(f"Resuming: {len(done)} case(s) already generated.")

    pending = [c for c in cases if c["id"] not in done]

    print("Loading records and tree indexes...")
    pipeline = RagPipeline()

    print(
        f"Generating {len(pending)} case(s). "
        f"At ~3 minutes each this is roughly "
        f"{len(pending) * 3} minutes.\n"
    )

    samples = list(done.values())
    started = time.time()

    for index, case in enumerate(pending, start=1):

        # A fresh session per case keeps memory from leaking between
        # unrelated questions, which would quietly change what the
        # contextualizer does to the next one.
        session_id = f"eval_{case['id']}"

        if case.get("history"):
            seed_history(pipeline, session_id, case["history"])

        t0 = time.time()
        error = None

        try:
            result = pipeline.answer(case["question"], session_id=session_id)
            answer = result["answer"]
            records = result["retrieved_records"]

        except Exception as exc:                       # noqa: BLE001
            # One bad case must not cost the whole two-hour run. It is
            # recorded with its error and skipped by Phase B.
            answer, records = "", []
            error = f"{type(exc).__name__}: {exc}"

        seconds = time.time() - t0

        blocked = detect_block(answer)
        can_score, reason = scorability(case, records, answer, blocked)
        guardrail = guardrail_outcome(case, blocked)

        samples.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "standalone_question": case.get("standalone_question"),
                "history": case.get("history", []),
                "needs_retrieval": case.get("needs_retrieval", True),
                "gold_record_ids": case.get("gold_record_ids", []),
                "expect_refusal": case.get("expect_refusal", False),
                "expected_guardrail": case.get("expected_guardrail"),
                "answer": answer,
                "contexts": build_contexts(records),
                "retrieved_ids": [r.id for r in records],
                "blocked_by": blocked,
                "guardrail_outcome": guardrail,
                "scorable": can_score,
                "not_scorable_because": reason,
                "seconds": round(seconds, 1),
                "error": error,
            }
        )

        status = "ERR " if error else ("OK  " if can_score else "SKIP")

        print(
            f"[{index:>2}/{len(pending)}] {status} {case['id']:<12} "
            f"{len(records)} record(s)  {seconds:.0f}s"
            + (f"  -- {reason}" if reason else "")
            + (f"  [{guardrail}]" if guardrail == "FALSE POSITIVE" else "")
            + (f"  {error}" if error else "")
        )

        # Written after every case, not at the end, so a killed run
        # loses at most the case in flight.
        write_output(out_path, samples)

    elapsed = time.time() - started

    false_positives = [
        s for s in samples if s.get("guardrail_outcome") == "FALSE POSITIVE"
    ]

    print()
    print("=" * 62)
    print(f"{len(pending)} generated in {elapsed / 60:.0f} min")
    print(f"  scorable by Tier 2 : {sum(1 for s in samples if s['scorable'])}")
    print(f"  not scorable       : {sum(1 for s in samples if not s['scorable'])}")
    print(f"  errored            : {sum(1 for s in samples if s['error'])}")
    print(f"  wrote              : {out_path}")

    # Surfaced here rather than left for Phase B, because a guardrail
    # that blocks legitimate questions makes every downstream number
    # meaningless -- those turns never reached retrieval or generation
    # at all, so there is nothing for Tier 2 to score and nothing wrong
    # with the components it would have scored.
    if false_positives:
        print("-" * 62)
        print(f"  GUARDRAIL FALSE POSITIVES: {len(false_positives)}")
        for sample in false_positives:
            print(f"    {sample['id']:<12} blocked as '{sample['blocked_by']}'"
                  f"  -- {sample['question'][:44]}")

    print("=" * 62)
    print()
    print("Next: score it with the ragas environment,")
    print("  python evaluation/eval_answers.py")


def write_output(path: Path, samples: list) -> None:
    """
    Record the settings alongside the answers.

    A Tier 2 number is only interpretable next to the configuration that
    produced it -- a faithfulness score from a run with RERANK_TOP_K=6
    says nothing about a run with 12. Storing them together means an old
    result file can still be read a month later.
    """
    payload = {
        "generated_at": time.time(),
        "models": {
            "generator": OLLAMA_GENERATOR_MODEL,
            "grader": OLLAMA_GRADER_MODEL,
            "router": OLLAMA_ROUTER_MODEL,
        },
        "settings": {
            "MAX_RESULTS_PER_SOURCE": MAX_RESULTS_PER_SOURCE,
            "RERANK_TOP_K": RERANK_TOP_K,
            "CRAG_MIN_RELEVANT": CRAG_MIN_RELEVANT,
            "MAX_CONTEXT_TOTAL_CHARS": MAX_CONTEXT_TOTAL_CHARS,
        },
        "samples": sorted(samples, key=lambda s: s["id"]),
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
