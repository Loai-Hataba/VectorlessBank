"""
Unit tests for the Tier 2 harness logic that needs neither Ollama nor
ragas: context reconstruction, scorability rules, NaN-aware averaging
and the cached-run reader.

Both eval scripts keep their ragas imports inside functions precisely so
this file can import them without the evaluation environment installed.

Run: python evaluation/test_eval_tier2.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # Phase 1/  -- config, pipeline
sys.path.insert(0, str(HERE))          # evaluation/

import eval_generate as gen            # noqa: E402
import eval_answers as ans             # noqa: E402

from config.settings import (          # noqa: E402
    MAX_CONTEXT_TOTAL_CHARS,
    MIN_RECORD_CHARS,
)

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")


class FakeRecord:
    def __init__(self, text, rid="r1"):
        self.display_text = text
        self.id = rid


# ---------------------------------------------------------------------
# build_contexts -- must mirror what the generator was shown
# ---------------------------------------------------------------------

print("--- contexts match the generator's own budget ---")

check("empty in, empty out", gen.build_contexts([]) == [])

small = [FakeRecord("a" * 100), FakeRecord("b" * 100)]
check("short records are not touched",
      [len(c) for c in gen.build_contexts(small)] == [100, 100])

one_huge = gen.build_contexts([FakeRecord("x" * 200_000)])
check("a single record gets the whole budget",
      len(one_huge[0]) > MAX_CONTEXT_TOTAL_CHARS - 1,
      f"got {len(one_huge[0])}")

six_huge = gen.build_contexts([FakeRecord("x" * 60_000) for _ in range(6)])
check("six records split the budget",
      all(abs(len(c) - MAX_CONTEXT_TOTAL_CHARS // 6) < 100 for c in six_huge),
      f"got {[len(c) for c in six_huge]}")

check("a truncated record says so",
      "truncated at" in six_huge[0])

# The floor exists to stop a large result set shaving records to
# uselessness; with RERANK_TOP_K at 6 it never binds today, but the
# harness must not silently disagree with context_builder if it ever does.
many = gen.build_contexts([FakeRecord("x" * 60_000) for _ in range(30)])
check("the per-record floor is respected",
      all(len(c) >= MIN_RECORD_CHARS for c in many),
      f"min was {min(len(c) for c in many)}")


# ---------------------------------------------------------------------
# scorability
# ---------------------------------------------------------------------

print("--- recognising a guardrail block ---")

from pipeline.input_guardrail import CANNED_RESPONSE_BY_VERDICT   # noqa: E402
from config.settings import BLOCKED_DISCLOSURE_REPLY              # noqa: E402

check("an ordinary answer is not a block",
      gen.detect_block("The annual fee is EGP 8000.") is None)

for verdict, text in CANNED_RESPONSE_BY_VERDICT.items():
    check(f"input guardrail '{verdict}' recognised",
          gen.detect_block(text) == verdict, gen.detect_block(text))

check("output guardrail disclosure block recognised",
      gen.detect_block(BLOCKED_DISCLOSURE_REPLY)
      == "output_guardrail_disclosure")

check("surrounding whitespace does not hide a block",
      gen.detect_block("  " + CANNED_RESPONSE_BY_VERDICT["off_topic"] + "\n")
      == "off_topic")

check("an annotated answer is not mistaken for a block",
      gen.detect_block(BLOCKED_DISCLOSURE_REPLY + "\n\nextra text") is None)


print("--- guardrail outcome against the testset's labels ---")

check("adversarial case blocked as labelled = correct",
      gen.guardrail_outcome(
          {"expected_guardrail": "injection_attempt"}, "injection_attempt")
      == "correct")

check("adversarial case blocked as the wrong verdict is reported",
      "expected injection_attempt" in gen.guardrail_outcome(
          {"expected_guardrail": "injection_attempt"}, "off_topic"))

check("adversarial case not blocked at all is reported",
      "expected injection_attempt" in gen.guardrail_outcome(
          {"expected_guardrail": "injection_attempt"}, None))

check("a legitimate question that got blocked is a false positive",
      gen.guardrail_outcome({}, "off_topic") == "FALSE POSITIVE")

check("a legitimate question that passed is correct",
      gen.guardrail_outcome({}, None) == "not blocked (correct)")


print("--- scorability rules ---")

general = {"needs_retrieval": False}
normal = {"needs_retrieval": True}

ok, why = gen.scorability(normal, [FakeRecord("z")], "a real answer", None)
check("context + answer is scorable", ok is True and why == "")

ok, why = gen.scorability(general, [], "APR means annual percentage rate", None)
check("general knowledge is not scorable", ok is False)
check("and says why", "no retrieval expected" in why, why)

ok, why = gen.scorability(normal, [], "sorry, nothing found", None)
check("a genuine retrieval miss is distinguished from a router skip",
      ok is False and "returned nothing" in why, why)

ok, why = gen.scorability(normal, [FakeRecord("z")], "   ", None)
check("an empty answer is not scorable", ok is False and "empty answer" in why)

ok, why = gen.scorability(normal, [], "canned", "off_topic")
check("a block is reported as a block, not as a retrieval miss",
      ok is False and "blocked by guardrail (off_topic)" in why, why)


# ---------------------------------------------------------------------
# mean_ignoring_nan -- a judge failure is not a score of zero
# ---------------------------------------------------------------------

print("--- NaN-aware averaging ---")

average, failures = ans.mean_ignoring_nan([1.0, 0.0])
check("plain mean", average == 0.5 and failures == 0)

average, failures = ans.mean_ignoring_nan([1.0, float("nan"), 1.0])
check("a judge failure is excluded, not counted as zero",
      average == 1.0, f"got {average}")
check("and is reported", failures == 1)

average, failures = ans.mean_ignoring_nan([float("nan"), float("nan")])
check("all failures give NaN, not 0.0", math.isnan(average))
check("and counts them all", failures == 2)

check("no metric needs a reference by default",
      ans.needs_reference(list(ans.REFERENCE_FREE)) is False)
check("context_recall does need one",
      ans.needs_reference(["context_recall"]) is True)


# ---------------------------------------------------------------------
# load_samples -- the handoff file
# ---------------------------------------------------------------------

print("--- reading a cached run ---")


class Args:
    category = None
    case = None
    limit = None

    def __init__(self, infile):
        self.infile = infile


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "generated_answers.json"

    payload = {
        "generated_at": 1.0,
        "models": {"generator": "llama3.1:8b"},
        "settings": {"RERANK_TOP_K": 6},
        "samples": [
            {"id": "a1", "category": "single_source_cards", "scorable": True,
             "question": "q", "answer": "a", "contexts": ["c"]},
            {"id": "b1", "category": "general_knowledge", "scorable": False,
             "not_scorable_because": "no retrieval expected",
             "question": "q", "answer": "a", "contexts": []},
            {"id": "c1", "category": "single_source_cards", "scorable": True,
             "question": "q", "answer": "a", "contexts": ["c"]},
        ],
    }

    path.write_text(json.dumps(payload), encoding="utf-8")

    scorable, skipped, meta = ans.load_samples(Args(str(path)))
    check("scorable and skipped are separated",
          [s["id"] for s in scorable] == ["a1", "c1"]
          and [s["id"] for s in skipped] == ["b1"])
    check("run metadata is carried through",
          meta["settings"]["RERANK_TOP_K"] == 6)

    args = Args(str(path))
    args.category = "single_source_cards"
    scorable, skipped, _ = ans.load_samples(args)
    check("category filter applies", [s["id"] for s in scorable] == ["a1", "c1"])

    args = Args(str(path))
    args.limit = 1
    scorable, _, _ = ans.load_samples(args)
    check("limit applies to scorable only", [s["id"] for s in scorable] == ["a1"])

    missing = Args(str(Path(tmp) / "nope.json"))
    try:
        ans.load_samples(missing)
        check("a missing run file is a clear error", False, "no SystemExit")
    except SystemExit as exc:
        check("a missing run file is a clear error",
              "eval_generate.py" in str(exc), str(exc)[:60])

print()
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
