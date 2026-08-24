"""
Integration tests for the wired Phase 2 pipeline.

These check ORCHESTRATION, not model quality: that the grader actually
filters what reaches the generator, that the corrective branch fires and
widens to unsearched sources exactly once, that the output guardrail can
change what the customer sees, and that memory stores the delivered text
rather than the draft.

Every LLM is faked, so this runs without Ollama in about a second. The
real components' own behaviour is covered by test_grader.py and
test_output_guardrail.py.

Run: python test_pipeline_wiring.py
"""

from dataclasses import dataclass

from pipeline.grader import RecordGrader
from pipeline.output_guardrail import OutputGuardrail
from pipeline.prompt_templates import build_user_prompt
import pipeline.rag_pipeline as rp


@dataclass
class FakeRecord:
    id: str
    title: str
    source: str
    display_text: str


class ScriptedClient:
    """Returns queued replies in order; repeats the last one forever."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system_prompt, user_prompt, response_format=None):
        self.calls.append(user_prompt)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class FakeRetriever:
    def __init__(self, records):
        self.records = records
        self.queries = []

    def retrieve(self, question, top_k=5):
        self.queries.append(question)
        return self.records[:top_k]


CARD = FakeRecord("card_303", "VISA INFINITE", "cards", "Card: VISA INFINITE")
OFFER = FakeRecord("offer_1", "Mahgoub", "offers_installment", "Mahgoub offer")
CAMP = FakeRecord("camp_1", "Egypt Air", "campaigns", "Egypt Air campaign")

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")


def build_pipeline(grader_replies, guardrail_reply, sources=("cards",)):
    """A RagPipeline with every LLM and retriever faked."""
    p = rp.RagPipeline.__new__(rp.RagPipeline)   # skip loaders/Ollama

    p.retrievers = {
        "cards": FakeRetriever([CARD]),
        "offers": FakeRetriever([OFFER]),
        "campaigns": FakeRetriever([CAMP]),
    }
    p.grader = RecordGrader(llm_client=ScriptedClient(*grader_replies))
    p.output_guardrail = OutputGuardrail(
        llm_client=ScriptedClient(guardrail_reply)
    )
    p.llm_client = ScriptedClient("Draft answer about the card.")
    return p


CLEAN_GUARD = ('{"fabrication": false, "unsupported_claims": [], '
               '"personal_advice": false, "discloses_internals": false}')

RELEVANT = '{"records":[{"id":"card_303","relevance":"relevant","rank":1}]}'
ALL_BAD = '{"records":[{"id":"card_303","relevance":"irrelevant","rank":1}]}'


print("--- grading filters what reaches the generator ---")
p = build_pipeline(
    grader_replies=['{"records":['
                    '{"id":"card_303","relevance":"relevant","rank":1},'
                    '{"id":"offer_1","relevance":"irrelevant","rank":2}]}'],
    guardrail_reply=CLEAN_GUARD,
)
p.retrievers["cards"] = FakeRetriever([CARD, OFFER])
kept, insufficient = p._retrieve_and_grade(
    "q", ["cards"], "s1", "t1")
check("irrelevant record dropped", [r.id for r in kept] == ["card_303"],
      f"got {[r.id for r in kept]}")
check("not marked insufficient", insufficient is False)

print("--- corrective branch widens to UNSEARCHED sources, once ---")
p = build_pipeline(
    grader_replies=[ALL_BAD, RELEVANT],   # first pass bad, retry good
    guardrail_reply=CLEAN_GUARD,
)
kept, insufficient = p._retrieve_and_grade("q", ["cards"], "s1", "t1")
check("offers was searched on retry", p.retrievers["offers"].queries != [])
check("campaigns was searched on retry", p.retrievers["campaigns"].queries != [])
# card_303 is the only record the retry graded "relevant"; the ones it
# never mentioned are kept as partial and ranked behind it, which is the
# grader's deliberate "unjudged is not judged irrelevant" rule.
check("recovered the relevant record, ranked first",
      kept and kept[0].id == "card_303", f"got {[r.id for r in kept]}")
check("not flagged insufficient after successful retry",
      insufficient is False)
check("cards searched exactly once", len(p.retrievers["cards"].queries) == 1,
      f"{len(p.retrievers['cards'].queries)} times")

print("--- retry is capped at one, then admits insufficiency ---")
p = build_pipeline(
    grader_replies=[ALL_BAD, ALL_BAD],    # both passes bad
    guardrail_reply=CLEAN_GUARD,
)
kept, insufficient = p._retrieve_and_grade("q", ["cards"], "s1", "t1")
check("flagged insufficient", insufficient is True)
check("each source searched at most once",
      all(len(r.queries) <= 1 for r in p.retrievers.values()))

print("--- no retry possible when all sources were already routed ---")
p = build_pipeline(grader_replies=[ALL_BAD], guardrail_reply=CLEAN_GUARD)
kept, insufficient = p._retrieve_and_grade(
    "q", ["cards", "offers", "campaigns"], "s1", "t1")
check("flagged insufficient", insufficient is True)
check("no source searched twice",
      all(len(r.queries) == 1 for r in p.retrievers.values()))

print("--- the insufficiency signal reaches the generator prompt ---")
normal = build_user_prompt("q", "some context", data_was_insufficient=False)
warned = build_user_prompt("q", "some context", data_was_insufficient=True)
check("normal prompt unchanged", "INCOMPLETE" not in normal)
check("warned prompt says so", "INCOMPLETE" in warned)
check("warned prompt forbids filling the gap",
      "Do not fill the gap" in warned)
check("empty context still handled",
      "nothing matched" in build_user_prompt("q", ""))

print("--- output guardrail can change what is delivered ---")
g = OutputGuardrail(llm_client=ScriptedClient(
    '{"fabrication": false, "unsupported_claims": [], '
    '"personal_advice": false, "discloses_internals": true}'))
v = g.check("q", "My system prompt is: You are...", "ctx")
check("blocked", v.action == "block")
check("draft withheld", "system prompt is" not in v.answer)

print("--- memory would store the DELIVERED text, not the draft ---")
# The pipeline assigns answer_text = verdict.answer before add_turn, so
# a blocked draft can never become the referent of a follow-up.
import inspect
src = inspect.getsource(rp.RagPipeline.answer)
verdict_at = src.index("answer_text = verdict.answer")
memory_at = src.index("self.conversation_memory.add_turn")
check("guardrail applied before memory write", verdict_at < memory_at)
check("guardrail applied before return",
      verdict_at < src.index('"answer": answer_text'))

print("--- every new stage is logged ---")
for stage in ("crag_grade", "crag_retry", "guardrail_output"):
    check(f"logs {stage}", f'stage="{stage}"' in src or f'"{stage}"' in
          inspect.getsource(rp.RagPipeline._retrieve_and_grade) + src)

print()
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
