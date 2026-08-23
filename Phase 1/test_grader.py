"""Unit tests for RecordGrader's validation. No LLM: a fake client
returns the malformed replies a small model actually produces."""
from dataclasses import dataclass
from pipeline.grader import (
    RecordGrader, RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT, UNRANKED,
)


@dataclass
class FakeRecord:
    id: str
    title: str
    source: str
    display_text: str


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_format=None):
        self.calls += 1
        return self.reply


RECS = [
    FakeRecord("card_303", "VISA INFINITE", "cards", "a card " * 400),
    FakeRecord("card_304", "VISA PRIVATE INFINITE", "cards", "another card"),
    FakeRecord("offer_1", "Carrefour", "offers_installment", "an offer"),
]

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")


def grade_with(reply, records=RECS):
    g = RecordGrader(llm_client=FakeClient(reply))
    return g.grade("What is the annual fee of the VISA INFINITE?", records)


print("--- happy path: filter + rank + cap ---")
r = grade_with("""{"records":[
 {"id":"offer_1","relevance":"irrelevant","rank":3},
 {"id":"card_303","relevance":"relevant","rank":1},
 {"id":"card_304","relevance":"partially_relevant","rank":2}]}""")
check("irrelevant filtered out", all(x.id != "offer_1" for x in r.kept))
check("ranked best-first", [x.id for x in r.kept] == ["card_303", "card_304"],
      f"got {[x.id for x in r.kept]}")
check("verdict correct", r.verdict == "correct")
check("no retry needed", r.needs_retry is False)
check("relevant_count", r.relevant_count == 1)

print("--- CRAG incorrect branch: everything irrelevant ---")
r = grade_with("""{"records":[
 {"id":"card_303","relevance":"irrelevant","rank":1},
 {"id":"card_304","relevance":"irrelevant","rank":2},
 {"id":"offer_1","relevance":"irrelevant","rank":3}]}""")
check("nothing kept", r.kept == [])
check("verdict incorrect", r.verdict == "incorrect")
check("retry requested", r.needs_retry is True)

print("--- zero records retrieved: no LLM call at all ---")
fc = FakeClient("{}")
r = RecordGrader(llm_client=fc).grade("q", [])
check("no call made", fc.calls == 0, f"calls={fc.calls}")
check("verdict incorrect", r.verdict == "incorrect")
check("retry requested", r.needs_retry is True)

print("--- invented record id is discarded (whitelist) ---")
r = grade_with("""{"records":[
 {"id":"card_999_HALLUCINATED","relevance":"relevant","rank":1},
 {"id":"card_303","relevance":"relevant","rank":2}]}""")
check("bogus id absent", all(x.id != "card_999_HALLUCINATED" for x in r.kept))
check("warning recorded", any("card_999" in w for w in r.warnings), r.warnings)

print("--- unmentioned record is kept, not silently dropped ---")
r = grade_with("""{"records":[{"id":"card_303","relevance":"relevant","rank":1}]}""")
ids = [x.id for x in r.kept]
check("card_304 survived", "card_304" in ids, f"got {ids}")
check("relevant one ranks first", ids[0] == "card_303", f"got {ids}")
check("warned about it", any("did not judge" in w for w in r.warnings))

print("--- unusable label falls back, does not crash ---")
r = grade_with("""{"records":[
 {"id":"card_303","relevance":"VERY RELEVANT!!","rank":1},
 {"id":"card_304","relevance":"irrelevant","rank":2},
 {"id":"offer_1","relevance":"irrelevant","rank":3}]}""")
g303 = [g for g in r.graded if g.record.id == "card_303"][0]
check("clamped to fallback", g303.relevance == PARTIALLY_RELEVANT, g303.relevance)
check("still usable", "card_303" in [x.id for x in r.kept])

print("--- non-integer rank does not crash ---")
r = grade_with("""{"records":[
 {"id":"card_303","relevance":"relevant","rank":"first"},
 {"id":"card_304","relevance":"relevant","rank":1}]}""")
check("bad rank sorted last", [x.id for x in r.kept][0] == "card_304",
      f"got {[x.id for x in r.kept]}")

print("--- JSON wrapped in prose is recovered ---")
r = grade_with("""Sure! Here are my grades:
{"records":[{"id":"card_303","relevance":"relevant","rank":1}]} Hope that helps!""")
check("recovered", "card_303" in [x.id for x in r.kept])
check("warned", any("prose" in w for w in r.warnings), r.warnings)

print("--- total garbage degrades to retry, no exception ---")
r = grade_with("I'm sorry, I cannot do that.")
check("verdict incorrect", r.verdict == "incorrect")
check("retry requested", r.needs_retry is True)
check("records still available as fallback", len(r.kept) == 3, f"got {len(r.kept)}")
check("warning recorded", len(r.warnings) > 0)

print("--- all-partial means retry, but keeps records as fallback ---")
r = grade_with("""{"records":[
 {"id":"card_303","relevance":"partially_relevant","rank":1},
 {"id":"card_304","relevance":"partially_relevant","rank":2},
 {"id":"offer_1","relevance":"irrelevant","rank":3}]}""")
check("verdict incorrect (no fully-relevant record)", r.verdict == "incorrect")
check("retry requested", r.needs_retry is True)
check("fallback records kept", [x.id for x in r.kept] == ["card_303", "card_304"])

print("--- duplicate judgement keeps first ---")
r = grade_with("""{"records":[
 {"id":"card_303","relevance":"relevant","rank":1},
 {"id":"card_303","relevance":"irrelevant","rank":9}]}""")
g303 = [g for g in r.graded if g.record.id == "card_303"][0]
check("first kept", g303.relevance == RELEVANT, g303.relevance)

print("--- top_k cap is honoured ---")
many = [FakeRecord(f"r{i}", f"t{i}", "cards", "x") for i in range(10)]
rows = ",".join(
    '{"id":"r%d","relevance":"relevant","rank":%d}' % (i, i + 1)
    for i in range(10)
)
g = RecordGrader(llm_client=FakeClient('{"records":[%s]}' % rows), top_k=6)
r = g.grade("q", many)
check("capped at 6", len(r.kept) == 6, f"got {len(r.kept)}")
check("kept the best 6", [x.id for x in r.kept] == [f"r{i}" for i in range(6)])

print("--- grader prompt is compact (context safety) ---")
fc = FakeClient('{"records":[]}')
g = RecordGrader(llm_client=fc, max_record_chars=1200)
big = FakeRecord("card_303", "VISA INFINITE", "cards", "x" * 42000)
prompt = g._build_user_prompt("q", [big])
check("huge record truncated", len(prompt) < 2000, f"prompt={len(prompt)} chars")

print()
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
