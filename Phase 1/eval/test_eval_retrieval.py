"""Check the tier-1 scoring maths and the testset's integrity."""
import json
from pathlib import Path
import sys

sys.path.insert(0, r"C:\Users\Omart\Documents\GitHub\VectorlessBank\.claude\worktrees\phase2-partner-b\Phase 1")
sys.path.insert(0, r"C:\Users\Omart\Documents\GitHub\VectorlessBank\.claude\worktrees\phase2-partner-b\Phase 1\eval")

from eval_retrieval import prf, TESTSET_PATH

p = f = 0


def check(name, cond, detail=""):
    global p, f
    if cond:
        p += 1
        print(f"  PASS  {name}")
    else:
        f += 1
        print(f"  FAIL  {name} {detail}")


print("--- precision/recall/F1 ---")
check("perfect", prf({"a"}, {"a"}) == (1.0, 1.0, 1.0))
check("half precision", prf({"a", "b"}, {"a"})[0] == 0.5)
check("half recall", prf({"a"}, {"a", "b"})[1] == 0.5)
check("total miss", prf({"z"}, {"a"}) == (0.0, 0.0, 0.0))
check("nothing retrieved", prf(set(), {"a"}) == (0.0, 0.0, 0.0))
check("empty gold, nothing retrieved = perfect", prf(set(), set()) == (1.0, 1.0, 1.0))
check("empty gold, something retrieved = 0 precision", prf({"a"}, set())[0] == 0.0)
check("empty gold recall stays 1.0", prf({"a"}, set())[1] == 1.0)

print("--- testset integrity ---")
data = json.loads(Path(TESTSET_PATH).read_text(encoding="utf-8"))
cases = data["cases"]
print(f"  {len(cases)} cases")

ids = [c["id"] for c in cases]
check("ids unique", len(ids) == len(set(ids)))
check("all have question", all(c.get("question") for c in cases))
check("all have category", all(c.get("category") for c in cases))
check("all have note", all(c.get("note") for c in cases))

# Gold IDs must exist in the real data.
from loaders.cards_loader import CardsLoader
from loaders.offers_loader import OffersLoader
from loaders.campaigns_loader import CampaignsLoader
real = {r.id for r in (CardsLoader().load() + OffersLoader().load()
                       + CampaignsLoader().load())}
bad = []
for c in cases:
    for g in c.get("gold_record_ids", []):
        if g not in real:
            bad.append((c["id"], g))
check("every gold id exists in loaded data", not bad, f"bogus: {bad}")

# Follow-ups must carry history and a resolved question.
fu = [c for c in cases if c["category"] == "follow_up"]
check("follow-ups have history", all(c.get("history") for c in fu))
check("follow-ups have standalone_question",
      all(c.get("standalone_question") for c in fu))

# Coverage required by blueprint section 10.1
cats = {c["category"] for c in cases}
for required in ["single_source_cards", "single_source_offers",
                 "single_source_campaigns", "multi_source", "follow_up",
                 "general_knowledge", "unanswerable"]:
    check(f"covers {required}", required in cats)
check("covers guardrail cases", any(c.startswith("guardrail") for c in cats))

scored = [c for c in cases if c.get("needs_retrieval", True)]
print(f"  {len(scored)} scored by tier-1, {len(cases) - len(scored)} router-only")

print()
print(f"RESULT: {p} passed, {f} failed")
raise SystemExit(1 if f else 0)
