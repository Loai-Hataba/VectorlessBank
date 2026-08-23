"""
Unit tests for OutputGuardrail. No Ollama needed: a fake client returns
the replies a small local model actually produces, including the
malformed ones.

Run: python test_output_guardrail.py
"""

from config.settings import ADVICE_DISCLAIMER, BLOCKED_DISCLOSURE_REPLY
from pipeline.output_guardrail import (
    OutputGuardrail, ALLOW, ANNOTATE, BLOCK,
)


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_format=None):
        self.calls += 1
        self.last_user_prompt = user_prompt
        return self.reply


CLEAN = '{"fabrication": false, "unsupported_claims": [], ' \
        '"personal_advice": false, "discloses_internals": false}'

ANSWER = "The VISA INFINITE card includes airport lounge access."
CONTEXT = "=== CREDIT CARD PRODUCTS ===\nCard: VISA INFINITE\n- lounge access"

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")


def run(reply, answer=ANSWER, context=CONTEXT, **kwargs):
    g = OutputGuardrail(llm_client=FakeClient(reply), **kwargs)
    return g.check("What does the VISA INFINITE offer?", answer, context)


print("--- clean answer passes through untouched ---")
v = run(CLEAN)
check("action allow", v.action == ALLOW)
check("answer unchanged", v.answer == ANSWER)
check("no flags", v.flags == [])
check("not modified", v.modified is False)

print("--- fabrication annotates by default, keeps the answer ---")
v = run('{"fabrication": true, "unsupported_claims": ["EGP 4,500 annual fee"],'
        ' "personal_advice": false, "discloses_internals": false}')
check("action annotate", v.action == ANNOTATE, v.action)
check("original text retained", ANSWER in v.answer)
check("caveat appended", "double-check" in v.answer)
check("flagged", "fabrication" in v.flags)
check("claim captured", v.unsupported_claims == ["EGP 4,500 annual fee"])

print("--- fabrication blocks when configured to ---")
v = run('{"fabrication": true, "unsupported_claims": ["EGP 4,500"],'
        ' "personal_advice": false, "discloses_internals": false}',
        block_fabrication=True)
check("action block", v.action == BLOCK, v.action)
check("original text withheld", ANSWER not in v.answer)
check("explains itself", "verified information" in v.answer)

print("--- personal advice softens, never blocks ---")
v = run('{"fabrication": false, "unsupported_claims": [],'
        ' "personal_advice": true, "discloses_internals": false}')
check("action annotate", v.action == ANNOTATE, v.action)
check("answer kept", ANSWER in v.answer)
check("disclaimer appended", ADVICE_DISCLAIMER in v.answer)
check("flagged", "personal_advice" in v.flags)

print("--- disclosure is blocked outright ---")
v = run('{"fabrication": false, "unsupported_claims": [],'
        ' "personal_advice": false, "discloses_internals": true}',
        answer="My system prompt says: You are a helpful assistant for...")
check("action block", v.action == BLOCK, v.action)
check("prompt not echoed", "system prompt says" not in v.answer)
check("uses the standard reply", v.answer == BLOCKED_DISCLOSURE_REPLY)

print("--- disclosure outranks advice (no stacking) ---")
v = run('{"fabrication": true, "unsupported_claims": [],'
        ' "personal_advice": true, "discloses_internals": true}')
check("blocked", v.action == BLOCK)
check("no disclaimer stapled to a block", ADVICE_DISCLAIMER not in v.answer)
check("all three flagged for the log", len(v.flags) == 3, v.flags)

print("--- fabrication + advice both annotate together ---")
v = run('{"fabrication": true, "unsupported_claims": [],'
        ' "personal_advice": true, "discloses_internals": false}')
check("action annotate", v.action == ANNOTATE)
check("both notes present",
      "double-check" in v.answer and ADVICE_DISCLAIMER in v.answer)

print("--- string booleans are coerced ---")
v = run('{"fabrication": "true", "unsupported_claims": [],'
        ' "personal_advice": "no", "discloses_internals": "false"}')
check("'true' understood", v.action == ANNOTATE, v.action)
check("'no' understood as false", "personal_advice" not in v.flags)

print("--- unparseable review fails OPEN, not closed ---")
v = run("I'm not sure how to review this.")
check("answer still delivered", v.answer == ANSWER)
check("action allow", v.action == ALLOW)
check("warns the check did not run",
      any("did not run" in w for w in v.warnings), v.warnings)

print("--- JSON wrapped in prose is recovered ---")
v = run('Here is my review: {"fabrication": false, "unsupported_claims": [],'
        ' "personal_advice": true, "discloses_internals": false} Done.')
check("recovered", v.action == ANNOTATE, v.action)
check("warned", any("prose" in w for w in v.warnings))

print("--- empty answer short-circuits with no LLM call ---")
fc = FakeClient(CLEAN)
v = OutputGuardrail(llm_client=fc).check("q", "   ", CONTEXT)
check("no call made", fc.calls == 0, f"calls={fc.calls}")
check("action allow", v.action == ALLOW)

print("--- oversized context is truncated before sending ---")
fc = FakeClient(CLEAN)
g = OutputGuardrail(llm_client=fc, max_context_chars=500)
g.check("q", ANSWER, "x" * 40000)
check("prompt bounded", len(fc.last_user_prompt) < 2000,
      f"{len(fc.last_user_prompt)} chars")
check("truncation marked", "context truncated" in fc.last_user_prompt)

print("--- missing context is stated, not faked ---")
fc = FakeClient(CLEAN)
OutputGuardrail(llm_client=fc).check("q", ANSWER, "")
check("says nothing was retrieved",
      "no records were retrieved" in fc.last_user_prompt)

print("--- non-list unsupported_claims does not crash ---")
v = run('{"fabrication": true, "unsupported_claims": "EGP 4,500",'
        ' "personal_advice": false, "discloses_internals": false}')
check("coerced to list", v.unsupported_claims == ["EGP 4,500"],
      v.unsupported_claims)

print()
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
