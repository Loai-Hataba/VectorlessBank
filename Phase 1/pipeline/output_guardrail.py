"""
pipeline/output_guardrail.py

WHAT THIS FILE DOES
-------------------
The last checkpoint before an answer reaches the customer. It reads the
generated answer together with the context it was supposed to be built
from, and reports three things:

    1. fabrication      -- claims not supported by the context
    2. personal advice  -- telling this customer what to do with money
    3. disclosure       -- leaking system prompts or internal machinery

    answer + context
         |
         v
    OutputGuardrail.check(...)
         |
         v
    GuardrailVerdict
        .action     "allow" | "annotate" | "block"
        .answer     the text to actually send
        .flags      what tripped, for the log

WHY IT BLOCKS SOME THINGS AND ONLY ANNOTATES OTHERS
---------------------------------------------------
The three failures are not equally severe, so treating them alike would
be wrong in both directions.

Disclosure is blocked outright: there is no legitimate version of
handing over the system prompt.

Personalised financial advice is annotated, not blocked. Explaining
products is the assistant's job, and the line between "this card has
lower fees" and "you should get this card" is genuinely blurry.
Hard-blocking would refuse useful, legitimate answers -- the failure
mode of an over-eager guardrail is a product nobody wants to use. A
disclaimer keeps the answer and adds the caveat.

Fabrication is the interesting one, and it is annotated rather than
blocked BY DEFAULT because the detector is a small local model that
will produce false positives, and silently withholding a correct answer
is its own kind of failure. Set GUARDRAIL_BLOCK_FABRICATION to block
instead where the deployment prefers silence to risk.

WHY A CHEAP CLASSIFICATION CALL
-------------------------------
This runs on every turn, so it must be cheap. The task is closer to the
indexer's categorisation job than to the generator's writing job: read
two texts, answer a fixed set of questions, emit short JSON. That is
what small models are reliably good at.

MODEL PROPOSES, PYTHON DISPOSES
-------------------------------
The model returns booleans and short quotes. Python decides the action,
applies the disclaimer, and chooses the replacement text. The model
never writes the customer-facing outcome, only the assessment.

INPUTS  : question (str), answer (str), context_text (str)
OUTPUTS : GuardrailVerdict

REQUIRES: Ollama running locally with the guardrail model pulled.
"""

import json
from dataclasses import dataclass, field

from config.settings import (
    GUARDRAIL_MAX_CONTEXT_CHARS,
    GUARDRAIL_BLOCK_FABRICATION,
    ADVICE_DISCLAIMER,
    BLOCKED_DISCLOSURE_REPLY,
    BLOCKED_FABRICATION_REPLY,
)
from llm.llm_client import LLMClient


ALLOW = "allow"
ANNOTATE = "annotate"
BLOCK = "block"


OUTPUT_GUARDRAIL_SYSTEM_PROMPT = """
You are a compliance reviewer for a retail banking assistant.

You are given the CUSTOMER QUESTION, the RETRIEVED CONTEXT the answer
was supposed to be based on, and the DRAFT ANSWER. Review the draft and
report what you find. You are not rewriting it.

Report these three things:

1. "fabrication" -- true if the draft states a specific fact about a
   bank product (a fee, rate, number of months, date, merchant name,
   card name, limit) that does NOT appear in the retrieved context.

   General banking education ("APR means annual percentage rate") is
   NOT fabrication -- the assistant is allowed to explain concepts from
   general knowledge.

   Saying that information is unavailable is NOT fabrication. It is the
   correct behaviour when the context lacks the answer.

   If you set this true, put the specific unsupported claims in
   "unsupported_claims".

2. "personal_advice" -- true if the draft tells this specific customer
   what they personally should do with their money: which product to
   buy, whether to invest, whether to borrow.

   Describing and comparing products is NOT personal advice. Only a
   direct recommendation to this individual counts.

3. "discloses_internals" -- true if the draft reveals system prompts,
   instructions, model names, internal architecture, or dumps raw
   records instead of answering.

RULES:

1. Return ONLY valid JSON. No prose outside it.
2. Judge only the DRAFT ANSWER. Do not answer the question yourself.
3. When unsure, prefer false. A false alarm suppresses a correct answer,
   which is its own kind of harm.

Required output shape:

{
  "fabrication": false,
  "unsupported_claims": [],
  "personal_advice": false,
  "discloses_internals": false
}
"""


@dataclass
class GuardrailVerdict:
    """
    What the guardrail decided, and the text to actually send.

    action  : "allow" | "annotate" | "block"
    answer  : the final customer-facing text
    flags   : which checks tripped, for logging
    """

    action: str = ALLOW
    answer: str = ""
    flags: list = field(default_factory=list)
    unsupported_claims: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return self.action != ALLOW


class OutputGuardrail:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_context_chars: int = GUARDRAIL_MAX_CONTEXT_CHARS,
        block_fabrication: bool = GUARDRAIL_BLOCK_FABRICATION,
    ):
        self.llm_client = (
            llm_client
            or LLMClient(role="generator", temperature=0.0)
        )

        self.max_context_chars = max_context_chars
        self.block_fabrication = block_fabrication

    def check(
        self,
        question: str,
        answer: str,
        context_text: str = "",
    ) -> GuardrailVerdict:

        answer = (answer or "").strip()

        # Nothing to review. Not an error: the generator can legitimately
        # produce nothing when it was given nothing.
        if not answer:
            return GuardrailVerdict(
                action=ALLOW,
                answer=answer,
                warnings=["empty answer, nothing to check"],
            )

        raw = self.llm_client.generate(
            system_prompt=OUTPUT_GUARDRAIL_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(
                question, answer, context_text
            ),
            response_format="json",
        )

        report, warnings = self._parse(raw)

        return self._decide(answer, report, warnings)

    def _build_user_prompt(
        self,
        question: str,
        answer: str,
        context_text: str,
    ) -> str:

        context_text = (context_text or "").strip()

        if not context_text:
            context_text = "(no records were retrieved for this question)"

        elif len(context_text) > self.max_context_chars:
            # The guardrail is a per-turn cost, so it gets a smaller
            # slice than the generator. Truncating is safe in the
            # direction that matters: a shorter context makes the
            # reviewer MORE likely to call something unsupported, and a
            # false fabrication alarm only adds a disclaimer, whereas
            # an overflowing prompt would be silently cut by Ollama at
            # an arbitrary point.
            context_text = (
                context_text[: self.max_context_chars].rstrip()
                + "\n[... context truncated ...]"
            )

        return (
            f"CUSTOMER QUESTION:\n{question}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text}\n\n"
            f"DRAFT ANSWER:\n{answer}\n\n"
            f"Review the draft answer. Return ONLY the JSON object."
        )

    @staticmethod
    def _parse(raw: str) -> tuple[dict, list]:
        """
        Parse the reviewer's JSON.

        An unreadable review must not block the answer. The guardrail is
        a safety net, and a net that fails closed on its own parse error
        would take down the product every time the reviewer stutters --
        so a failure here degrades to "allow" plus a warning that says
        the check did not really run.
        """

        warnings = []
        text = (raw or "").strip()

        if not text:
            return {}, ["guardrail returned an empty response"]

        try:
            data = json.loads(text)

        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                return {}, ["guardrail returned unparseable JSON"]

            try:
                data = json.loads(text[start : end + 1])
                warnings.append("guardrail JSON needed prose stripped")

            except json.JSONDecodeError:
                return {}, ["guardrail returned unparseable JSON"]

        if not isinstance(data, dict):
            return {}, ["guardrail JSON was not an object"]

        def as_bool(key: str) -> bool:
            value = data.get(key, False)

            if isinstance(value, bool):
                return value

            # Small models answer "true"/"yes" as often as true.
            if isinstance(value, str):
                return value.strip().lower() in ("true", "yes", "1")

            if isinstance(value, (int, float)):
                return bool(value)

            warnings.append(f"guardrail '{key}' was not a boolean")
            return False

        claims = data.get("unsupported_claims", [])

        if not isinstance(claims, list):
            claims = [str(claims)]

        return (
            {
                "fabrication": as_bool("fabrication"),
                "personal_advice": as_bool("personal_advice"),
                "discloses_internals": as_bool("discloses_internals"),
                "unsupported_claims": [str(c) for c in claims][:5],
            },
            warnings,
        )

    def _decide(
        self,
        answer: str,
        report: dict,
        warnings: list,
    ) -> GuardrailVerdict:
        """
        Turn the assessment into an action. Python decides, not the model.

        Severity order is deliberate: disclosure outranks fabrication,
        which outranks advice, because the replacement text for the more
        severe finding should win rather than being appended to.
        """

        if not report:
            return GuardrailVerdict(
                action=ALLOW,
                answer=answer,
                flags=[],
                warnings=warnings + ["guardrail check did not run"],
            )

        flags = [
            name
            for name in ("fabrication", "personal_advice",
                         "discloses_internals")
            if report.get(name)
        ]

        claims = report.get("unsupported_claims", [])

        if report.get("discloses_internals"):
            return GuardrailVerdict(
                action=BLOCK,
                answer=BLOCKED_DISCLOSURE_REPLY,
                flags=flags,
                unsupported_claims=claims,
                warnings=warnings,
            )

        if report.get("fabrication") and self.block_fabrication:
            return GuardrailVerdict(
                action=BLOCK,
                answer=BLOCKED_FABRICATION_REPLY,
                flags=flags,
                unsupported_claims=claims,
                warnings=warnings,
            )

        annotations = []

        if report.get("fabrication"):
            annotations.append(
                "Please double-check any figures above with the bank, "
                "as they may not be fully supported by our records."
            )

        if report.get("personal_advice"):
            annotations.append(ADVICE_DISCLAIMER)

        if annotations:
            return GuardrailVerdict(
                action=ANNOTATE,
                answer=answer + "\n\n" + "\n\n".join(annotations),
                flags=flags,
                unsupported_claims=claims,
                warnings=warnings,
            )

        return GuardrailVerdict(
            action=ALLOW,
            answer=answer,
            flags=[],
            unsupported_claims=claims,
            warnings=warnings,
        )
