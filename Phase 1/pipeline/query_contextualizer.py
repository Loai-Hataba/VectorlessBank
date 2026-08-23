"""
pipeline/query_contextualizer.py

WHAT THIS FILE DOES
--------------------
Takes the user's raw message plus their conversation memory (rolling
summary + recent raw turns) and rewrites it into a standalone
question -- one that makes sense with NO prior context. This is the
question every downstream stage (guardrail, router, retrievers) sees;
they never see the user's literal follow-up wording.

    "What's the annual fee on the VISA Infinite?"
    "What about the platinum one?"
        -> contextualized to:
    "What's the annual fee on the VISA Platinum?"

WHY IT EXISTS AS ITS OWN STEP
--------------------------------
The Router (next stage) decides which data sources to query based on
the text of the question. If it saw "what about the platinum one?"
with no context, it has nothing to route on -- "platinum" alone
doesn't say whether that's a card, a campaign tier, or an offer. By
resolving pronouns/references BEFORE routing, every later stage can
stay simple and treat each question as if it were asked cold, with
all the context it needs already folded in.

WHY THIS SKIPS THE LLM ON A FIRST TURN
--------------------------------------------
If conversation_memory has no summary and no raw turns yet, the raw
message IS already standalone by definition -- there's nothing prior
to resolve it against. Calling the LLM anyway would just add latency
and give it a chance to accidentally reword a fine question into a
worse one.

WHY THE ORIGINAL QUESTION IS THE FALLBACK, NOT AN EMPTY STRING
--------------------------------------------------------------------
Per the project's "model proposes, Python disposes" rule (Phase 2
blueprint §13), the rewrite is treated as a claim to validate, not a
fact to trust blindly. If the LLM call fails or returns something
unusable (empty after stripping), downstream stages get the user's
original raw question instead -- worse context resolution is a much
smaller failure than silently sending an empty question to the router.

INPUTS  : raw_question (str), memory_context (dict from
          ConversationMemory.get_context()), session_id (str),
          turn_id (str)
OUTPUTS : contextualized_question (str)
"""

from llm.llm_client import LLMClient
from pipeline.pipeline_logger import log_stage


CONTEXTUALIZER_SYSTEM_PROMPT = """You rewrite a customer's latest message \
into a standalone question, using the conversation history provided.

Rules:
1. If the latest message is already standalone (doesn't depend on \
anything said earlier), return it UNCHANGED.
2. Resolve pronouns and references ("it", "that one", "the second \
option", "what about X instead") into the specific thing they refer \
to, using the conversation history.
3. Do not answer the question. Do not add information that wasn't in \
the message or the history. Only rewrite for clarity/standalone-ness.
4. Output ONLY the rewritten question, nothing else -- no preamble, \
no quotes around it, no explanation.
"""


class QueryContextualizer:

    def __init__(self):
        self.llm_client = LLMClient(role="summarizer")

    def contextualize(
        self,
        raw_question: str,
        memory_context: dict,
        session_id: str,
        turn_id: str,
    ) -> str:

        summary = memory_context.get("summary", "")
        raw_turns = memory_context.get("raw_turns", [])

        if not summary and not raw_turns:
            # Nothing to resolve against -- this IS the standalone
            # question already.
            return raw_question

        history_text = self._format_history(summary, raw_turns)

        user_prompt = (
            f"CONVERSATION HISTORY:\n"
            f"{history_text}\n\n"
            f"LATEST MESSAGE: {raw_question}"
        )

        try:
            rewritten = self.llm_client.generate(
                system_prompt=CONTEXTUALIZER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            ).strip()

        except RuntimeError:
            rewritten = ""

        contextualized_question = rewritten or raw_question

        log_stage(
            stage="contextualizer",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "raw_question": raw_question,
                "memory_summary": summary,
                "memory_raw_turn_count": len(raw_turns),
                "contextualized_question": contextualized_question,
                "fell_back_to_raw": not bool(rewritten),
            },
        )

        return contextualized_question

    @staticmethod
    def _format_history(summary: str, raw_turns: list) -> str:

        parts = []

        if summary:
            parts.append(f"Summary of earlier conversation: {summary}")

        for turn in raw_turns:
            parts.append(f"Customer: {turn['user']}")
            parts.append(f"Assistant: {turn['assistant']}")

        return "\n".join(parts)