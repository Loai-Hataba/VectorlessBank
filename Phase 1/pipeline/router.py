"""
pipeline/router.py

WHAT THIS FILE DOES
--------------------
Given the contextualized question (already resolved to be standalone
by query_contextualizer.py), decides:
  1. does this question need retrieval at all, or is it a greeting /
     general-knowledge question the generator can handle on its own?
  2. if retrieval is needed, which of cards / offers / campaigns
     should actually be queried?

WHY IT EXISTS AS ITS OWN STEP
--------------------------------
Phase 1's rag_pipeline.py queried every source on every question and
let empty keyword-match results filter out the irrelevant ones (see
that file's own docstring for why that was an intentional, "dumb but
working" baseline). That wastes retrieval/traversal calls on sources
that were never going to match, and gives no signal for the case
where NO source should be queried at all -- e.g. "hi", or "what is
APR?", which prompt_templates.py's SYSTEM_PROMPT already says the
generator may answer from its own general knowledge. The Router
replaces "ask everyone" with an actual decision, without changing
anything about how the retrievers/tree traversal themselves work.

WHY THE ROUTER'S OUTPUT IS VALIDATED, NEVER TRUSTED DIRECTLY
--------------------------------------------------------------------
Same "model proposes, Python disposes" discipline as the tree
traverser's node-id validation in Phase 1: the LLM is asked for JSON
naming which sources to query, but Python is the only thing that
decides what's actually queried. Any source name that isn't in
config.settings.VALID_SOURCES is dropped rather than trusted, and if
the whole response is unusable, the router fails toward Phase 1's old
"query everything" behavior -- a known-working, if less efficient,
state -- rather than toward "query nothing," which would look
identical to a real "no relevant data" case and silently produce
worse answers with no error to point at.

INPUTS  : contextualized_question (str), session_id (str), turn_id (str)
OUTPUTS : dict: {"needs_retrieval": bool, "sources": list[str]}
          `sources` is always a subset of config.settings.VALID_SOURCES,
          and is empty whenever needs_retrieval is False.
"""

import json

import time

from config.settings import VALID_SOURCES
from llm.llm_client import LLMClient
from pipeline.pipeline_logger import log_stage


ROUTER_SYSTEM_PROMPT = """You are a routing component for a bank's customer \
support chatbot. You do not answer questions -- you only decide where to \
look for the answer.

Available data sources:
- "cards": credit card products -- fees, limits, benefits, eligibility.
- "offers": merchant installment and discount offers.
- "campaigns": time-limited promotional campaigns.

Given the customer's question, decide:
1. needs_retrieval: false if the question is a greeting/small talk, or a \
general banking knowledge question that doesn't require looking up this \
bank's specific data (e.g. "what is APR", "how do credit cards work"). \
true otherwise.
2. sources: which of "cards", "offers", "campaigns" are relevant. Include \
more than one if the question could plausibly touch more than one. Empty \
list if needs_retrieval is false.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"needs_retrieval": true, "sources": ["cards"]}
"""


class Router:

    def __init__(self):
        self.llm_client = LLMClient(role="traverser")

    def route(
        self,
        contextualized_question: str,
        session_id: str,
        turn_id: str,
    ) -> dict:

        raw_output = None
        started = time.time()

        try:
            raw_output = self.llm_client.generate(
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=contextualized_question,
                response_format="json",
            )
            decision = self._parse(raw_output)

        except (RuntimeError, ValueError):
            decision = None

        # A parsed decision that wanted retrieval but named only
        # invalid/hallucinated sources is just as unusable as a
        # decision that failed to parse at all -- both fall back the
        # same way, rather than silently retrieving from zero sources.
        if decision is not None and decision["needs_retrieval"] and not decision["sources"]:
            decision = None

        if decision is None:
            decision = {
                "needs_retrieval": True,
                "sources": sorted(VALID_SOURCES),
            }
            fell_back = True
        else:
            fell_back = False

        log_stage(
            stage="router",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "contextualized_question": contextualized_question,
                "raw_llm_output": raw_output,
                "decision": decision,
                "fell_back_to_query_all": fell_back,
            },
            duration_ms=(time.time() - started) * 1000,
            llm=self.llm_client.last_metrics,
        )

        return decision

    @staticmethod
    def _parse(raw_output: str) -> dict | None:
        """
        Turns the LLM's raw JSON text into a validated decision dict,
        or None if it can't be trusted.

        "Validated" means: it parses as JSON, has the two expected
        keys of the right types, and every source name is checked
        against VALID_SOURCES -- an invented source name is dropped,
        not passed through, exactly like the tree traverser drops
        invented node IDs in Phase 1.
        """

        try:
            data = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(data, dict):
            return None

        needs_retrieval = data.get("needs_retrieval")

        if not isinstance(needs_retrieval, bool):
            return None

        raw_sources = data.get("sources")

        if not isinstance(raw_sources, list):
            return None

        sources = [
            s for s in raw_sources
            if isinstance(s, str) and s in VALID_SOURCES
        ]

        if not needs_retrieval:
            sources = []

        return {
            "needs_retrieval": needs_retrieval,
            "sources": sources,
        }