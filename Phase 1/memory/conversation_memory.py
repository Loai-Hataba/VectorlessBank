"""
memory/conversation_memory.py

WHAT THIS FILE DOES
--------------------
Tracks, per session_id, what's been said in a conversation so far:
  - a small buffer of the most recent raw turns, kept verbatim
  - a rolling plain-text summary of everything older than that

Two other Phase 2 modules read this: the Query Contextualizer (needs
recent turns + summary to rewrite a follow-up question into a
standalone one) and pipeline_logger (session_id/turn_id are logged
on every stage, defined here since a "turn" is a conversation_memory
concept).

WHY IT EXISTS AS ITS OWN STEP
--------------------------------
Phase 1 answered each question with zero memory of anything said
before it -- "what about the second one?" had no way to resolve
"the second one" to anything. Keeping ALL raw turns forever would
work for short chats but would eventually blow past any model's
context window and slow every single call down as the conversation
grew. Keeping a small raw window plus a rolling summary bounds the
memory size while still preserving the gist of a long conversation.

WHY THIS IS IN-MEMORY, NOT A DATABASE
------------------------------------------
Per the Phase 2 blueprint's scope, this is a single-process Flask app
with no persistence requirement across restarts. A plain dict keyed
by session_id is enough, and restarting the process is expected to
wipe conversation state -- that's a scope decision, not a bug.

WHY SUMMARIZATION IS LAZY
-------------------------------
The summarizer LLM call only runs when a turn is actually about to
fall out of the raw buffer, not on every turn. Calling it on every
single message would add an LLM call BEFORE the guardrail/router even
run, doubling latency on every short exchange for no benefit -- most
conversations in this assignment's scope will never exceed the raw
window at all.

INPUTS  : session_id (str), user_message (str), assistant_message (str)
OUTPUTS : ConversationMemory.get_context(session_id) -> dict:
            {
              "summary": str,            # "" if nothing summarized yet
              "raw_turns": list[dict],   # [{"user": ..., "assistant": ...}, ...]
            }
"""

import itertools

from config.settings import MEMORY_RAW_TURNS_KEPT
from llm.llm_client import LLMClient
from pipeline.pipeline_logger import log_stage


SUMMARIZER_SYSTEM_PROMPT = """You maintain a running summary of a customer \
support conversation with a bank's chatbot.

You will be given the CURRENT SUMMARY (may be empty, if this is the \
first update) and one OLDER TURN that is about to be dropped from the \
recent-turns window.

Update the summary to fold in that turn. Rules:
1. Keep it short -- a few sentences, not a transcript.
2. Preserve concrete details that later questions might refer back to: \
card names, merchant names, amounts, dates, and what the customer was \
trying to do.
3. Do not lose information that was already in the current summary --  \
you are ADDING to it, not replacing it with only the new turn.
4. Write plain prose, no bullet points, no preamble like "Here is the \
updated summary:".
"""


class ConversationMemory:

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._turn_counters: itertools.count = None
        self._turn_counters_by_session: dict[str, itertools.count] = {}
        self.summarizer = LLMClient(role="summarizer")

    def next_turn_id(self, session_id: str) -> str:
        """
        One incrementing counter per session, e.g. "3" for the third
        user message in that session. Kept separate from the message
        content itself since pipeline_logger needs it before any
        stage has produced anything to log.
        """
        counter = self._turn_counters_by_session.setdefault(
            session_id,
            itertools.count(start=1),
        )
        return str(next(counter))

    def get_context(self, session_id: str) -> dict:
        """
        Returns the memory a Query Contextualizer needs: the rolling
        summary (possibly empty) and the raw turns kept verbatim, in
        chronological order.
        """
        state = self._sessions.get(session_id)

        if state is None:
            return {"summary": "", "raw_turns": []}

        return {
            "summary": state["summary"],
            "raw_turns": list(state["raw_turns"]),
        }

    def add_turn(
        self,
        session_id: str,
        turn_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Records one completed turn (a user message + the assistant's
        final answer to it). Call this AFTER the answer has been
        generated, once per turn -- never mid-pipeline.

        If this pushes the raw buffer past MEMORY_RAW_TURNS_KEPT, the
        oldest raw turn is folded into the rolling summary via the
        summarizer LLM and dropped from the raw list.
        """
        state = self._sessions.setdefault(
            session_id,
            {"summary": "", "raw_turns": []},
        )

        state["raw_turns"].append(
            {"user": user_message, "assistant": assistant_message}
        )

        if len(state["raw_turns"]) > MEMORY_RAW_TURNS_KEPT:
            oldest_turn = state["raw_turns"].pop(0)
            state["summary"] = self._fold_into_summary(
                session_id=session_id,
                turn_id=turn_id,
                current_summary=state["summary"],
                turn=oldest_turn,
            )

    def _fold_into_summary(
        self,
        session_id: str,
        turn_id: str,
        current_summary: str,
        turn: dict,
    ) -> str:

        user_prompt = (
            f"CURRENT SUMMARY:\n"
            f"{current_summary if current_summary else '(none yet)'}\n\n"
            f"OLDER TURN TO FOLD IN:\n"
            f"Customer: {turn['user']}\n"
            f"Assistant: {turn['assistant']}"
        )

        new_summary = self.summarizer.generate(system_prompt=SUMMARIZER_SYSTEM_PROMPT,user_prompt=user_prompt,).strip()
        
        log_stage(
            stage="memory_summarize",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "previous_summary": current_summary,
                "folded_turn": turn,
                "new_summary": new_summary,
            },
        )

        return new_summary