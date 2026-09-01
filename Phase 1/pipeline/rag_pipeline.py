"""
pipeline/rag_pipeline.py

WHAT THIS FILE DOES
--------------------
This is the "conductor" that runs one full question-answering turn:

    user question
        -> ask EVERY retriever for its best matches (cards, offers, campaigns)
        -> merge the results into one context block
        -> build the final prompt
        -> call the LLM
        -> return the answer (plus the retrieved records, for transparency)

WHY PHASE 1's FLOW IS "DUMB" ON PURPOSE
-------------------------------------------
Notice this pipeline does NOT decide which source is relevant --
it queries ALL of them, every time, and lets the keyword scores
naturally return nothing (empty list) from sources that don't match.
That's intentional: Phase 1's goal is a working, understandable
baseline. Phase 2 replaces "ask everyone" with an `agent/router.py`
step that uses the LLM to decide which source(s) to query -- but
that step will simply choose WHICH of these same retrievers to call.
Nothing here needs to be rewritten for that upgrade, only the
decision of which retrievers to invoke.

WHY WE RETURN retrieved_records TOO, NOT JUST THE ANSWER
------------------------------------------------------------
Returning what was actually retrieved (not just the final text) is
essential for you to debug/understand the system, and it's exactly
what Phase 2's pipeline_logger.py will start writing to a log file
automatically.

INPUTS  : question (str) -- the user's message
OUTPUTS : a dict: {
            "answer": str,
            "retrieved_records": list[Record],
            "context_text": str,
          }
"""

import time

from retrievers.tree_retriever import TreeRetriever
from pipeline.context_builder import build_context
from pipeline.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from llm.llm_client import LLMClient
from loaders.cards_loader import CardsLoader
from loaders.offers_loader import OffersLoader
from loaders.campaigns_loader import CampaignsLoader
from config.settings import MAX_RESULTS_PER_SOURCE, VALID_SOURCES
from pipeline.pipeline_logger import log_stage
from pipeline.input_guardrail import InputGuardrail
from pipeline.query_contextualizer import QueryContextualizer
from pipeline.router import Router
from pipeline.grader import RecordGrader
from pipeline.output_guardrail import OutputGuardrail
from memory.conversation_memory import ConversationMemory


# What each stage is called when a CUSTOMER is watching it happen.
#
# The log uses the internal stage names ("guardrail_input"); a person
# waiting on an answer should not have to. These are deliberately about
# what is happening to THEIR question, not which component is running.
STAGE_LABELS = {
    "guardrail_input": "Checking your question",
    "contextualizer": "Reading our conversation",
    "router": "Deciding where to look",
    "retrieval": "Searching",
    "crag_grade": "Reviewing what I found",
    "crag_retry": "Looking further afield",
    "generate": "Writing your answer",
    "guardrail_output": "Checking my answer",
}


def _stage_event(stage: str, detail: str = "") -> tuple:
    """One progress event, in the shape every consumer expects."""

    return (
        "stage",
        {
            "stage": stage,
            "label": STAGE_LABELS.get(stage, stage),
            "detail": detail,
        },
    )


class RagPipeline:
    def __init__(self):
        # The tree index only stores record IDs, so the actual Records
        # still have to be loaded here for the retrievers to resolve
        # those IDs back into answerable content.
        self.cards_records = CardsLoader().load()
        self.offers_records = OffersLoader().load()
        self.campaigns_records = CampaignsLoader().load()

        self.retrievers = {
            "cards": TreeRetriever(
                source="cards",
                records=self.cards_records,
            ),
            "offers": TreeRetriever(
                source="offers",
                records=self.offers_records,
            ),
            "campaigns": TreeRetriever(
                source="campaigns",
                records=self.campaigns_records,
            ),
        }
        self.input_guardrail = InputGuardrail()
        self.query_contextualizer = QueryContextualizer()
        self.router = Router()
        self.conversation_memory = ConversationMemory()
        self.grader = RecordGrader()
        self.output_guardrail = OutputGuardrail()
        self.llm_client = LLMClient()

    def answer(self, question: str, session_id: str) -> dict:
        """
        Run one turn and return only the finished result.

        Kept as the simple entry point -- the Flask JSON route and
        every test use it. It drains _run_turn() rather than
        duplicating the orchestration, so there is exactly one
        description of what a turn does, and the progress-reporting
        path can never fall out of step with this one.
        """

        result = None

        for kind, payload in self._run_turn(question, session_id):
            if kind == "result":
                result = payload

        return result

    def answer_with_progress(self, question: str, session_id: str):
        """
        Run one turn, reporting which step it is on as it goes.

        Yields ("stage", {...}) as each step begins and finally
        ("result", {...}) carrying the same dict answer() returns.

        WHY PROGRESS BUT NOT THE ANSWER ITSELF
        --------------------------------------
        An earlier version also streamed the answer token by token.
        That is removed, for two reasons that both point the same way.

        It bought almost nothing. Generation is the fifth of six
        steps, so on a real question the first token did not appear
        until 110s into a 125s turn -- 88% of the wait was already
        over. What makes the wait legible is knowing the assistant is
        SEARCHING rather than stalled, and that is these stage events,
        which start at 0.0s.

        And it cost something real. The output guardrail can only
        judge a finished answer, so streaming necessarily puts
        unchecked text on screen -- text that may then be amended or
        withdrawn. Paying that for a 12% improvement in perceived
        latency was a bad trade.

        So nothing reaches the customer until the guardrail has passed
        it, exactly as in answer().
        """

        return self._run_turn(question, session_id)

    def _run_turn(self, question: str, session_id: str):
        """
        The single description of one question-answering turn.

        A generator, so callers who want progress get it and callers
        who do not can drain it -- see answer() and
        answer_with_progress().
        Every stage is timed and logged with what its model call
        actually cost, because on this hardware "which stage was slow"
        and "why was it slow" are different questions.
        """

        turn_started = time.time()
        turn_id = self.conversation_memory.next_turn_id(session_id)
        stage_ms = {}

        began = _stage_event

        def finished(stage, started_at):
            elapsed = int((time.time() - started_at) * 1000)
            stage_ms[stage] = elapsed
            return elapsed

        # ---- 1. is this a question we will answer at all ------------
        yield began("guardrail_input")
        t = time.time()

        guardrail_result = self.input_guardrail.check(
            raw_message=question,
            session_id=session_id,
            turn_id=turn_id,
        )

        finished("guardrail_input", t)

        if guardrail_result["blocked"]:
            log_stage(
                stage="turn_complete",
                session_id=session_id,
                turn_id=turn_id,
                data={"outcome": "blocked_at_input", "stage_ms": stage_ms},
                duration_ms=(time.time() - turn_started) * 1000,
            )

            yield (
                "result",
                {
                    "answer": guardrail_result["user_facing_message"],
                    "retrieved_records": [],
                    "context_text": "",
                },
            )
            return

        # ---- 2. resolve "what about that one?" into a real question --
        memory_context = self.conversation_memory.get_context(session_id)

        yield began("contextualizer")
        t = time.time()

        contextualized_question = self.query_contextualizer.contextualize(
            raw_question=question,
            memory_context=memory_context,
            session_id=session_id,
            turn_id=turn_id,
        )

        finished("contextualizer", t)

        # ---- 3. which sources, if any -------------------------------
        yield began("router")
        t = time.time()

        routing_decision = self.router.route(
            contextualized_question=contextualized_question,
            session_id=session_id,
            turn_id=turn_id,
        )

        finished("router", t)

        # ---- 4. retrieve and grade ----------------------------------
        if routing_decision["needs_retrieval"]:
            t = time.time()

            retrieved_records = []
            data_was_insufficient = False

            for kind, payload in self._retrieve_and_grade_stream(
                question=contextualized_question,
                sources=routing_decision["sources"],
                session_id=session_id,
                turn_id=turn_id,
            ):
                if kind == "kept":
                    retrieved_records, data_was_insufficient = payload
                else:
                    yield (kind, payload)

            finished("retrieval", t)
        else:
            # The router decided this is general banking knowledge. No
            # retrieval means nothing to grade, and an empty context is
            # not "insufficient data" -- it is the correct state.
            retrieved_records = []
            data_was_insufficient = False

        context_text = build_context(retrieved_records)

        user_prompt = build_user_prompt(
            contextualized_question,
            context_text,
            data_was_insufficient=data_was_insufficient,
        )

        # ---- 5. write the answer ------------------------------------
        yield began("generate")
        t = time.time()

        answer_text = self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        finished("generate", t)

        log_stage(
            stage="generation",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "context_chars": len(context_text),
                "records_in_context": len(retrieved_records),
                "answer_chars": len(answer_text),
                "data_was_insufficient": data_was_insufficient,
            },
            duration_ms=stage_ms["generate"],
            llm=getattr(self.llm_client, "last_metrics", None),
        )

        # ---- 6. last checkpoint before the customer sees anything ---
        yield began("guardrail_output")
        t = time.time()

        verdict = self.output_guardrail.check(
            question=contextualized_question,
            answer=answer_text,
            context_text=context_text,
        )

        finished("guardrail_output", t)

        log_stage(
            stage="guardrail_output",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "action": verdict.action,
                "flags": verdict.flags,
                "unsupported_claims": verdict.unsupported_claims,
                "warnings": verdict.warnings,
            },
            duration_ms=stage_ms["guardrail_output"],
            llm=getattr(
                self.output_guardrail.llm_client, "last_metrics", None
            ),
        )

        answer_text = verdict.answer

        # Memory records what the customer was actually told, not the
        # draft. Otherwise a follow-up would be resolved against an
        # answer that was blocked or amended before it was sent.
        self.conversation_memory.add_turn(
            session_id=session_id,
            turn_id=turn_id,
            user_message=question,
            assistant_message=answer_text,
        )

        # One line per turn carrying the whole shape of it, so "where
        # did the time go" is a single read rather than a join across
        # every stage line.
        log_stage(
            stage="turn_complete",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "outcome": verdict.action,
                "needed_retrieval": routing_decision["needs_retrieval"],
                "sources": sorted(routing_decision["sources"]),
                "records_kept": len(retrieved_records),
                "data_was_insufficient": data_was_insufficient,
                "stage_ms": stage_ms,
                "slowest_stage": (
                    max(stage_ms, key=stage_ms.get) if stage_ms else None
                ),
            },
            duration_ms=(time.time() - turn_started) * 1000,
        )

        yield (
            "result",
            {
                "answer": answer_text,
                "retrieved_records": retrieved_records,
                "context_text": context_text,
            },
        )

    def _retrieve_and_grade(
        self,
        question: str,
        sources: list,
        session_id: str,
        turn_id: str,
    ) -> tuple[list, bool]:
        """
        Retrieve, grade and correct -- returning only the outcome.

        A thin drain of _retrieve_and_grade_stream(), kept because most
        callers and every test want the tuple, not the progress.
        """

        for kind, payload in self._retrieve_and_grade_stream(
            question=question,
            sources=sources,
            session_id=session_id,
            turn_id=turn_id,
        ):
            if kind == "kept":
                return payload

        return [], True

    def _retrieve_and_grade_stream(
        self,
        question: str,
        sources: list,
        session_id: str,
        turn_id: str,
    ):
        """
        Retrieve, grade, and correct once if the grade comes back bad.

        Yields ("stage", {...}) as each step begins and finally
        ("kept", (records, data_was_insufficient)). A generator because
        retrieval is the slowest part of a turn by a wide margin -- on
        this hardware the offers search alone is over a minute -- and a
        customer watching a spinner deserves to know which of those
        minutes is which.

        Returns the records worth generating from, and whether the
        pipeline should tell the generator outright that it does not
        have enough grounded data.

        THE CORRECTIVE BRANCH
        ---------------------
        We drop CRAG's "search the web" arm entirely. Pulling unvetted
        external content into a banking assistant is not meaningfully
        different from letting the model invent facts, and it directly
        contradicts the grounded-answers requirement. Widening the
        INTERNAL search is the equivalent move here: re-ask the sources
        the router chose not to consult.

        Exactly one retry, always. That bounds worst-case latency, which
        matters a great deal when every call is a local model, and it
        removes any possibility of a correction loop.
        """

        yield _stage_event("retrieval", detail=", ".join(sorted(sources)))
        t = time.time()

        retrieved = self._retrieve_from(question, sources=sources)

        retrieval_ms = int((time.time() - t) * 1000)

        yield _stage_event("crag_grade")
        t = time.time()

        grade = self.grader.grade(question, retrieved)

        log_stage(
            stage="crag_grade",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "searched": sorted(sources),
                "retrieval_ms": retrieval_ms,
                "retrieved": len(retrieved),
                "kept": [r.id for r in grade.kept],
                "verdict": grade.verdict,
                "relevant": grade.relevant_count,
                "warnings": grade.warnings,
            },
            duration_ms=(time.time() - t) * 1000,
            llm=getattr(self.grader.llm_client, "last_metrics", None),
        )

        if not grade.needs_retry:
            yield ("kept", (grade.kept, False))
            return

        unsearched = [s for s in VALID_SOURCES if s not in sources]

        if unsearched:

            yield _stage_event(
                "crag_retry", detail=", ".join(sorted(unsearched))
            )
            t = time.time()

            extra = self._retrieve_from(question, sources=unsearched)

            retry_retrieval_ms = int((time.time() - t) * 1000)
            t = time.time()

            # Grade the combined set rather than only the new records:
            # the ranking is only meaningful across everything that is
            # competing for a place in the context.
            combined = list(retrieved)
            seen = {r.id for r in combined}

            for record in extra:
                if record.id not in seen:
                    seen.add(record.id)
                    combined.append(record)

            retry_grade = self.grader.grade(question, combined)

            log_stage(
                stage="crag_retry",
                session_id=session_id,
                turn_id=turn_id,
                data={
                    "reason": "no relevant record in the routed sources",
                    "widened_to": sorted(unsearched),
                    "retrieval_ms": retry_retrieval_ms,
                    "retrieved": len(combined),
                    "kept": [r.id for r in retry_grade.kept],
                    "verdict": retry_grade.verdict,
                    "warnings": retry_grade.warnings,
                },
                duration_ms=(time.time() - t) * 1000,
                llm=getattr(self.grader.llm_client, "last_metrics", None),
            )

            if not retry_grade.needs_retry:
                yield ("kept", (retry_grade.kept, False))
                return

            # The retry did not find anything relevant either, but it may
            # still have surfaced better partial matches than the first
            # pass. Keep whichever set is non-empty, preferring the
            # wider one, and be honest with the generator either way.
            yield ("kept", ((retry_grade.kept or grade.kept), True))
            return

        # Every source was already searched, so there is nothing left to
        # widen to. Say so rather than retrying the same query.
        log_stage(
            stage="crag_retry",
            session_id=session_id,
            turn_id=turn_id,
            data={
                "reason": "all sources already searched, no retry possible",
                "kept": [r.id for r in grade.kept],
            },
        )

        yield ("kept", (grade.kept, True))

    def _retrieve_from(self, question: str, sources: list) -> list:

        all_records = []
        seen_ids = set()

        for source in sources:
            retriever = self.retrievers[source]

            records = retriever.retrieve(
                question,
                top_k=MAX_RESULTS_PER_SOURCE,
            )

            for record in records:
                if record.id not in seen_ids:
                    seen_ids.add(record.id)
                    all_records.append(record)

        return all_records
