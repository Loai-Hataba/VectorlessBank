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

from retrievers.tree_retriever import TreeRetriever
from pipeline.context_builder import build_context
from pipeline.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from llm.llm_client import LLMClient
from loaders.cards_loader import CardsLoader
from loaders.offers_loader import OffersLoader
from loaders.campaigns_loader import CampaignsLoader
from config.settings import MAX_RESULTS_PER_SOURCE


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
        self.llm_client = LLMClient()

    def answer(self, question: str) -> dict:
        retrieved_records = self._retrieve_all(question)
        context_text = build_context(retrieved_records)
        user_prompt = build_user_prompt(question, context_text)

        answer_text = self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return {
            "answer": answer_text,
            "retrieved_records": retrieved_records,
            "context_text": context_text,
        }

    def _retrieve_all(self, question: str):

        all_records = []
        seen_ids = set()

        for source, retriever in self.retrievers.items():

            records = retriever.retrieve(
                question,
                top_k=MAX_RESULTS_PER_SOURCE,
            )

            for record in records:

                if record.id not in seen_ids:

                    seen_ids.add(record.id)
                    all_records.append(record)

        return all_records
