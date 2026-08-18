"""
retrievers/campaigns_retriever.py

WHAT THIS FILE DOES
--------------------
The keyword retriever for the "campaigns" source. All the real logic
lives in KeywordRetriever (see base_retriever.py) -- this file just
tells it to load its data from CampaignsLoader.

NOTE ON "active" CAMPAIGNS
----------------------------
Phase 1 does NOT filter by campaign_status or dates -- it only does
keyword matching, so an expired campaign could still be returned if
its words match the query. We deliberately leave that correctness
concern for Phase 2 (dynamic retrieval / CRAG), so Phase 1 stays
"vanilla" as intended. Campaign status IS included in each record's
display_text, so the LLM can still see and mention it if relevant.

INPUTS  : query (str)
OUTPUTS : list[Record] -- the most relevant campaign records, best first
"""

from retrievers.base_retriever import KeywordRetriever
from loaders.campaigns_loader import CampaignsLoader


class CampaignsRetriever(KeywordRetriever):
    def __init__(self):
        super().__init__(loader=CampaignsLoader())
