"""
retrievers/cards_retriever.py

WHAT THIS FILE DOES
--------------------
The keyword retriever for the "cards" source. All the real logic
lives in KeywordRetriever (see base_retriever.py) -- this file just
tells it to load its data from CardsLoader.

INPUTS  : query (str) -- the user's question or a sub-query from the router
OUTPUTS : list[Record] -- the most relevant card records, best first
"""

from retrievers.base_retriever import KeywordRetriever
from loaders.cards_loader import CardsLoader


class CardsRetriever(KeywordRetriever):
    def __init__(self):
        super().__init__(loader=CardsLoader())
