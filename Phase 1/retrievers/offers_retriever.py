"""
retrievers/offers_retriever.py

WHAT THIS FILE DOES
--------------------
The keyword retriever for the "offers" source (installment plans +
discounts). All the real logic lives in KeywordRetriever (see
base_retriever.py) -- this file just tells it to load its data from
OffersLoader.

INPUTS  : query (str)
OUTPUTS : list[Record] -- the most relevant offer records, best first
"""

from retrievers.base_retriever import KeywordRetriever
from loaders.offers_loader import OffersLoader


class OffersRetriever(KeywordRetriever):
    def __init__(self):
        super().__init__(loader=OffersLoader())
