"""
retrievers/base_retriever.py

WHAT THIS FILE DOES
--------------------
Defines two things:

1. `BaseRetriever` -- the contract every retriever follows: a
   `retrieve(query, top_k)` method that takes the user's question and
   returns the most relevant Record objects for that source.

2. `KeywordRetriever` -- a ready-to-use implementation of that contract
   using the simple keyword-overlap scoring from keyword_scoring.py.
   Every Phase 1 retriever (cards, offers, campaigns) is just a
   one-line subclass of this that says "load your records from this
   loader" -- see cards_retriever.py for an example.

WHY SPLIT IT THIS WAY
-----------------------
Cards, offers, and campaigns retrieval logic was IDENTICAL in Phase 1
(all keyword matching) -- only the underlying data differed. Rather
than copy-pasting the same scoring loop three times, we write it once
here. If in Phase 2 you want ONE specific source to retrieve
differently (say, campaigns should also filter by active status),
that source's file stops inheriting from KeywordRetriever and
implements BaseRetriever directly instead -- the rest of the app
doesn't need to change, because it only ever calls `.retrieve()`.

INPUTS  : none (abstract class) / a BaseLoader subclass (KeywordRetriever)
OUTPUTS : none (abstract class) -- subclasses return list[Record]
"""

from abc import ABC, abstractmethod
from loaders.record import Record
from loaders.base_loader import BaseLoader
from retrievers.keyword_scoring import score


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Record]:
        """
        Return the `top_k` Records most relevant to `query`.
        Must be implemented by every subclass.
        """
        raise NotImplementedError


class KeywordRetriever(BaseRetriever):
    """
    A reusable retriever that loads records once (via the given loader)
    and scores them against each query using plain keyword overlap.
    Subclass this and pass a loader instance to get a working retriever
    for a new source in one line -- see cards_retriever.py.
    """

    def __init__(self, loader: BaseLoader):
        self._records: list[Record] = loader.load()

    def retrieve(self, query: str, top_k: int) -> list[Record]:
        scored = [(score(query, r.search_text), r) for r in self._records]
        scored = [(s, r) for s, r in scored if s > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [r for _, r in scored[:top_k]]
