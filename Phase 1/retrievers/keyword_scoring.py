"""
retrievers/keyword_scoring.py

WHAT THIS FILE DOES
--------------------
A small, dependency-free scoring function: given a user query and a
Record's search_text, returns a relevance score (higher = more
relevant). This is Phase 1's ENTIRE retrieval "algorithm" -- no
embeddings, no vector database, just counting overlapping words.

WHY IT EXISTS AS ITS OWN FILE
------------------------------
Every source-specific retriever (cards, offers, campaigns) needs the
SAME scoring logic -- only the data differs. Putting it here once
means:
  1. All retrievers score consistently.
  2. If you later want to upgrade Phase 1's matching (e.g. add fuzzy
     matching, or handle Arabic stemming), you change ONE function
     and every retriever benefits immediately.

HOW IT WORKS (intentionally simple, on purpose)
------------------------------------------------
1. Lowercase and split the query into individual words ("tokens").
2. For each Record, count how many of those query words appear
   anywhere in its search_text.
3. That count IS the score. Records with score 0 are considered
   irrelevant and dropped.

This is a bag-of-words / keyword-overlap approach -- the simplest
possible form of "search" without any ML. It's intentionally naive:
Phase 1's goal is to prove the end-to-end loop works, not to have
great retrieval quality yet. Phase 2's re-ranker and dynamic router
will make this smarter.

INPUTS  : query (str), a Record's search_text (str)
OUTPUTS : an integer score
"""

import re

# Very short, common English words add noise to keyword matching
# (e.g. matching on "the", "is", "a" would make almost every record
# score > 0). We filter these out before scoring.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which",
    "who", "whom", "this", "that", "of", "for", "on", "in", "to",
    "do", "does", "can", "i", "me", "my", "you", "your", "it", "its",
    "and", "or", "with", "about", "have", "has", "there", "any",
}


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens, dropping punctuation and stopwords."""
    words = re.findall(r"[a-zA-Z0-9%]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def score(query: str, search_text: str) -> int:
    """
    Return the number of query tokens that appear in search_text.
    A score of 0 means "no overlap -- not relevant".
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0
    matches = sum(1 for token in query_tokens if token in search_text)
    return matches
