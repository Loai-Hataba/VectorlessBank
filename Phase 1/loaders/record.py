"""
loaders/record.py

WHAT THIS FILE DOES
--------------------
Defines ONE simple data shape, called `Record`, that every loader in
this project must produce and every retriever must consume.

WHY IT EXISTS
-------------
We have 3 very different raw files (a wide Excel sheet, a flat Excel
sheet, a nested JSON file). If every loader returned data in its own
custom shape, the rest of the pipeline (retrievers, context builder)
would need special-case code for each source. Instead, every loader
translates its messy source into this ONE common shape. After that
point, nothing downstream needs to know or care whether a Record
originally came from Excel or JSON.

This is the single most important design decision for modularity:
it's the "plug" that lets us swap any loader or retriever without
touching the rest of the system.

FIELDS
------
id            : str  - unique identifier for this record within its source
source        : str  - which source this came from, e.g. "cards", "offers_installment"
title         : str  - short human-readable name (card name / merchant name / campaign name)
search_text   : str  - a lowercase blob of text used for simple keyword matching in Phase 1
display_text  : str  - a clean, readable version of the full record, used as LLM context
metadata      : dict - the original structured fields, kept in case Phase 2 needs to
                        filter/sort/rerank by specific fields (e.g. card segment, dates)

INPUTS  : none (this is just a data class)
OUTPUTS : the Record type, imported by every loader and retriever
"""

from dataclasses import dataclass, field


@dataclass
class Record:
    id: str
    source: str
    title: str
    search_text: str
    display_text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a plain dict, e.g. for caching to JSON or for logging."""
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "search_text": self.search_text,
            "display_text": self.display_text,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict) -> "Record":
        """Rebuild a Record from a plain dict, e.g. when loading a cache file."""
        return Record(
            id=d["id"],
            source=d["source"],
            title=d["title"],
            search_text=d["search_text"],
            display_text=d["display_text"],
            metadata=d.get("metadata", {}),
        )
