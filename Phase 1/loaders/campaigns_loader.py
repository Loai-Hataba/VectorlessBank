"""
loaders/campaigns_loader.py

WHAT THIS FILE DOES
--------------------
Reads campaigns_clean.json and converts each JSON campaign object
into a Record.

WHAT THE RAW FILE LOOKS LIKE
------------------------------
This is the easiest source to load because it's already structured
and even ships with a "query_tags" list -- keywords the data owner
already thought were useful for search. Each entry has fields like
campaign_name, campaign_status, campaign_start/end_date,
target_entity_value, eligible/excluded card segments, and a nested
"details" dict with the fine print (installment months, cashback
amounts, etc).

We keep the ENTIRE original JSON object in metadata (nothing is
thrown away), and build display_text/search_text from the
human-readable parts (name, dates, tags, details) rather than
source_text, since source_text is a long raw bilingual paragraph
that's redundant with the structured fields.

INPUTS  : the JSON file at config.settings.CAMPAIGNS_JSON_PATH
OUTPUTS : list[Record], one per campaign
"""

import json

from loaders.base_loader import BaseLoader
from loaders.record import Record
from config.settings import CAMPAIGNS_JSON_PATH


class CampaignsLoader(BaseLoader):
    def load(self) -> list[Record]:
        with open(CAMPAIGNS_JSON_PATH, "r", encoding="utf-8") as f:
            campaigns = json.load(f)

        records = []
        for c in campaigns:
            records.append(
                Record(
                    id=c.get("id", ""),
                    source="campaigns",
                    title=c.get("campaign_name", "Unnamed campaign"),
                    search_text=self._build_search_text(c),
                    display_text=self._build_display_text(c),
                    metadata=c,  # keep the full original record -- nothing dropped
                )
            )
        return records

    @staticmethod
    def _build_display_text(c: dict) -> str:
        details = c.get("details", {})
        details_lines = "\n".join(f"  - {k}: {v}" for k, v in details.items())

        return (
            f"Campaign: {c.get('campaign_name')}\n"
            f"Status: {c.get('campaign_status')}\n"
            f"Applies to: {c.get('target_entity_type')} = {c.get('target_entity_value')}\n"
            f"Valid: {c.get('campaign_start_date') or 'no start date listed'} "
            f"to {c.get('campaign_end_date') or 'no end date listed'}\n"
            f"Eligible card segments: {', '.join(c.get('eligible_card_segments', []))}\n"
            f"Excluded card segments: {', '.join(c.get('excluded_card_segments', []))}\n"
            f"Channels: {', '.join(c.get('channels', []))}\n"
            f"Details:\n{details_lines}"
        )

    @staticmethod
    def _build_search_text(c: dict) -> str:
        parts = [
            c.get("campaign_name", ""),
            c.get("target_entity_value", ""),
            c.get("campaign_category", ""),
            " ".join(c.get("query_tags", [])),
            " ".join(c.get("eligible_card_segments", [])),
        ]
        return " ".join(str(p) for p in parts if p).lower()
