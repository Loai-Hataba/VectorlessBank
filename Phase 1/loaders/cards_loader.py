"""
loaders/cards_loader.py

WHAT THIS FILE DOES
--------------------
Reads the "Credit card" sheet of the product catalog Excel file and
converts it into a list of Record objects, one per credit card
product (e.g. "VISA INFINITE", "MASTERCARD ISLAMIC GOLD").

WHAT THE RAW SHEET LOOKS LIKE (so the parsing logic below makes sense)
------------------------------------------------------------------
- It is a WIDE table: each ROW is one card product, and there are
  ~366 COLUMNS, one per business attribute (min limit, max limit,
  fees, terms of use, benefits, etc). Many attribute names repeat
  across several columns (e.g. "Benefits" appears 40 times, because
  each benefit is listed as its own column) -- we disambiguate those
  with a counter, e.g. "Benefits #1", "Benefits #2".
- Row 2 holds the English column headers. Rows 1 and 3 hold extra
  formatting/Arabic-label rows we don't need. Rows 4-5 are a
  worked example / legend ("Description", "Description translate to
  arabic") -- not real data. Real product rows start at row 6.
- Column A = a "Serial"/grouping number, Column C = Product Code,
  Column D = Product Name (English).

INPUTS  : the Excel file at config.settings.CARDS_XLSX_PATH
OUTPUTS : list[Record], one per credit card product, where
            - metadata holds every non-empty {attribute: value} pair
            - display_text is a clean "Attribute: Value" listing for the LLM
            - search_text is a lowercase blob for Phase 1 keyword matching
"""

import openpyxl
from collections import Counter

from loaders.base_loader import BaseLoader
from loaders.record import Record
from config.settings import CARDS_XLSX_PATH

SHEET_NAME = "Credit card"
HEADER_ROW = 2
FIRST_DATA_ROW = 6  # rows 4-5 are a worked example/legend, not real products

PRODUCT_CODE_COL = 3   # column C
PRODUCT_NAME_COL = 4   # column D


class CardsLoader(BaseLoader):
    def load(self) -> list[Record]:
        wb = openpyxl.load_workbook(CARDS_XLSX_PATH, data_only=True)
        ws = wb[SHEET_NAME]

        headers = self._build_disambiguated_headers(ws)

        records: list[Record] = []
        for row_idx in range(FIRST_DATA_ROW, ws.max_row + 1):
            product_code = ws.cell(row_idx, PRODUCT_CODE_COL).value
            product_name = ws.cell(row_idx, PRODUCT_NAME_COL).value

            # Skip rows that aren't real products (blank rows, stray legend rows)
            if product_code in (None, "") or product_name in (None, ""):
                continue

            attributes = {}
            for col_idx, header in enumerate(headers, start=1):
                if header is None:
                    continue
                value = ws.cell(row_idx, col_idx).value
                if value in (None, ""):
                    continue
                attributes[header] = value

            display_text = self._build_display_text(product_name, attributes)
            search_text = self._build_search_text(product_name, attributes)

            records.append(
                Record(
                    id=f"card_{product_code}",
                    source="cards",
                    title=str(product_name).strip(),
                    search_text=search_text,
                    display_text=display_text,
                    metadata=attributes,
                )
            )

        return records

    @staticmethod
    def _build_disambiguated_headers(ws) -> list[str]:
        """
        Row 2 has the attribute name for every column, but many names repeat
        (e.g. "Benefits" x40). We append a counter so each header is unique,
        e.g. "Benefits #1", "Benefits #2", ... This keeps every column's data
        instead of silently overwriting earlier columns with the same name.
        """
        raw_headers = [ws.cell(HEADER_ROW, c).value for c in range(1, ws.max_column + 1)]
        seen = Counter()
        unique_headers = []
        for h in raw_headers:
            if h is None:
                unique_headers.append(None)
                continue
            h_clean = str(h).strip()
            seen[h_clean] += 1
            if seen[h_clean] > 1:
                unique_headers.append(f"{h_clean} #{seen[h_clean]}")
            else:
                unique_headers.append(h_clean)
        return unique_headers

    @staticmethod
    def _build_display_text(product_name, attributes: dict) -> str:
        lines = [f"Card: {product_name}"]
        for k, v in attributes.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _build_search_text(product_name, attributes: dict) -> str:
        parts = [str(product_name)]
        parts.extend(str(v) for v in attributes.values())
        return " ".join(parts).lower()
