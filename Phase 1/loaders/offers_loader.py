"""
loaders/offers_loader.py

WHAT THIS FILE DOES
--------------------
Reads the two sheets of the offers Excel file and converts each row
into a Record:
  1. "Nov. Installment Arb - Eng"   -> merchant installment plans
  2. "Nov. Discount Offer,Arb-Eng"  -> merchant discounts

WHAT THE RAW SHEETS LOOK LIKE
------------------------------
Both are FLAT tables (one row = one merchant offer), unlike the wide
cards sheet. Columns:
  Installment sheet : merchant name | installment duration (Arabic) |
                       terms (Arabic) | terms (English) | category
  Discount sheet     : merchant name | terms (Arabic) | terms (English) | category

We keep both the Arabic and English terms in metadata (some merchants
in this bank's data are more clearly described in one language than
the other) but build display_text/search_text primarily from the
English column plus merchant name and category.

INPUTS  : the Excel file at config.settings.OFFERS_XLSX_PATH
OUTPUTS : list[Record], one per merchant offer row (installment or discount)
"""

import openpyxl

from loaders.base_loader import BaseLoader
from loaders.record import Record
from config.settings import OFFERS_XLSX_PATH

INSTALLMENT_SHEET = "Nov. Installment Arb - Eng"
DISCOUNT_SHEET = "Nov. Discount Offer,Arb-Eng "  # note: trailing space in real sheet name


class OffersLoader(BaseLoader):
    def load(self) -> list[Record]:
        wb = openpyxl.load_workbook(OFFERS_XLSX_PATH, data_only=True)
        records: list[Record] = []
        records.extend(self._load_installment_sheet(wb))
        records.extend(self._load_discount_sheet(wb))
        return records

    def _load_installment_sheet(self, wb) -> list[Record]:
        # Sheet name has slight naming inconsistencies in the source file,
        # so we look it up case/space-insensitively instead of hardcoding
        # the exact match, to make this loader resilient to small typos
        # in future versions of the same file.
        ws = self._find_sheet(wb, "installment")
        if ws is None:
            return []

        records = []
        for row_idx in range(2, ws.max_row + 1):  # row 1 = header
            merchant = ws.cell(row_idx, 1).value
            duration_ar = ws.cell(row_idx, 2).value
            terms_ar = ws.cell(row_idx, 3).value
            terms_en = ws.cell(row_idx, 4).value
            category = ws.cell(row_idx, 5).value

            if merchant in (None, ""):
                continue

            merchant = str(merchant).strip()
            terms_en = (str(terms_en).strip() if terms_en else "")
            category = (str(category).strip() if category else "")

            display_text = (
                f"Merchant: {merchant}\n"
                f"Offer type: Installment\n"
                f"Category: {category}\n"
                f"Terms (English): {terms_en}\n"
                f"Installment duration (as listed, Arabic): {duration_ar or ''}"
            )
            search_text = " ".join(
                str(x) for x in [merchant, category, terms_en, duration_ar, terms_ar] if x
            ).lower()

            records.append(
                Record(
                    id=f"offer_installment_{row_idx}_{merchant}",
                    source="offers_installment",
                    title=f"{merchant} - Installment Offer",
                    search_text=search_text,
                    display_text=display_text,
                    metadata={
                        "merchant": merchant,
                        "category": category,
                        "terms_english": terms_en,
                        "terms_arabic": terms_ar,
                        "duration_arabic": duration_ar,
                    },
                )
            )
        return records

    def _load_discount_sheet(self, wb) -> list[Record]:
        ws = self._find_sheet(wb, "discount")
        if ws is None:
            return []

        records = []
        for row_idx in range(2, ws.max_row + 1):  # row 1 = header
            merchant = ws.cell(row_idx, 1).value
            terms_ar = ws.cell(row_idx, 2).value
            terms_en = ws.cell(row_idx, 3).value
            category = ws.cell(row_idx, 4).value

            if merchant in (None, ""):
                continue

            merchant = str(merchant).strip()
            terms_en = (str(terms_en).strip() if terms_en else "")
            category = (str(category).strip() if category else "")

            display_text = (
                f"Merchant: {merchant}\n"
                f"Offer type: Discount\n"
                f"Category: {category}\n"
                f"Terms (English): {terms_en}"
            )
            search_text = " ".join(
                str(x) for x in [merchant, category, terms_en, terms_ar] if x
            ).lower()

            records.append(
                Record(
                    id=f"offer_discount_{row_idx}_{merchant}",
                    source="offers_discount",
                    title=f"{merchant} - Discount Offer",
                    search_text=search_text,
                    display_text=display_text,
                    metadata={
                        "merchant": merchant,
                        "category": category,
                        "terms_english": terms_en,
                        "terms_arabic": terms_ar,
                    },
                )
            )
        return records

    @staticmethod
    def _find_sheet(wb, keyword: str):
        """Find a sheet whose name contains `keyword`, ignoring case/whitespace."""
        keyword = keyword.lower()
        for name in wb.sheetnames:
            if keyword in name.lower():
                return wb[name]
        return None
