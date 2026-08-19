"""
config/settings.py

WHAT THIS FILE DOES
--------------------
This is the ONE file that holds every "knob" you might want to turn:
which Ollama model to use, where the data files live, how many
retrieved records to show the LLM, etc.

WHY IT EXISTS
-------------
If these values were scattered across the codebase (a model name typed
inside llm_client.py, a file path typed inside cards_loader.py, ...),
changing your setup later would mean hunting through many files.
Instead, every other module imports FROM here and never hardcodes
these values itself. Want to switch from "llama3.1" to "qwen2.5"?
Change ONE line, here.

INPUTS  : none (this file only reads environment variables, with safe defaults)
OUTPUTS : constants that other modules import, e.g.
          from config.settings import OLLAMA_MODEL, DATA_DIR
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CARDS_XLSX_PATH = RAW_DATA_DIR / "20260306_Product_Catalog_new_version.xlsx"
OFFERS_XLSX_PATH = RAW_DATA_DIR / "Feb_2026_offers_-_Wave_1_-_Call_Center.xlsx"
CAMPAIGNS_JSON_PATH = RAW_DATA_DIR / "campaigns_clean.json"

# ============================================================
# LLM / OLLAMA CONFIGURATION
# ============================================================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
OLLAMA_INDEXER_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
OLLAMA_TRAVERSER_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
OLLAMA_GENERATOR_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"

# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------
OLLAMA_INDEXER_TEMPERATURE = 0.0
OLLAMA_TRAVERSER_TEMPERATURE = 0.0
OLLAMA_GENERATOR_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# LLM request timeout
# ---------------------------------------------------------------------------

OLLAMA_TIMEOUT_SECONDS = int(
    os.environ.get(
        "OLLAMA_TIMEOUT_SECONDS",
        "300",
    )
)

# ---------------------------------------------------------------------------
# LLM tree indexing
# ---------------------------------------------------------------------------

# Maximum number of records sent to the indexer in one LLM request.
#
# This is deliberately small because the local 1B model must understand
# the records, organize them, and generate structured JSON.
TREE_INDEX_BATCH_SIZE = int(
    os.environ.get(
        "TREE_INDEX_BATCH_SIZE",
        "10",
    )
)

# Number of times the indexer may retry after invalid output.
TREE_INDEX_MAX_RETRIES = int(
    os.environ.get(
        "TREE_INDEX_MAX_RETRIES",
        "0",
    )
)

# Maximum number of records a leaf node should directly reference.
#
# This prevents a generic category node from becoming a giant bucket.
TREE_INDEX_MAX_RECORDS_PER_LEAF = int(
    os.environ.get(
        "TREE_INDEX_MAX_RECORDS_PER_LEAF",
        "5",
    )
)

# ---------------------------------------------------------------------------
# Retrieval settings (Phase 1: simple keyword retrieval)
# ---------------------------------------------------------------------------
# Max records each retriever is allowed to return per query. Keeps the
# context we send to the LLM small and focused instead of dumping
# everything.
MAX_RESULTS_PER_SOURCE = int(os.environ.get("MAX_RESULTS_PER_SOURCE", "5"))

# ---------------------------------------------------------------------------
# Flask app settings
# ---------------------------------------------------------------------------
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
