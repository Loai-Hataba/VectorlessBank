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

# Markdown rendered from Records, fed to PageIndex. Also handy to open
# by hand when a generated tree looks wrong.
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"
LOGS_DIR = PROJECT_ROOT / "logs"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline.jsonl"

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
OLLAMA_INDEXER_MODEL   = "llama3.1:8b"
OLLAMA_TRAVERSER_MODEL = "llama3.1:8b"
OLLAMA_GENERATOR_MODEL = "llama3.1:8b"
OLLAMA_ROUTER_MODEL = "llama3.1:8b"
OLLAMA_SUMMARIZER_MODEL = "llama3.1:8b"
OLLAMA_GUARDRAIL_INPUT_MODEL = "llama3.1:8b"
OLLAMA_GUARDRAIL_OUTPUT_MODEL = "llama3.1:8b"
OLLAMA_GRADER_MODEL = "llama3.1:8b"

# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------
OLLAMA_INDEXER_TEMPERATURE = 0.0
OLLAMA_TRAVERSER_TEMPERATURE = 0.0
OLLAMA_GENERATOR_TEMPERATURE = 0.2
OLLAMA_ROUTER_TEMPERATURE = 0.0
OLLAMA_SUMMARIZER_TEMPERATURE = 0.0
OLLAMA_GUARDRAIL_INPUT_TEMPERATURE = 0.0
OLLAMA_GUARDRAIL_OUTPUT_TEMPERATURE = 0.0
OLLAMA_GRADER_TEMPERATURE = 0.0

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
# PageIndex tree indexing
# ---------------------------------------------------------------------------

# Whether PageIndex should write an LLM-generated summary onto each node.
# "no" makes indexing fully deterministic and requires no LLM at all;
# "yes" gives the traversal LLM more to go on, at one call per node.
PAGEINDEX_ADD_NODE_SUMMARY = os.environ.get(
    "PAGEINDEX_ADD_NODE_SUMMARY",
    "no",
)

# PageIndex talks to models through LiteLLM, whose naming is
# "provider/model" -- so the local Ollama model needs an "ollama/" prefix.
PAGEINDEX_SUMMARY_MODEL = os.environ.get(
    "PAGEINDEX_SUMMARY_MODEL",
    f"ollama/{OLLAMA_INDEXER_MODEL}",
)

# Nodes shorter than this many tokens are not sent to the LLM at all --
# PageIndex uses their own text as the summary.
PAGEINDEX_SUMMARY_TOKEN_THRESHOLD = int(
    os.environ.get(
        "PAGEINDEX_SUMMARY_TOKEN_THRESHOLD",
        "200",
    )
)

# Every node summary is shown to the traversal LLM, so an un-truncated
# summary of a 366-column card record would blow up the traversal prompt.
PAGEINDEX_MAX_SUMMARY_CHARS = int(
    os.environ.get(
        "PAGEINDEX_MAX_SUMMARY_CHARS",
        "400",
    )
)

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
# Max records each retriever is allowed to return per query. Keeps the
# context we send to the LLM small and focused instead of dumping
# everything.
MAX_RESULTS_PER_SOURCE = int(os.environ.get("MAX_RESULTS_PER_SOURCE", "5"))
VALID_SOURCES = frozenset({"cards", "offers", "campaigns"})

# how many turns of raw user input to keep in memory.
MEMORY_RAW_TURNS_KEPT = int(os.environ.get("MEMORY_RAW_TURNS_KEPT", "3"))

# ---------------------------------------------------------------------------
# Flask app settings
# ---------------------------------------------------------------------------
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
