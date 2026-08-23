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

# ------------------------------------------------------------
# Context window (num_ctx)
# ------------------------------------------------------------
# WHY THIS EXISTS
# ---------------
# Ollama does NOT default to a model's full context window. Unless
# num_ctx is passed explicitly it uses a small default (4096 tokens for
# llama3.1:8b) and then SILENTLY TRUNCATES anything longer -- no error,
# no warning, just a shorter prompt than the one that was sent.
#
# This was a real, measured Phase 1 bug, not a theoretical risk. The
# traversal prompt for the offers tree is ~26,500 tokens; Ollama
# reported prompt_eval_count=4095, i.e. it discarded ~85% of the tree
# index. The truncation also cut the tail of the traversal system
# prompt, so the model stopped returning the required
# {"selected_node_ids": [...]} shape and emitted invented node IDs.
# The visible symptom was that EVERY question retrieved the same few
# records, because the model was never shown the part of the tree that
# made one question different from another.
#
# These are per-role for the same reason models and temperatures are:
# the roles have genuinely different input sizes. The traverser reads a
# whole flattened tree; the indexer sees one record at a time.
#
# COST NOTE: a bigger context window costs memory (KV cache), not
# accuracy. Keep each role no larger than the biggest prompt it really
# sends, and lower these on a machine that is short on VRAM.
OLLAMA_INDEXER_NUM_CTX = int(os.environ.get("OLLAMA_INDEXER_NUM_CTX","8192",))
# The largest consumer: an entire tree index in a single prompt. The
# offers tree needs ~26.5k tokens today, so this leaves headroom for
# the tree to grow before silent truncation returns.
OLLAMA_TRAVERSER_NUM_CTX = int(os.environ.get("OLLAMA_TRAVERSER_NUM_CTX","32768",))
# Retrieved records + conversation memory + the answer being written.
OLLAMA_GENERATOR_NUM_CTX = int(os.environ.get("OLLAMA_GENERATOR_NUM_CTX","16384",))
# Small, single-purpose classification/rewrite prompts -- nowhere near
# what indexer/traverser/generator need. Sized generously above their
# actual current inputs (a raw message, a contextualized question +
# source list, a short conversation window) rather than copied from
# another role's number, per the measured-truncation lesson above:
# guessing "should be fine" is exactly how the traverser bug happened.
OLLAMA_ROUTER_NUM_CTX = int(os.environ.get("OLLAMA_ROUTER_NUM_CTX", "4096"))
OLLAMA_SUMMARIZER_NUM_CTX = int(os.environ.get("OLLAMA_SUMMARIZER_NUM_CTX", "4096"))
OLLAMA_GUARDRAIL_INPUT_NUM_CTX = int(os.environ.get("OLLAMA_GUARDRAIL_INPUT_NUM_CTX", "2048"))
OLLAMA_GUARDRAIL_OUTPUT_NUM_CTX = int(os.environ.get("OLLAMA_GUARDRAIL_OUTPUT_NUM_CTX", "4096"))
OLLAMA_GRADER_NUM_CTX = int(os.environ.get("OLLAMA_GRADER_NUM_CTX", "8192"))

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
