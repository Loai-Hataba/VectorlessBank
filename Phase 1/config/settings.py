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
# PROJECT_ROOT = the vectorless_rag_bank/ folder itself, computed automatically
# so the code works no matter where you clone/copy the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CARDS_XLSX_PATH = RAW_DATA_DIR / "20260306_Product_Catalog_new_version.xlsx"
OFFERS_XLSX_PATH = RAW_DATA_DIR / "Feb_2026_offers_-_Wave_1_-_Call_Center.xlsx"
CAMPAIGNS_JSON_PATH = RAW_DATA_DIR / "campaigns_clean.json"

# ---------------------------------------------------------------------------
# Ollama (local LLM) settings
# ---------------------------------------------------------------------------
# Ollama runs a local HTTP server, by default at this address, exposing
# an OpenAI-incompatible but simple JSON API. We read these from
# environment variables so you can override them without editing code,
# e.g. `OLLAMA_MODEL=qwen2.5:7b-instruct python run.py`
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Pick any model you have pulled locally with `ollama pull <model>`.
# llama3.1:8b-instruct is a solid, widely-available default with decent
# instruction-following for a project this size. Swap freely.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# How long (seconds) to wait for a local generation before giving up.
# Local models on CPU can be slow, so this is generous.
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))

# Sampling temperature: lower = more deterministic/factual answers.
# For a banking Q&A bot grounded in retrieved facts, we want low creativity.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

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
