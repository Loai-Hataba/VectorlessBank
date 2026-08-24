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
#
# Raised from 16384 after a measured failure: a nine-record context came
# to ~20,955 tokens (a single card record is ~10,600 tokens of
# display_text), so the prompt was truncated at 16,383 and the model
# answered "the annual fee is EGP 4,500" -- a number that appears
# nowhere in that card's record. With the full prompt in view the same
# question is answered correctly and the fee is honestly reported as not
# stated. See also MAX_CONTEXT_TOTAL_CHARS below, which attacks the same
# problem from the other end.
OLLAMA_GENERATOR_NUM_CTX = int(
    os.environ.get(
        "OLLAMA_GENERATOR_NUM_CTX",
        "32768",
    )
)

# ---- Phase 2 roles ----
#
# Sizes below are per role because the roles read genuinely different
# amounts. Where the two workloads disagreed on a number during the
# merge, the owner of the role won: whoever built a component measured
# what it actually sends, and copying a neighbouring role's number is
# how the original truncation bug happened.

# Router and summarizer see a short question, a source list and a small
# conversation window. Partner A sized and tested these.
OLLAMA_ROUTER_NUM_CTX = int(
    os.environ.get("OLLAMA_ROUTER_NUM_CTX", "4096")
)

OLLAMA_SUMMARIZER_NUM_CTX = int(
    os.environ.get("OLLAMA_SUMMARIZER_NUM_CTX", "4096")
)

# Reads one raw user message and classifies it. The smallest prompt in
# the pipeline.
OLLAMA_GUARDRAIL_INPUT_NUM_CTX = int(
    os.environ.get("OLLAMA_GUARDRAIL_INPUT_NUM_CTX", "2048")
)

# NOT 4096. The output guardrail reads the generated answer AND the
# whole context it must be checked against, so it needs the same room as
# the generator itself. At 4096 it sees a fraction of the evidence and
# reports the generator's legitimate, correctly-sourced claims as
# fabrication -- which is exactly what happened when its context was
# capped too low. See the note on GUARDRAIL_MAX_CONTEXT_CHARS.
OLLAMA_GUARDRAIL_OUTPUT_NUM_CTX = int(
    os.environ.get("OLLAMA_GUARDRAIL_OUTPUT_NUM_CTX", "32768")
)

# Sees every candidate record before filtering -- up to three sources'
# worth -- as compact GRADER_MAX_RECORD_CHARS renderings. Fifteen
# candidates at 1200 characters is already ~4,500 tokens before the
# prompt, so 8192 leaves too little headroom.
OLLAMA_GRADER_NUM_CTX = int(
    os.environ.get("OLLAMA_GRADER_NUM_CTX", "16384")
)

# Fallback for any role that has no explicit entry, so that adding a
# role to MODEL_BY_ROLE without adding one here degrades to a usable
# default instead of raising KeyError at construction time.
OLLAMA_DEFAULT_NUM_CTX = int(
    os.environ.get(
        "OLLAMA_DEFAULT_NUM_CTX",
        "8192",
    )
)

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

# Total characters of record text allowed in one generator context,
# shared out between however many records were kept.
#
# WHY A SHARED BUDGET AND NOT A FIXED PER-RECORD CAP
# --------------------------------------------------
# The first version of this capped every record at a flat 6000
# characters. That bounded the prompt correctly but silently cost
# answers: a card record is ~42,000 characters, its fees sit around
# character 2,000 and its BENEFITS sit around character 12,000, so a
# 6000 cap kept the fees and threw the benefits away. The output
# guardrail then flagged perfectly good benefit claims as unsupported,
# because in the context it was shown they genuinely were.
#
# A shared budget adapts instead: when re-ranking keeps two records they
# get ~30,000 characters each and nothing important is lost, and when it
# keeps six they get ~10,000 each and the prompt is still bounded. The
# guarantee that matters -- the total never overflows the window -- is
# preserved either way, because it is now stated directly rather than
# inferred from a per-record guess.
#
# 60,000 characters is roughly 15,000 tokens, which sits comfortably in
# the generator's 32,768-token window alongside conversation memory and
# the answer being written.
MAX_CONTEXT_TOTAL_CHARS = int(
    os.environ.get("MAX_CONTEXT_TOTAL_CHARS", "60000")
)

# Floor on any single record's share, so that a large result set cannot
# shrink every record into uselessness. If the floor and the record
# count together exceed the budget, the budget gives way -- a prompt
# slightly over target is recoverable, whereas records cut to a few
# hundred characters answer nothing.
MIN_RECORD_CHARS = int(os.environ.get("MIN_RECORD_CHARS", "4000"))

# ---------------------------------------------------------------------------
# Corrective RAG / re-ranking (Phase 2)
# ---------------------------------------------------------------------------

# How many records survive re-ranking and reach the generator.
#
# This is the second bound on prompt size, alongside MAX_RECORD_CHARS:
# that one caps how big a record may be, this one caps how many there
# may be. Six is enough to answer questions that genuinely span several
# products while still leaving room for conversation memory.
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "6"))

# Characters of a record shown to the grader.
#
# Much smaller than MAX_RECORD_CHARS because the tasks differ: the
# generator must quote exact fees and terms, while the grader only has
# to recognise what a record is about. Sending full records here would
# recreate the context overflow this project has already been bitten by
# twice, and would make the grading call as slow as generation.
GRADER_MAX_RECORD_CHARS = int(
    os.environ.get("GRADER_MAX_RECORD_CHARS", "1200")
)

# How many records must be graded fully "relevant" before CRAG accepts
# the retrieval as Correct. Below this, the Incorrect branch fires and
# the search is widened once.
#
# 1 rather than 0 on purpose. "At least one partially relevant record"
# is too weak a bar for a local model, which will nearly always find
# something loosely on-topic -- asked about Carrefour it happily rates a
# different merchant's installment offer as useful. Requiring a record
# it was willing to call outright relevant is what makes the corrective
# branch fire when retrieval has actually missed.
CRAG_MIN_RELEVANT = int(os.environ.get("CRAG_MIN_RELEVANT", "1"))

# ---------------------------------------------------------------------------
# Output guardrail (Phase 2)
# ---------------------------------------------------------------------------

# Context shown to the output reviewer.
#
# THIS MUST NOT BE SMALLER THAN THE GENERATOR'S BUDGET.
#
# It was 8000 on the reasoning that a reviewer only has to check claims,
# not quote them, so it could work from less. That reasoning is wrong,
# and measurably so: a verifier shown less than the writer saw will
# report the writer's legitimate content as unsupported, every single
# time that content happens to sit past the verifier's cut.
#
# It produced exactly that. An answer citing the VISA INFINITE's lounge
# access, Booking.com discount and London Cab cashback -- all verbatim
# from the record, all around character 12,000 -- was flagged as
# fabrication, because the reviewer's 8000-character view stopped short
# of them. The check was not wrong about what it could see; it was
# shown the wrong thing.
#
# Tying it to the generator's own budget makes the guarantee structural
# rather than a number that has to be remembered twice.
GUARDRAIL_MAX_CONTEXT_CHARS = int(
    os.environ.get(
        "GUARDRAIL_MAX_CONTEXT_CHARS",
        str(MAX_CONTEXT_TOTAL_CHARS),
    )
)

# Whether a suspected fabrication blocks the answer or merely annotates
# it.
#
# Default off. The detector is a small local model and will raise false
# alarms, and withholding a correct answer is its own failure -- the
# customer gets nothing and cannot tell why. Annotating keeps the answer
# and flags the doubt. Turn this on where a deployment would rather say
# nothing than risk a wrong figure.
GUARDRAIL_BLOCK_FABRICATION = os.environ.get(
    "GUARDRAIL_BLOCK_FABRICATION", "false"
).lower() == "true"

# Appended when the answer strays into telling one customer what to do
# with their money. Softening rather than blocking is deliberate: see
# the reasoning in pipeline/output_guardrail.py.
ADVICE_DISCLAIMER = (
    "This is general product information, not personal financial "
    "advice. For guidance about your own situation, please speak to a "
    "Banque Misr representative."
)

BLOCKED_DISCLOSURE_REPLY = (
    "I can help with questions about our credit cards, merchant offers "
    "and campaigns, but I can't share how this assistant is configured "
    "internally. What would you like to know about our products?"
)

BLOCKED_FABRICATION_REPLY = (
    "I don't have enough verified information to answer that "
    "accurately, and I don't want to give you figures I can't confirm. "
    "Please check with a Banque Misr representative, or ask me "
    "something else about our cards, offers or campaigns."
)
VALID_SOURCES = frozenset({"cards", "offers", "campaigns"})

# how many turns of raw user input to keep in memory.
MEMORY_RAW_TURNS_KEPT = int(os.environ.get("MEMORY_RAW_TURNS_KEPT", "3"))

# ---------------------------------------------------------------------------
# Tier 2 evaluation (RAGAS)
# ---------------------------------------------------------------------------
#
# Read only by evaluation/eval_answers.py, never by the running app. They
# live here anyway because this file is the one place the project keeps
# knobs, and a judge model hardcoded inside an eval script is exactly the
# drift this file exists to prevent.
#
# NOTE ON ENVIRONMENTS: eval_answers.py runs with ragas installed, which
# the application deliberately does not have. It can still import this
# file because this file imports nothing but os and pathlib -- keep it
# that way, or Tier 2 stops being able to read its own configuration.

# Which model grades answer quality. Defaults to the generator's model so
# that a machine able to run the pipeline can also run the evaluation,
# but a judge SHOULD ideally be a different (larger) model than the one
# being judged -- a model grading its own output shares its blind spots.
# Point this at something stronger when hardware allows.
#
# THIS MODEL NEEDS num_ctx BAKED IN. READ THIS BEFORE TRUSTING A SCORE.
# ---------------------------------------------------------------------
# Every other role in this project passes num_ctx explicitly, because
# Ollama otherwise truncates prompts to ~4096 tokens without saying so
# (see the long note on OLLAMA_TRAVERSER_NUM_CTX). RAGAS cannot do that:
# it reaches Ollama through the OpenAI-compatible /v1 endpoint, which has
# no num_ctx field, and Ollama ignores it if you smuggle it through
# extra_body. Measured, not assumed.
#
# The result is not a lower score but a MISSING one -- the judge stops
# emitting parseable JSON and the sample is dropped. Cards suffer most,
# a card record being ~41,000 characters, so the metric quietly stops
# covering the source the fabrication bug came from.
#
# So the window has to live in the model itself:
#
#     printf 'FROM llama3.1:8b\nPARAMETER num_ctx 32768\n' > Modelfile.judge
#     ollama create llama3.1:8b-eval32k -f Modelfile.judge
#     set RAGAS_JUDGE_MODEL=llama3.1:8b-eval32k
#
# eval_answers.py checks for this at startup and warns if it is missing.
RAGAS_JUDGE_MODEL = os.environ.get(
    "RAGAS_JUDGE_MODEL",
    OLLAMA_GENERATOR_MODEL,
)

# Judging is a classification task, like every other non-generator role.
RAGAS_JUDGE_TEMPERATURE = float(
    os.environ.get("RAGAS_JUDGE_TEMPERATURE", "0.0")
)

# Each metric makes several calls per sample -- faithfulness alone
# extracts claims and then verifies each one -- so a per-call timeout has
# to allow for a slow local model without stalling a batch run forever.
RAGAS_JUDGE_TIMEOUT_SECONDS = int(
    os.environ.get("RAGAS_JUDGE_TIMEOUT_SECONDS", "300")
)

# How many samples RAGAS scores concurrently.
#
# Deliberately small. Ollama serves a local model essentially serially,
# so a high worker count does not finish sooner -- it just puts every
# request in the same queue while multiplying the chance of hitting the
# timeout above. This is the same lesson as PageIndex's unbounded
# summary concurrency, which is the one known way to make indexing hang.
RAGAS_MAX_WORKERS = int(os.environ.get("RAGAS_MAX_WORKERS", "2"))

# ---------------------------------------------------------------------------
# Flask app settings
# ---------------------------------------------------------------------------
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
