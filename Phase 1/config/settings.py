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

DATAFRAME_INDEX_DIR = PROJECT_ROOT / "data" / "dataframes"
DATAFRAME_SOURCES = frozenset({"cards", "offers"})
# ============================================================
# LLM PROVIDER
# ============================================================
# Which runtime serves every role. "ollama" runs locally; "gemini"
# calls Google's hosted API.
#
# WHY THIS SWITCH EXISTS
# ----------------------
# This project was built local-first and the whole OLLAMA section
# below is tuned for a 6 GB laptop GPU. That tuning works -- it took
# one question from 381s to 125s -- but it is tuning around a wall:
# llama3.1:8b plus a 32k context does not fit in 6 GB, so nearly half
# the model runs on the CPU and prompt reading crawls at ~700 tok/s.
#
# Measured on the real offers traversal (28,229 tokens):
#
#     llama3.1:8b, local, 45% on CPU ....... 62.5 s
#     gemini-flash-lite-latest, hosted ......  2.1 s
#
# Thirty times faster, and MORE accurate on that call: the hosted
# model returned only the specific node the question asked about,
# while the local one also grabbed the broad parent -- the exact
# imprecision behind the Carrefour retrieval bug.
#
# THE TRADE, STATED PLAINLY
# -------------------------
# "gemini" sends card pricing, campaign terms and offer data to a
# third party on every question. That is a data-governance decision,
# not a performance one. Set LLM_PROVIDER=ollama to keep everything on
# the machine; nothing else needs to change, because every role goes
# through LLMClient and both backends implement the same methods.
# DEFAULT IS LOCAL, DELIBERATELY.
#
# "gemini" was measured and works -- one question went from 125s to
# 14s, a ~9x speedup. It is not the default anyway, because the cost
# is not latency: every question ships card pricing, campaign terms
# and offer data to a third party. That is a data-governance decision
# about bank data, and it is not one to inherit from a default.
#
# Switch with LLM_PROVIDER=gemini in the environment if and when that
# decision is made deliberately, and note it also needs GEMINI_API_KEY.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

VALID_LLM_PROVIDERS = frozenset({"ollama", "gemini"})

# ------------------------------------------------------------
# Gemini
# ------------------------------------------------------------
# Read from the environment only. Never hardcode a key here -- this
# file is in git.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
)

# Per role, for the same reason the Ollama models are per role: the
# roles do genuinely different work. "-latest" aliases are used rather
# than pinned versions because a pinned one can be retired out from
# under you -- gemini-2.5-flash already returns 404 "no longer
# available to new users" on a new key.
#
# flash-lite for the roles that classify or navigate, flash for the
# ones that write prose or judge whether prose is grounded.
# MEASURED, NOT ASSUMED: gemini-flash-latest read-timed-out at 120s on
# a TRIVIAL prompt from this machine, twice, while flash-lite answered
# the same prompt in 1.1s. Until that is understood, both names point
# at the model that actually responds. Change this one line to try the
# bigger model again.
GEMINI_FLASH = os.environ.get("GEMINI_FLASH_MODEL", "gemini-flash-lite-latest")
GEMINI_FLASH_LITE = os.environ.get(
    "GEMINI_FLASH_LITE_MODEL", "gemini-flash-lite-latest"
)

GEMINI_MODEL_BY_ROLE = {
    "indexer": GEMINI_FLASH_LITE,
    "traverser": GEMINI_FLASH_LITE,
    "generator": GEMINI_FLASH,
}

# Gemini 2.5 models think before answering, and those thinking tokens
# are billed against maxOutputTokens -- so a tight cap can be consumed
# entirely by thinking, returning an EMPTY string that every parser
# here would read as a failure.
#
# The obvious fix, thinkingBudget=0, DOES NOT WORK on flash-lite:
#
#     no thinkingConfig      1.11s  ok, 12 output tokens
#     thinkingBudget = 0     1.17s  HTTP 400 INVALID_ARGUMENT
#     thinkingBudget = -1    1.70s  ok, but 140 thinking tokens
#
# That model cannot have thinking switched off, and asking costs a
# 400. So the default is to send no thinkingConfig at all, which is
# both the fastest and the only universally accepted option. Set this
# to an integer only for a model known to accept it; leave it empty to
# omit the field.
GEMINI_THINKING_BUDGET = os.environ.get("GEMINI_THINKING_BUDGET", "").strip()

# The local num_predict ceilings exist to stop a small local model
# rambling. A hosted model does not have that failure mode, and a cap
# that is too tight here truncates mid-JSON, so the floor is generous.
GEMINI_MIN_OUTPUT_TOKENS = int(
    os.environ.get("GEMINI_MIN_OUTPUT_TOKENS", "2048")
)

GEMINI_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "120"))

# Hosted APIs rate-limit and occasionally 503. One quiet retry keeps a
# transient blip from surfacing as a failed turn.
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "2"))


# ============================================================
# LLM / OLLAMA CONFIGURATION
# ============================================================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
# THREE ROLES, NOT NINE
# ---------------------
# There were nine -- indexer, traverser, generator, router,
# summarizer, guardrail_input, guardrail_output, grader, querier --
# each with its own model, temperature, context size and reply cap.
# Thirty-six constants describing what turned out to be three
# genuinely different jobs, all pointed at the same model anyway.
#
# The three that remain are the three that actually differ in kind:
#
#   indexer    builds the tree index, offline, once
#   traverser  picks from a known set of options -- which nodes,
#              which sources, which filters. Small fixed JSON out.
#   generator  reads and writes natural language, or judges whether
#              some natural language is grounded
#
# Everything else was one of those three wearing a different name.
# router and querier are selection, so they are traverser work.
# summarizer, grader and both guardrails read or judge prose, so they
# are generator work. LLMClient.ROLE_ALIASES maps the old names onto
# the new ones, so existing call sites and the other worktrees keep
# running rather than raising on an unknown role.
#
# WHAT THIS COSTS
# ---------------
# Per-role reply caps go with the per-role names: guardrail_input had
# a 32-token ceiling and now inherits the generator's unbounded one.
# That ceiling was worth a second or two against a rambling local
# model and is worth nothing against a hosted one. Where a caller
# still needs a specific knob it passes it directly --
# LLMClient(role="generator", temperature=0.0) -- which is how the
# grader and both guardrails keep their determinism below.
# WHY qwen3.5:4b AND NOT llama3.1:8b
# ---------------------------------
# The bottleneck on this machine is VRAM, not model quality. A 6 GB
# card cannot hold 4.9 GB of llama weights plus a 32k KV cache, so
# ~45% of that model runs on the CPU. A 3.4 GB model leaves room.
#
# Measured on the real offers traversal, four unseen questions each,
# both models warm (mean wall clock):
#
#     llama3.1:8b               57.9s   45% CPU / 55% GPU   7.6 GB
#     qwen3.5:4b (think off)    26.7s   28% CPU / 72% GPU   4.4 GB
#     qwen2.5:14b              162.0s   66% CPU / 34% GPU
#
# 2.2x faster on traversal, same node selected on all four, and on one
# question it found a second relevant node llama missed. The 14b is far
# worse for the obvious reason: 9 GB does not fit in 6 GB.
#
# SO WHY IS LLAMA STILL THE DEFAULT
# ---------------------------------
# Because traversal is not the whole pipeline, and the rest of it did
# not hold up. Across three real end-to-end questions qwen3.5:4b
# produced one FALSE BLOCK -- "what is a credit card grace period" was
# refused with the internals-disclosure reply, flags=
# ['discloses_internals'] -- and in a targeted output-guardrail test it
# also flagged a correct general-knowledge answer as fabrication.
# llama3.1:8b did neither, on the same inputs.
#
# Refusing a real customer question is the failure this codebase has
# already been bitten by once (see input_guardrail.py) and it is worse
# than being slow. Both models now score 27/27 on
# evaluation/eval_input_guardrail.py, so the guardrail improvement that
# arrived alongside this was the PROMPT, not the model.
#
# qwen3.5:4b remains a good trade if you want the speed and can accept
# that risk -- it is one environment variable away, and the think flag
# and fence stripping below exist to make it work. Re-run
# eval_input_guardrail.py and a few real questions before trusting it.
#
# BEWARE ONE MEASUREMENT TRAP: asking the same question twice hits
# Ollama's prompt cache and returns in ~5s. That is real for repeat
# questions but says nothing about the first ask. Always benchmark
# with unseen questions.
#
# Env-readable so switching back is one variable, not an edit.
OLLAMA_INDEXER_MODEL = os.environ.get("OLLAMA_INDEXER_MODEL", "llama3.1:8b")
OLLAMA_TRAVERSER_MODEL = os.environ.get("OLLAMA_TRAVERSER_MODEL", "llama3.1:8b")
OLLAMA_GENERATOR_MODEL = os.environ.get("OLLAMA_GENERATOR_MODEL", "llama3.1:8b")

# Thinking models spend output tokens reasoning BEFORE they answer,
# and those tokens count against num_predict. qwen3.5:4b with a 256
# token cap spends all 256 thinking and returns an EMPTY string --
# which every parser here reads as a failure, and which fails open in
# the guardrails. Switching thinking off is what makes this model
# usable, and it is also most of the speed win.
#
# Sent as a top-level request field. Verified harmless on models that
# do not think: llama3.1:8b accepts think=false and ignores it.
# Set to "" to omit the field entirely.
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "false").strip().lower()


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------
OLLAMA_INDEXER_TEMPERATURE = 0.0
OLLAMA_TRAVERSER_TEMPERATURE = 0.0
OLLAMA_GENERATOR_TEMPERATURE = 0.2

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
# THE SECOND REASON THIS SECTION MATTERS: RELOAD COST
# ---------------------------------------------------
# Everything above is still true -- too small a window silently
# truncates and produces the bugs described. But a SECOND cost was
# measured later, and it pulls in the opposite direction.
#
# Ollama keys its loaded model runner on (model + num_ctx). Two roles
# that share a model but ask for different num_ctx values are, to
# Ollama, two different runners -- and this machine's 6 GB card only
# has room for one. So every role switch EVICTS the model and reloads
# ~5 GB from scratch. Measured on this machine, llama3.1:8b:
#
#     same num_ctx twice  ->  2.9s wall, 0.4s of it loading
#     num_ctx changed     -> 19.7s wall, 16.1s of it loading
#     num_ctx changed back-> 17.1s wall, 14.5s of it loading
#
# One turn used to walk 2048 -> 4096 -> 4096 -> 32768 -> 16384 ->
# 32768 -> 32768: five reloads on a clean turn, seven when CRAG
# widens. At ~15s each that was 75-105 SECONDS PER QUESTION spent
# loading and doing no work at all.
#
# So the per-role numbers below now all default to ONE shared value.
# The per-role names and their env overrides are deliberately kept:
# the reasoning above is still the reasoning that sets the FLOOR, and
# a future deployment on a bigger card can raise any single role again
# by exporting its variable. What changed is only the default, and the
# rule it now follows: pick the largest window any role genuinely
# needs, and give it to all of them, because a shared window is free
# and a switched window costs fifteen seconds.
#
# The largest genuine need is the traverser at ~26,560 tokens for the
# offers tree, so the shared value is its old 32768. Lowering this
# below 32768 will silently truncate the offers traversal and bring
# back the "every question retrieves the same records" bug.
#
# MEMORY: 32768 tokens of KV cache costs ~4 GB at f16 on this model
# (32 layers x 8 KV heads x 128 dim x 2 x 2 bytes = 128 KB/token),
# which does NOT fit alongside 4.9 GB of weights on a 6 GB card. Run
# the Ollama SERVER with OLLAMA_FLASH_ATTENTION=1 and
# OLLAMA_KV_CACHE_TYPE=q8_0 to halve that to ~2 GB. Those are server
# environment variables, not request options -- they must be set
# before `ollama serve` starts, and this file cannot set them.
OLLAMA_SHARED_NUM_CTX = int(os.environ.get("OLLAMA_SHARED_NUM_CTX", "32768"))

_SHARED = str(OLLAMA_SHARED_NUM_CTX)

OLLAMA_INDEXER_NUM_CTX = int(os.environ.get("OLLAMA_INDEXER_NUM_CTX", _SHARED))
# The largest consumer: an entire tree index in a single prompt. The
# offers tree needs ~26.5k tokens today, so this leaves headroom for
# the tree to grow before silent truncation returns.
OLLAMA_TRAVERSER_NUM_CTX = int(os.environ.get("OLLAMA_TRAVERSER_NUM_CTX", _SHARED))
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
OLLAMA_GENERATOR_NUM_CTX = int(os.environ.get("OLLAMA_GENERATOR_NUM_CTX", _SHARED))



# ---- Phase 2 roles ----
#
# Sizes below are per role because the roles read genuinely different
# amounts. Where the two workloads disagreed on a number during the
# merge, the owner of the role won: whoever built a component measured
# what it actually sends, and copying a neighbouring role's number is
# how the original truncation bug happened.

# Router and summarizer see a short question, a source list and a small
# conversation window. Partner A sized and tested these.


# Reads one raw user message and classifies it. The smallest prompt in
# the pipeline.

# NOT 4096. The output guardrail reads the generated answer AND the
# whole context it must be checked against, so it needs the same room as
# the generator itself. At 4096 it sees a fraction of the evidence and
# reports the generator's legitimate, correctly-sourced claims as
# fabrication -- which is exactly what happened when its context was
# capped too low. See the note on GUARDRAIL_MAX_CONTEXT_CHARS.

# Sees every candidate record before filtering -- up to three sources'
# worth -- as compact GRADER_MAX_RECORD_CHARS renderings. Fifteen
# candidates at 1200 characters is already ~4,500 tokens before the
# prompt, so 8192 leaves too little headroom.

# Fallback for any role that has no explicit entry, so that adding a
# role to MODEL_BY_ROLE without adding one here degrades to a usable
# default instead of raising KeyError at construction time.
OLLAMA_DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_DEFAULT_NUM_CTX", _SHARED))

# ---------------------------------------------------------------------------
# LLM request timeout
# ---------------------------------------------------------------------------

OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS","300",))


# ---------------------------------------------------------------------------
# How long Ollama keeps the model resident between calls
# ---------------------------------------------------------------------------
# Ollama unloads an idle model after 5 minutes by default. With one
# shared num_ctx the model is loaded once and then serves every role,
# so the only thing that still evicts it is that idle timer -- which
# means the first question after a coffee break pays the full ~15s
# reload that the rest of this section exists to avoid.
#
# Sent per request rather than set on the server, so it travels with
# the code instead of depending on how Ollama happens to be launched.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "60m")


# ---------------------------------------------------------------------------
# Reply length caps (num_predict)
# ---------------------------------------------------------------------------
# Ollama has no reply-length limit unless num_predict is sent, so a
# role that should answer {"verdict": "safe"} in 8 tokens is free to
# emit hundreds. That is not hypothetical for small local models
# asked for JSON: one run away and the turn's latency doubles.
#
# These are CEILINGS, not targets -- generous enough that no correct
# reply is ever truncated. The classifier roles return a fixed small
# JSON shape and are capped tightly; the grader's reply grows with
# the number of records it judges; the generator writes prose to a
# customer and is left effectively unbounded.
#
# -1 means "no limit" to Ollama.
OLLAMA_INDEXER_NUM_PREDICT = int(os.environ.get("OLLAMA_INDEXER_NUM_PREDICT", "512"))
OLLAMA_TRAVERSER_NUM_PREDICT = int(os.environ.get("OLLAMA_TRAVERSER_NUM_PREDICT", "256"))
OLLAMA_GENERATOR_NUM_PREDICT = int(os.environ.get("OLLAMA_GENERATOR_NUM_PREDICT", "-1"))
# Not tight: this one returns the unsupported claims it found, so its
# reply is as long as the problems it saw. Truncating it would look
# exactly like "found nothing wrong", which is the one failure this
# role must never have.
# Scales with RERANK_TOP_K-ish candidate counts: one small judgement
# object per record, up to ~15 records.
OLLAMA_DEFAULT_NUM_PREDICT = int(os.environ.get("OLLAMA_DEFAULT_NUM_PREDICT", "-1"))


# ---------------------------------------------------------------------------
# LLM Querying (filters) using python
#---------------------------------------------------------------------------
VALID_FILTER_OPS = frozenset({"==", "!=", "contains", ">", "<", ">=", "<="})
QUERIER_MAX_SAMPLE_ROWS = 3


# ---------------------------------------------------------------------------
# PageIndex tree indexing
# ---------------------------------------------------------------------------

# Whether PageIndex should write an LLM-generated summary onto each node.
# "no" makes indexing fully deterministic and requires no LLM at all;
# "yes" gives the traversal LLM more to go on, at one call per node.
PAGEINDEX_ADD_NODE_SUMMARY = os.environ.get("PAGEINDEX_ADD_NODE_SUMMARY", "no",)

# PageIndex talks to models through LiteLLM, whose naming is
# "provider/model" -- so the local Ollama model needs an "ollama/" prefix.
PAGEINDEX_SUMMARY_MODEL = os.environ.get("PAGEINDEX_SUMMARY_MODEL",    f"ollama/{OLLAMA_INDEXER_MODEL}",)

# Nodes shorter than this many tokens are not sent to the LLM at all --
# PageIndex uses their own text as the summary.
PAGEINDEX_SUMMARY_TOKEN_THRESHOLD = int(os.environ.get("PAGEINDEX_SUMMARY_TOKEN_THRESHOLD","200",))

# Every node summary is shown to the traversal LLM, so an un-truncated
# summary of a 366-column card record would blow up the traversal prompt.
PAGEINDEX_MAX_SUMMARY_CHARS = int(os.environ.get("PAGEINDEX_MAX_SUMMARY_CHARS","400",))

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
# Flask app settings
# ---------------------------------------------------------------------------
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
