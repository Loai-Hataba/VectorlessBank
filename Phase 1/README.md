# BM Retail Banking Assistant — Phase 1 (Vanilla Vectorless RAG)

A chatbot that answers questions about Banque Misr credit cards, merchant
offers, and campaigns — **without** embeddings or a vector database.

This is Phase 1: the simplest possible working version, built to make every
part of the pipeline easy to understand. Phase 2 will add dynamic retrieval
routing, corrective RAG, re-ranking, guardrails, memory, logging, and
evaluation on top of this same foundation — without rewriting anything here.

## How it works, in one paragraph

Your product data (credit cards, merchant offers, campaigns) lives in Excel
and JSON files, not free-text documents — so instead of chunking text and
embedding it into a vector database, each source is parsed into structured
`Record` objects. When a question comes in, three keyword-matching
retrievers (one per source) independently score their records against the
question and return their best matches. Whatever gets returned is formatted
into a context block and handed to a locally-running Ollama model, which is
told to answer using only that context. No vectors anywhere.

## Project structure

```
config/settings.py         All configuration (model name, paths, limits) — one place to tune everything
loaders/                   raw file -> normalized Record objects (one file per data source)
  record.py                   the shared Record shape every loader/retriever agrees on
  base_loader.py               abstract interface every loader implements
  cards_loader.py               parses the wide credit-card product Excel sheet
  offers_loader.py              parses the installment + discount offer Excel sheets
  campaigns_loader.py           parses the campaigns JSON file
retrievers/                 Record objects -> the ones relevant to a query (one file per source)
  base_retriever.py            abstract interface + shared KeywordRetriever implementation
  cards_retriever.py / offers_retriever.py / campaigns_retriever.py   one line each
  keyword_scoring.py            the actual (intentionally simple) matching algorithm
llm/llm_client.py           the ONLY file that talks to Ollama's HTTP API
pipeline/
  context_builder.py            retrieved Records -> one text block for the LLM
  prompt_templates.py           all prompt wording, kept separate from logic
  rag_pipeline.py                orchestrates: question -> retrieve -> context -> LLM -> answer
app/
  server.py                     Flask routes (GET /, POST /api/chat)
  templates/index.html, static/style.css, static/chat.js    vanilla JS chat UI, no build tooling
run.py                      start the app
```

**Why this layout matters:** every loader/retriever implements the same
tiny interface. `rag_pipeline.py` never imports anything specific to Excel
parsing or JSON parsing — it only ever calls `.retrieve(query, top_k)` on a
retriever. That means you can replace, delete, or rewrite any single piece
(swap the local model, rewrite how cards are matched, add a 4th source)
without touching the rest of the codebase.

## Setup

1. **Install Ollama** (if you haven't): https://ollama.com
2. **Pull a model** — the default is `llama3.1:8b`:
   ```bash
   ollama pull llama3.1:8b
   ```
   To use a different model, either edit `OLLAMA_MODEL` in
   `config/settings.py`, or set an environment variable:
   ```bash
   export OLLAMA_MODEL=qwen2.5:7b-instruct
   ```
3. **Start Ollama** (if it isn't already running in the background):
   ```bash
   ollama serve
   ```
4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the app:**
   ```bash
   python run.py
   ```
6. Open **http://127.0.0.1:5000** in your browser.

## What Phase 1 deliberately does NOT do yet

These are not bugs — they're the boundary of Phase 1 on purpose, so the
system stays simple enough to fully understand before we add complexity:

- **No source routing.** Every question queries all three retrievers, every
  time. A smarter "which source(s) does this question need?" decision is
  Phase 2's `agent/router.py`.
- **No correction / retry.** If retrieval comes back empty or weak, the
  pipeline doesn't try again with a rewritten query. That's Phase 2's
  Corrective RAG (CRAG) step.
- **No re-ranking.** Keyword-overlap scoring is crude — e.g. asking about
  "the Platinum card" can be out-scored by generic words like "card" or
  "limit" that appear in nearly every record. Phase 2 adds a re-ranking
  step (likely LLM-based) to fix this.
- **No guardrails beyond prompt instructions.** The system prompt tells the
  model to stick to retrieved facts, but nothing programmatically checks
  the output.
- **No conversation memory.** Every question is answered independently;
  there's no way to ask a follow-up like "and what about its annual fee?"
- **No logging or evaluation pipeline.** We can see what happened by
  reading code output, but nothing is written to a log file or measured
  systematically yet.
- **"General Information" source.** The project brief mentions a 4th data
  source alongside Cards/Offers/Campaigns. Per your call, general banking
  knowledge questions (e.g. "what is APR") are answered from the model's
  own training knowledge rather than a retrieved source — this is stated
  explicitly as rule #1 in `pipeline/prompt_templates.py`, not a silent
  fallback.

## Known data quirk worth knowing

The credit card Excel sheet has ~366 columns because many attributes (like
"Benefits") are split across many repeated columns rather than one cell.
`cards_loader.py` disambiguates repeated column headers by appending a
counter (`Benefits #1`, `Benefits #2`, ...) so no data is silently
overwritten.
