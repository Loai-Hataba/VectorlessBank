# VectorlessBank — Phase 2 Blueprint
**For: Claude Code**
**Authors: Loai partner A+ Omar partner B **
**Status: Phase 1 complete. This document is the spec for Phase 2.**

---

## 0. READ THIS FIRST — Rules for Claude Code

1. **Do not write any Phase 2 code until you have verified Phase 1 is working end-to-end.** See §1 "Pre-flight check." If Phase 1 is broken, stop and report what's broken — do not attempt to silently fix it and move on into Phase 2 work.
2. **This is a 2-person project. Do not implement both partners' workloads yourself in one pass.** §9 defines "Partner A" and "Partner B" workloads. After the pre-flight check passes and you've presented your understanding of the plan back to the user, **ask which partner's workload you're being asked to implement** (or which specific component within that workload) before writing any Phase 2 code. Only work on one partner's slice at a time unless explicitly told otherwise. If the person doesn't say which partner they are, ask — don't guess.
3. **Every new architectural decision in this document has a stated justification.** When you implement something, keep the reasoning intact in code comments / docstrings — this project is graded partly on the team's ability to explain *why* a component was designed the way it was, not just that it works.
4. **Model-agnosticism is a hard requirement.** Nothing in Phase 2 should hardcode a specific model name in logic. All roles must remain configurable through `config/settings.py`, exactly like Phase 1's `role="indexer"/"traverser"/"generator"` pattern. The team is actively testing different local models (currently moved from Llama-3.2-1B to Llama-3.1-8B) and will keep testing others — code must not assume any specific model's quirks.
5. Keep the existing frontend as-is. No UI changes in Phase 2 (per team decision) unless explicitly requested later.
6. Keep the existing simplicity constraints from Phase 1: vanilla JS frontend (no frameworks), Flask backend, local Ollama models, no vector DB, no embeddings — Phase 2 must not quietly introduce any of these.


---

## 1. Pre-flight check (do this before anything else)

Before touching Phase 2:

1. Confirm `ollama serve` is running.
2. Confirm all three tree indexes exist and are current: `data/indexes/campaigns_tree.json`, `data/indexes/offers_tree.json`, `data/indexes/cards_tree.json`. (Per the milestone doc, only `campaigns_tree.json` was known to exist and was built by the *old* indexer — rebuild all three with the current pipeline if needed: `python -m indexing.build_index campaigns`, then `offers`, then `cards`.)
3. Start `python run.py` and confirm the Flask server boots without error (RagPipeline constructs successfully, all three `TreeRetriever`s load and validate against their record sets).
4. Send at least 3 test questions through `POST /api/chat` covering: a cards-only question, an offers-only question, and a question that should pull from multiple sources. Confirm each returns a grounded `answer` plus non-empty `sources`.
5. Confirm the known Phase 1 gaps are understood (from the milestone doc) so Phase 2 doesn't get confused by them:
   - No router yet — every question currently queries all three sources.
   - Summary concurrency in PageIndex is unbounded (`PAGEINDEX_ADD_NODE_SUMMARY=no` is the escape hatch for large sources).
   - `search_text` and the old keyword retriever are vestigial/unreferenced — ignore them.
   - No `.gitignore` yet — low priority, can be added opportunistically.
6. check for old unused files regarding old manual index tree files that were used before but now you shouldn't find them being used in the main project, check for the files and ask before you delete them; make sure those files are regarding the old tree indexing only and make sure to ask before taking any action

**Report the results of this check back to the user before proceeding.** If anything fails, stop.

---

## 2. Project goal (unchanged, restated for context)

Build a dynamic **Agentic Vectorless RAG** chatbot for retail banking that:
- Retrieves from one or multiple structured data sources without embeddings or a vector database.
- Dynamically selects the appropriate retrieval method per query.
- Combines information across sources when required.
- Generates grounded answers, refusing to fabricate.
- Uses conversation history for follow-up questions.

Phase 1 built the indexing pipeline (Excel/JSON → categorized records → markdown → PageIndex tree → validated `TreeIndex` on disk) and a basic answering pipeline (fan-out to all three tree retrievers → merge → generate). Phase 2 adds everything the assignment requires on top of that foundation: **dynamic retrieval (router), corrective RAG, re-ranking, guardrails, conversation memory, pipeline logging, and an evaluation pipeline.**

---

## 3. Scope decision: the 4th data source

The requirements doc lists four sources: Cards, Offers, Campaigns, and **General Information**. Phase 1 only indexed the first three.

**Resolved scope:** "General Information" does **not** need its own indexed tree. Per team decision, general banking/financial knowledge (e.g. "what is APR," "what's a grace period") is answered by the generator LLM directly from its own training knowledge — this is already partially supported by the Phase 1 generator prompt, which carves out exactly this exception. Phase 2 formalizes it: the **Router** (§4) is responsible for recognizing when a question needs *no retrieval at all* and routing straight to generation. No new loader, no new tree, no new data source is being added in Phase 2.

---

## 4. Component 1 — Dynamic Retrieval (Router)

**Requirement it satisfies:** Dynamic Retrieval.

**Design:** A new LLM role, `role="router"`, following the exact pattern of `indexer`/`traverser`/`generator` in `llm/llm_client.py` and `config/settings.py`. One call per question (after query contextualization — see §7), `temp=0.0`, `format=json`.

**Input:** the (contextualized, standalone) user question.
**Output contract:**
```json
{
  "needs_retrieval": true,
  "sources": ["cards", "offers"],
  "reasoning": "short justification, logged but not shown to user"
}
```
- If `needs_retrieval` is `false`, `sources` is empty and the pipeline skips straight to generation with no context (the existing "nothing matched" prompt branch already handles empty context gracefully).
- `sources` is a subset of `{"cards", "offers", "campaigns"}`. Any value outside that set is discarded in Python (same "model proposes, Python disposes" discipline as the traverser's node-id validation) — never trust the router's raw output, whitelist-check it.
- This **replaces** the current `_retrieve_all` fan-out that queries all three `TreeRetriever`s unconditionally. After Phase 2, only the retrievers named in `sources` run.

**Why a router call and not a rule/keyword classifier:** the three sources overlap semantically (e.g. a question about "installment offers on a specific card" touches both Cards and Offers), and rigid keyword rules would be brittle and unmaintainable as the domain grows. An LLM classification call is cheap (short input/output, JSON-constrained, temp 0) and is the same pattern already proven reliable for `RecordCategorizer` in Phase 1.

**Why it must see the contextualized question, not the raw follow-up:** "what about its installment plan" is unroutable in isolation — it needs the referent resolved first. This is why Router depends on the Query Contextualizer (§7.1) running before it, not after.

---

## 5. Component 2 — Corrective RAG (CRAG)

**Requirement it satisfies:** Corrective RAG.

**Design rationale:** We adapt Yan et al.'s CRAG pattern (retrieval evaluator grades documents, then branches into Correct / Incorrect / Ambiguous handling) but **drop the "Incorrect → web search" branch entirely**, because pulling live web content into a retail-banking assistant is a compliance and groundedness risk that directly contradicts the project's "no fabrication, only grounded answers" requirement — an external, unvetted source is not meaningfully different from the model inventing facts. Instead, "Incorrect" is handled by widening the *internal* search, which fits naturally into the tree architecture:

**New role:** `role="grader"` (combined with reranking — see §6 for why they're merged into one call).

1. After the router-selected `TreeRetriever`s return their records, the grader evaluates each candidate record for relevance to the question and assigns a label: `relevant`, `partially_relevant`, or `irrelevant`.
2. **Branch logic (computed in Python, not by the model):**
   - **Correct** (enough `relevant`/`partially_relevant` records, e.g. ≥1 relevant record): proceed to reranking and generation using only the relevant/partially-relevant records — irrelevant ones are filtered out before they ever reach the generator's context window.
   - **Incorrect** (all records graded `irrelevant`, or zero records returned): trigger **one** corrective retry:
     - Re-run the traversal step with a relaxed selection strategy — instruct the traverser to also consider parent/category nodes (broadening recall instead of precision), **and/or**
     - Expand to sources the router did *not* originally select, as a fallback fan-out.
     - Re-grade the new results once.
   - **Still insufficient after the retry:** do not force an answer. Pass an explicit "insufficient grounded data" signal to the generator so it tells the user honestly that it doesn't have enough information, rather than inventing details. This is a direct anti-hallucination safeguard and ties into the Prompt Guardrails component (§8).
3. **Hard cap: exactly one retry.** This bounds worst-case latency (critical since local models are already the bottleneck) and avoids infinite correction loops. Log every retry with its reason (§10).

**Why grading is categorical (3-way) rather than a numeric relevance score:** categorical labels are far more reliable to elicit from a small/local LLM in a single low-temperature JSON call than well-calibrated numeric scores would be, and the downstream logic only needs a routing decision (retry or don't), not fine-grained ranking — that finer-grained ordering is exactly what the separate re-ranking step (§6) is for.

---

## 6. Component 3 — Re-ranking

**Requirement it satisfies:** Re-ranking.

**Design:** Because this system is explicitly vectorless (no embeddings, no cross-encoder model available), re-ranking cannot use similarity scores. It is done by the same LLM call as CRAG grading (§5), extended to also emit an ordinal rank/score per record:

```json
{
  "records": [
    {"id": "card_xyz", "relevance": "relevant", "rank": 1},
    {"id": "offer_abc", "relevance": "partially_relevant", "rank": 2}
  ]
}
```

**Why combine grading and reranking into one role instead of two separate LLM calls:** both tasks require the model to make the same underlying judgment — "how relevant is this record to the question" — just consumed two different ways (a categorical filter vs. an ordering). Making two separate calls would double LLM latency for no accuracy benefit. This is documented explicitly as **one implementation satisfying two separate assignment requirements** (Corrective RAG's grading step and Re-ranking), which is worth stating clearly in the final documentation/PPT so it doesn't read as a missing component.

After ranking, take only the top-K records (config value, suggest starting at `RERANK_TOP_K = 6`, tunable) into `build_context` — this also bounds prompt size, which matters for local-model latency and for staying well under context limits when conversation memory (§7) is also injected.

---

## 7. Component 4 — Conversation Memory

**Requirement it satisfies:** Conversation Memory.

**Design: rolling summarization**, per the team's stated preference. Per session (`session_id`):
- Keep the **last 2–3 raw turns** verbatim, for immediate, high-fidelity context.
- Keep a **rolling summary** of everything older than that, updated incrementally: after each turn, a new role (`role="summarizer"`) folds the latest exchange into the existing summary and returns an updated summary — never re-summarizing from scratch, so cost stays roughly constant per turn instead of growing with conversation length.

**Why rolling summarization over a fixed-window buffer or full-history-in-context:** a fixed window silently forgets anything older than N turns (bad for a banking assistant where a customer might reference something from earlier in a long conversation), while stuffing full history into every prompt grows unbounded and directly fights against the local-model latency/context constraints this team is already working within. Rolling summarization is the standard middle ground (same idea as LangChain's `ConversationSummaryBufferMemory`) and keeps memory cost flat regardless of conversation length — important given the team is deliberately testing multiple model sizes and wants prompt size to stay predictable across all of them.

**Storage:** in-memory dict keyed by `session_id` is sufficient for Phase 2 — consistent with the project's existing preference for simplicity and no added infrastructure. (Optional future enhancement, not required now: persist sessions to a local JSON/SQLite file so history survives a server restart — flag this as a "nice to have," don't build it unless asked.)

### 7.1 Query Contextualizer (part of this component)

Before the Router or any retriever runs, a lightweight step rewrites the raw user message into a standalone question using the last raw turns + rolling summary — e.g. "what about its installment plan" → "What is the installment plan for the Platinum Cashback card?" This can be the same `summarizer` role or a dedicated cheap call; either is acceptable, but document the choice. This directly enables the Router (§4) to work correctly on follow-ups, which the assignment explicitly requires ("use conversation history for follow-up questions").

---

## 8. Component 5 — Prompt Guardrails

**Requirement it satisfies:** Prompt Guardrails.

**Design: two lightweight checkpoints**, both LLM classification calls at `temp=0.0`, `format=json`, kept intentionally cheap since they run on every turn:

**Input guardrail** (runs before the Router, on the raw user message):
- Classifies the message as `safe`, `off_topic` (unrelated to banking — politics, unrelated general chit-chat, etc.), `injection_attempt` (message tries to override system instructions, e.g. "ignore previous instructions"), or `sensitive_request` (user is pasting or asking to store sensitive personal data like a full card number, CVV, password, national ID).
- On `off_topic` → polite redirect, no retrieval.
- On `injection_attempt` → refuse to comply with the embedded instruction, continue normally with the underlying (non-injected) part of the question if any exists, log the attempt.
- On `sensitive_request` → do not process or echo the sensitive data; ask the user not to share it in chat.

**Output guardrail** (runs after generation, before the answer is returned):
- Checks the generated answer against the retrieved context for **fabrication** (claims not supported by context or general banking knowledge — a second safety net on top of CRAG's "insufficient data" honesty path).
- Blocks **definitive personalized financial/investment/legal advice** (the assistant can explain products and general concepts, but should not tell a specific user what to do with their money) — flag and soften/append a disclaimer rather than hard-block, since over-blocking legitimate product-explanation questions would hurt usefulness.
- Refuses to reveal system prompts, internal architecture, or raw record dumps if asked.

**Why a cheap classification call rather than a bigger model or a rules engine:** these are high-frequency (run on essentially every turn) but low-reasoning-complexity tasks — closer to the `indexer`'s categorization job than to the `generator`'s free-text writing job. The same "small/fast model for classification, stronger model for generation" principle used elsewhere (see §11) applies here. This is proposed as a **standard baseline set**, explicitly extensible later per the team's note that more guardrails can be added.

---

## 9. Component 6 — Pipeline Logging

**Requirement it satisfies:** Pipeline Logging.

**Design:** structured JSON Lines (`.jsonl`) log, one line per pipeline stage per request, e.g. `logs/pipeline.jsonl`. Each line:
```json
{"ts": "...", "session_id": "...", "turn_id": 3, "stage": "router", "input_summary": "...", "output_summary": "...", "model": "llama3.1:8b", "latency_ms": 210}
```
Stages to log: contextualizer, guardrail_input, router, retrieval (per source), crag_grade (+ retry if triggered), rerank, generator, guardrail_output.

**Why JSON Lines specifically:** it's append-only (safe for a single long-running Flask process to write to continuously), trivially greppable/tail-able for debugging (matches the project's existing debugging philosophy — see the milestone's note that "the generated markdown is the best debugging surface," i.e. this team values plain, directly-inspectable artifacts over opaque ones), and loads straight into `pandas` with one line of code, which the Evaluation Pipeline (§10) needs anyway. No new logging infrastructure/dependency required.

---

## 10. Component 7 — Evaluation Pipeline

**Requirement it satisfies:** Evaluation Pipeline.

### 10.1 Test set
Claude Code should help build a test set of **~30–50 question/answer cases**, covering:
- Single-source questions, one per source (cards / offers / campaigns).
- Multi-source questions requiring combination.
- Follow-up questions that require conversation memory to resolve.
- General-banking-knowledge questions that require **no retrieval** (tests the Router's `needs_retrieval: false` path).
- Deliberately unanswerable-from-data questions (tests CRAG's "insufficient data, don't fabricate" path).
- Guardrail-trigger cases: off-topic, prompt-injection attempts, requests for personalized financial advice.

Each test case should carry, where applicable, a **hand-labeled gold set of expected record IDs** — this is possible and cheap here specifically because records have stable, human-readable IDs (`card_xyz`, `campaign_egypt_air_07`, …), unlike a typical vector-DB RAG system where you'd need embeddings just to define "the right chunk."

### 10.2 Metrics — and why these
Two tiers, deliberately kept separate because they answer different questions:

**Tier 1 — Retrieval correctness (deterministic, no LLM judge needed):**
Precision/recall of selected record IDs vs. the hand-labeled gold set. This is possible *because* this architecture works over structured, ID-addressable records rather than opaque text chunks — take advantage of that. No library needed; a short custom script comparing sets of IDs is sufficient and more trustworthy than an LLM judge for something this objectively checkable.

**Tier 2 — Answer quality (needs an LLM judge, since "is this answer good" isn't a deterministic check):**
Recommend **RAGAS** for this tier, with reasons stated explicitly:
- Its **faithfulness** metric measures whether generated claims are actually supported by the retrieved context — directly measures the exact failure mode (fabrication) that CRAG and the output guardrail are both designed to prevent, giving an independent check on whether those components are working.
- Its **answer relevancy** metric measures whether the answer actually addresses the question asked — catches cases where retrieval and grounding are fine but the answer wanders.
- Its **context precision/recall** metrics double-check retrieval+reranking quality from a different angle than the Tier-1 ID-based check, useful as a cross-validation.
- It's **model-agnostic**: RAGAS lets you plug in any LLM as the judge (including a local Ollama model via its LangChain-compatible wrapper), which matters directly to this team's stated need to keep testing different models — the evaluation harness shouldn't be locked to one vendor's model any more than the pipeline roles are.
- Using an established, published library here is preferable to writing custom LLM-judge rubrics from scratch, precisely because its metric definitions are documented and citable — useful when explaining evaluation methodology in the final PPT.

Also track **latency per stage** (pulled straight from the JSON Lines log — see, this is why §9 exists) and **guardrail pass/fail rate** on the adversarial subset of the test set.

---

## 11. Model selection guidance (not a hard requirement — config-driven, swap freely)

The team is deliberately testing multiple models (currently moved from Llama-3.2-1B to Llama-3.1-8B) and wants flexibility to keep doing so. The guidance below is about **matching model size to task demands**, not prescribing one fixed model — every role stays configurable in `config/settings.py`, same as Phase 1.

| Role | Task shape | Suggested tier | Why |
|---|---|---|---|
| `guardrail` (input/output) | High-frequency, low-reasoning classification, strict short JSON | Smaller/faster (e.g. 3B–8B class) | Runs on nearly every turn; task is close to Phase 1's `indexer` categorization job, which already proved a small model handles simple JSON classification fine. Latency here is pure overhead the user feels on every message. |
| `router` | Classification over a short fixed label set, needs to read conversation context | Smaller/faster (e.g. 3B–8B class) | Same reasoning as guardrails — structured, low-ambiguity decision, called once per turn. |
| `summarizer` (memory + contextualizer) | Needs to compress/rewrite text faithfully, but only once per turn | Mid tier (e.g. 8B class) | Summarization quality directly affects every downstream step (router, retrieval), so it deserves more capability than pure classification, but it's still a single well-scoped call. |
| `traverser` | Reads the *entire* flattened tree as text, must reliably return valid JSON node IDs | Mid tier (e.g. 8B class) | Already proven in Phase 1 to need decent instruction-following over a longer input; this is the role most sensitive to "did the model actually follow the JSON-only, don't-answer-the-question constraint." |
| `grader`/`reranker` | Nuanced relevance judgment across multiple records, not just pattern matching | Mid tier (e.g. 8B class), consider testing a step up if grading quality looks weak | This is the most judgment-heavy classification task in the pipeline — worth spending a bit more capability here than on router/guardrails. |
| `generator` | User-facing free text, must stay grounded, longest and most quality-sensitive output | Largest tier the team's hardware comfortably supports | This is the one role where output quality is directly what the end user experiences and judges the product by — it's the right place to spend the most compute budget if forced to choose. |

**General principle to keep applying as new models get tested:** classify by task shape first (strict-JSON/short-answer classification vs. open-ended generation vs. long-context reading), then pick model size for that shape — don't assume one model size fits every role just because it's the newest one being tested.

---

## 12. Updated architecture (Phase 2 request flow)

```
POST /api/chat {message, session_id}
        │
        ▼
Input Guardrail  (safe / off_topic / injection / sensitive) ──► early exit if blocked
        │
        ▼
Query Contextualizer  (rolling summary + last turns → standalone question)
        │
        ▼
Router  (needs_retrieval? which source(s)?) ──► if no retrieval needed, skip to Generator
        │
        ▼
TreeRetriever(s)  — only the router-selected sources (was: all three, unconditionally)
        │
        ▼
CRAG Grader + Reranker  (label + rank each record)
        │
   ┌────┴────┐
"Incorrect"   "Correct" / "Ambiguous"
   │              │
   ▼              │
One retry:         │
relax traversal /   │
expand sources /    │
re-grade (capped)   │
   │              │
   └──────┬───────┘
          ▼
Merge + build_context  (top-K reranked records, existing group-by-source formatting)
          │
          ▼
Generator  (existing role, now also told explicitly when data was insufficient)
          │
          ▼
Output Guardrail  (fabrication check / advice-disclaimer / no internal-detail leaks)
          │
          ▼
Update rolling memory summary for this session
          │
          ▼
Response {answer, sources, ...} — every stage above writes one line to logs/pipeline.jsonl
```

Everything below the Router line reuses Phase 1's existing `TreeRetriever`, `build_context`, and generator machinery essentially unchanged — Phase 2 is additive around that core, not a rewrite of it.

---

## 9-repeat. Suggested two-person work split

No prior division existed, so this is a proposed starting point — the team can rebalance freely, but the dependency structure below is why it's split this way (not arbitrary):

### Partner A — "Query Understanding & Retrieval Control"
1. **Conversation Memory** — rolling summarizer + last-turns buffer (§7)
2. **Query Contextualizer** (§7.1) — depends on memory, feeds the router
3. **Router** (§4) — depends on the contextualizer
4. **Input Guardrail** (§8, input half) — sits right before the router in the flow
5. **Logging utility** (§9) — build the shared JSON-Lines logger utility first, since Partner A's stages run first in the pipeline; Partner B's stages then just reuse it

### Partner B — "Retrieval Quality & Output Control"
1. **CRAG Grader + Re-ranker** (§5, §6, combined role) — depends on retrieval output (downstream of Partner A's router)
2. **Output Guardrail** (§8, output half) — sits right after generation
3. **Evaluation Pipeline** (§10) — test set, Tier-1 ID precision/recall script, RAGAS integration

### Joint (both partners, after individual pieces work standalone)
- Wiring everything into `pipeline/rag_pipeline.py`'s orchestration end-to-end.
- Running the full evaluation pass together and reviewing results.
- Final documentation/PPT prep (not detailed in this blueprint — separate task).

This split follows the natural pipeline order: Partner A owns everything from the incoming message up through "which sources to query," Partner B owns everything from "here's what came back" through to the guardrailed answer, plus proving it all works via evaluation.

---

## 13. Summary of new roles/config to add

New `role=` values to add alongside Phase 1's `indexer`/`traverser`/`generator` in `config/settings.py` and `llm/llm_client.py`:
- `router`
- `grader` (grading + reranking combined, per §6's justification)
- `summarizer` (memory rolling-summary + query contextualization)
- `guardrail` (can reuse one role for both input and output checks, or split into `guardrail_input`/`guardrail_output` if the team prefers clearer separation in logs — either is fine, just be consistent)

Every one of these follows the exact same "model proposes, Python disposes" discipline Phase 1 already established: LLM output is JSON, low/zero temperature, and every field is validated/whitelisted in Python before it's trusted downstream. Do not relax this discipline anywhere in Phase 2 — it's the core reliability property of the whole system.
