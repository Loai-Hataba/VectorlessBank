"""
pipeline/pipeline_logger.py

WHAT THIS FILE DOES
--------------------
Writes one structured JSON line per pipeline stage per request to
logs/pipeline.jsonl (config.settings.PIPELINE_LOG_PATH). Every other
Phase 2 module (guardrail, contextualizer, router, retrieval, CRAG,
generation) calls `log_stage(...)` at the point it finishes its work,
instead of using print() or writing its own log file.

WHY IT EXISTS AS ITS OWN STEP
--------------------------------
Phase 1 returned retrieved_records so a developer could see what
happened in one request. Phase 2 has 5+ stages per turn, across two
people's code, and needs to answer questions like "how often does the
router pick the wrong source" or "what fraction of turns get flagged
by the guardrail" -- across MANY requests, not just the one you're
looking at in a terminal. That means every stage's decision has to be
written down somewhere queryable, in the same shape, every time.

JSON Lines (one JSON object per line, appended) was chosen over a
single JSON file because:
  - It's append-only: nothing has to be read back and rewritten on
    every request, so it's safe from one long-running Flask process.
  - Every line is independently parseable, so a crashed/killed process
    never corrupts previous entries the way a truncated single JSON
    array would.
  - pandas.read_json(path, lines=True) loads it directly for the
    Evaluation Pipeline (Phase 2, Partner B).

WHY LOGGING FAILURES NEVER RAISE
------------------------------------
This file is an observability tool, not a feature the chatbot depends
on to answer questions. A disk-full error or a bad LOGS_DIR permission
should show up loudly in the terminal, but must never turn into a
500 for a live user question. Every write is wrapped accordingly.

INPUTS  : stage (str), session_id (str), turn_id (str), data (dict)
OUTPUTS : none (writes a line to PIPELINE_LOG_PATH; returns nothing)
"""

import json
import sys
import time

from config.settings import LOGS_DIR, PIPELINE_LOG_PATH


def log_stage(
    stage: str,
    session_id: str,
    turn_id: str,
    data: dict,
) -> None:
    """
    Append one JSON line recording what happened at one pipeline stage.

    stage      : short stage name, e.g. "guardrail_input", "contextualizer",
                 "router", "retrieval", "generation". Free text by
                 convention, not validated against a fixed list --
                 new stages (CRAG, reranking, guardrail_output) should
                 just pick a clear name and start calling this.
    session_id : identifies one ongoing conversation. Same value as
                 used by memory/conversation_memory.py, so log lines
                 and memory state can be correlated for the same user.
    turn_id    : identifies one user message within that session
                 (e.g. an incrementing int or a uuid). Every stage
                 that runs while answering ONE user message should be
                 logged with the SAME turn_id, so the Evaluation
                 Pipeline can group all of a turn's stages together.
    data       : whatever that stage wants to record (its decision,
                 timing, input/output sizes, etc.) -- must be
                 JSON-serializable. No fixed schema is enforced here
                 on purpose: each stage knows its own shape best, and
                 forcing one shared schema would mean editing this
                 file every time any stage's logging needs change.

    Never raises. A logging failure is printed to stderr and swallowed
    so it can never turn into a user-facing error.
    """

    entry = {
        "timestamp": time.time(),
        "stage": stage,
        "session_id": session_id,
        "turn_id": turn_id,
        **data,
    }

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        with open(PIPELINE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    except Exception as e:
        print(
            f"[pipeline_logger] WARNING: failed to write log entry "
            f"for stage '{stage}': {e}",
            file=sys.stderr,
        )