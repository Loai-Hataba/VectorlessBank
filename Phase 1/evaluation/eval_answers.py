"""
evaluation/eval_answers.py

TIER 2 EVALUATION -- answer quality, judged by an LLM via RAGAS.

WHAT THIS MEASURES
------------------
Tier 1 (eval_retrieval.py) asks "did retrieval FIND the right records?"
and answers it with set arithmetic. This asks the question set
arithmetic cannot: "is the answer the system WROTE actually supported
by what it retrieved?"

    generated_answers.json      <- written by eval_generate.py
        |
        v
    RAGAS + local Ollama judge
        |
        v
    faithfulness / context precision, per case and averaged

WHY RAGAS RATHER THAN A HAND-WRITTEN JUDGE PROMPT
--------------------------------------------------
Its metric definitions are published and citable, which matters when
the evaluation methodology has to be explained rather than just run. It
is also model-agnostic: the judge is whatever the config points at, so
the harness is not locked to one vendor any more than the pipeline
roles are.

THE METRICS, AND WHY THESE TWO BY DEFAULT
------------------------------------------
faithfulness
    Breaks the answer into individual claims and checks each one
    against the retrieved context. This is a direct measurement of the
    exact failure the whole project is built to prevent -- the
    fabricated "EGP 4,500 annual fee" -- and an independent check on
    whether CRAG and the output guardrail are actually working.

context precision (without reference)
    Of the records that reached the generator, how many were actually
    useful for the answer? This grades the router, the tree traversal
    and the re-ranker together, from a different angle than Tier 1's
    ID matching: Tier 1 knows whether the right record was fetched,
    this knows whether the fetched records earned their place.

Both are reference-free -- they need only the question, the answer and
the contexts. That is why they are the default: the testset labels gold
record IDs but has no hand-written model answers, so any metric needing
a `reference` cannot run until someone writes 29 of them.

WHY THE EMBEDDING METRIC IS OPT-IN AND OFF BY DEFAULT
-----------------------------------------------------
RAGAS's answer-relevancy metric works by generating questions from the
answer and comparing them to the real question WITH AN EMBEDDING MODEL.
This project's defining constraint is that it uses no embeddings, and
the Phase 2 blueprint says that constraint must not be quietly
weakened.

There is a real distinction available here -- an embedding used to
score a test is not an embedding used to retrieve an answer, and the
product would remain vectorless either way -- but that is a call for
the team to make explicitly, not for an eval script to make silently by
importing a model on first run. So it is behind --with-embeddings, and
off unless asked for.

MODEL PROPOSES, PYTHON DISPOSES -- STILL
-----------------------------------------
The judge is a small local model and will fail on some samples, which
RAGAS reports as NaN rather than raising. Those are counted and shown
separately instead of being averaged in as zero, because "the judge
could not read this" and "the answer was unfaithful" are different
findings and averaging them together would silently understate quality.

WHY IT RUNS IN A SEPARATE ENVIRONMENT
--------------------------------------
RAGAS brings a large langchain/datasets dependency tree. The
application must not acquire it -- see eval_generate.py's docstring.
This file imports nothing from the project except config/settings.py,
which is pure stdlib, so it runs happily in an environment that has
ragas but none of the app's own dependencies.

KNOWN LIMITATION: AN 8B JUDGE CANNOT SCORE FAITHFULNESS ON CARDS
-----------------------------------------------------------------
Measured, on a real run, so that nobody has to rediscover it:

    case         faithfulness   context_precision
    offers_002         1.000               1.000
    cards_001    judge failed               0.000

Two separate causes were found, and only one of them is fixed.

FIXED -- the silent truncation. RAGAS talks to Ollama over /v1, which
accepts no num_ctx, so the judge was seeing ~4096 tokens of a ~10,250
token card record. Baking the window into a model variant fixes it and
preflight_judge_context() now warns when it is missing. With that done,
context precision on the cards case went from "judge failed" to a real
number.

NOT FIXED -- faithfulness on the largest records still fails, even with
a 32k window. The metric decomposes the answer into claims and runs an
entailment check against the whole context per claim, and an 8B model
does not hold the required JSON shape across that much text; RAGAS
retries, gives up, and reports the sample as unparseable.

That is a judge capability limit, not a defect in the pipeline being
measured, and it lands precisely on the source that matters most --
cards, where a record is ~41,000 characters and where the fabricated
"EGP 4,500 annual fee" came from. Point RAGAS_JUDGE_MODEL at a larger
model before reading any faithfulness number as coverage of cards.

The harness reports these as "judge failed" and excludes them from the
average rather than scoring them zero, so the gap is visible instead of
being quietly averaged into a plausible-looking result.

USAGE
-----
    # in the ragas environment
    python evaluation/eval_answers.py
    python evaluation/eval_answers.py --limit 5
    python evaluation/eval_answers.py --category single_source_cards
    python evaluation/eval_answers.py --metrics faithfulness
    python evaluation/eval_answers.py --json tier2_scores.json
    python evaluation/eval_answers.py --with-embeddings     # see above

INPUTS  : evaluation/generated_answers.json
OUTPUTS : a printed report, optionally a JSON file

REQUIRES: pip install -r evaluation/requirements-eval.txt
          Ollama running with the judge model pulled.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

# config/settings.py imports only os and pathlib, so this works in an
# environment that has ragas but not flask/openpyxl/requests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (                              # noqa: E402
    OLLAMA_BASE_URL,
    RAGAS_JUDGE_MODEL,
    RAGAS_JUDGE_TEMPERATURE,
    RAGAS_JUDGE_TIMEOUT_SECONDS,
    RAGAS_MAX_WORKERS,
)

DEFAULT_IN = Path(__file__).resolve().parent / "generated_answers.json"

# RAGAS names its output columns after the metric class, so the report
# needs the mapping to read them back.
METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "context_precision": "llm_context_precision_without_reference",
    "context_recall": "context_recall",
    "answer_relevancy": "answer_relevancy",
}

REFERENCE_FREE = ("faithfulness", "context_precision")


def preflight_judge_context() -> str | None:
    """
    Refuse to run quietly against a judge that will be silently truncated.

    THE PROBLEM
    -----------
    RAGAS reaches Ollama through its OpenAI-compatible /v1 endpoint,
    which accepts no num_ctx. Ollama therefore falls back to its own
    small default (4096 for llama3.1:8b) and silently discards the rest
    of the prompt -- the exact failure documented at length on the
    OLLAMA_*_NUM_CTX settings, resurfacing in a place where none of
    those settings can reach.

    It is not theoretical here. Measured: a prompt with a keyword at
    character zero and ~7,200 tokens of text after it comes back "there
    is no secret code". Passing extra_body={"options": {"num_ctx": ...}}
    changes nothing; the parameter is not part of the OpenAI schema and
    Ollama ignores it.

    The consequence for Tier 2 is worse than a low score: the judge
    fails to produce parseable output at all and the sample is dropped.
    The cards source is affected most, because a card record is ~41,000
    characters -- so the metric silently stops covering the very source
    the project's fabrication bug came from.

    THE FIX
    -------
    Bake the window into a model variant, which /v1 then honours:

        printf 'FROM llama3.1:8b\\nPARAMETER num_ctx 32768\\n' \\
            > Modelfile.judge
        ollama create llama3.1:8b-eval32k -f Modelfile.judge
        set RAGAS_JUDGE_MODEL=llama3.1:8b-eval32k

    Verified: the same prompt that failed above returns the keyword.

    Returns a warning string when the judge has no baked num_ctx, or
    None when it looks correctly configured. Never raises -- a probe
    failure must not stop an evaluation that might otherwise work.
    """
    import json
    import urllib.error
    import urllib.request

    try:
        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/show",
            data=json.dumps({"model": RAGAS_JUDGE_MODEL}).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            info = json.load(response)

    except (urllib.error.URLError, OSError, ValueError):
        return (
            f"could not inspect '{RAGAS_JUDGE_MODEL}' -- is Ollama running?"
        )

    if "num_ctx" in (info.get("parameters") or ""):
        return None

    return (
        f"judge model '{RAGAS_JUDGE_MODEL}' has no num_ctx baked in, so\n"
        f"         Ollama's /v1 endpoint will truncate long prompts to its\n"
        f"         default (~4096 tokens) WITHOUT SAYING SO. Long records --\n"
        f"         cards especially -- will fail to score.\n"
        f"         Fix: see preflight_judge_context() in this file."
    )


def build_judge():
    """
    The judge LLM: whatever config points at, reached through Ollama's
    OpenAI-compatible endpoint.

    Ollama serves an OpenAI-shaped API at /v1, and langchain_openai is
    already a RAGAS dependency, so this needs no extra package. The API
    key is required by the client and ignored by Ollama.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=RAGAS_JUDGE_MODEL,
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key="ollama",
            temperature=RAGAS_JUDGE_TEMPERATURE,
            timeout=RAGAS_JUDGE_TIMEOUT_SECONDS,
        )
    )


def build_metrics(names: list, with_embeddings: bool):
    """
    Instantiate the requested metrics, and say plainly what each one
    will need that this project may not have.
    """
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )

    metrics = []

    for name in names:

        if name == "faithfulness":
            metrics.append(Faithfulness())

        elif name == "context_precision":
            metrics.append(LLMContextPrecisionWithoutReference())

        elif name == "context_recall":
            # Needs a hand-written model answer per case. The testset
            # has none today, so this stays opt-in rather than failing
            # 29 samples with a confusing error.
            metrics.append(LLMContextRecall())

        elif name == "answer_relevancy":
            if not with_embeddings:
                raise SystemExit(
                    "answer_relevancy needs an embedding model, which this "
                    "project deliberately does not use.\n"
                    "Pass --with-embeddings to accept that, and see the "
                    "note in this file's docstring first."
                )
            from ragas.metrics import ResponseRelevancy
            metrics.append(ResponseRelevancy())

        else:
            raise SystemExit(f"Unknown metric '{name}'.")

    return metrics


def load_samples(args) -> tuple[list, list, dict]:
    """
    Split the cached run into what can be scored and what cannot.

    Returns (scorable, skipped, run_metadata).
    """
    path = Path(args.infile)

    if not path.exists():
        raise SystemExit(
            f"No generated answers at {path}.\n"
            f"Run this first, in the project environment:\n"
            f"    python evaluation/eval_generate.py"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])

    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]

    if args.case:
        samples = [s for s in samples if s.get("id") == args.case]

    scorable = [s for s in samples if s.get("scorable")]
    skipped = [s for s in samples if not s.get("scorable")]

    if args.limit:
        scorable = scorable[: args.limit]

    metadata = {
        "generated_at": payload.get("generated_at"),
        "models": payload.get("models", {}),
        "settings": payload.get("settings", {}),
    }

    return scorable, skipped, metadata


def needs_reference(names: list) -> bool:
    return any(n not in REFERENCE_FREE for n in names)


def mean_ignoring_nan(values: list) -> tuple[float, int]:
    """
    Average the scores the judge actually produced, and report how many
    it could not.

    A NaN means the judge failed on that sample, not that the answer
    scored zero. Folding those in as zeros would understate the system
    and hide a judge problem as a product problem.
    """
    usable = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]

    if not usable:
        return float("nan"), len(values)

    return sum(usable) / len(usable), len(values) - len(usable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default=str(DEFAULT_IN))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--case")
    parser.add_argument(
        "--metrics",
        default=",".join(REFERENCE_FREE),
        help="comma-separated: faithfulness, context_precision, "
             "context_recall, answer_relevancy",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="allow metrics that require an embedding model",
    )
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    names = [n.strip() for n in args.metrics.split(",") if n.strip()]

    scorable, skipped, metadata = load_samples(args)

    if not scorable:
        print("Nothing scorable in that file.")
        if skipped:
            print(f"({len(skipped)} sample(s) were marked unscorable.)")
        raise SystemExit(1)

    if needs_reference(names) and not any(s.get("reference") for s in scorable):
        print(
            "WARNING: a requested metric needs a hand-written reference "
            "answer per case,\n         and none of the samples has one. "
            "Those scores will be NaN.\n"
        )

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.get("standalone_question") or s["question"],
                retrieved_contexts=s["contexts"],
                response=s["answer"],
                reference=s.get("reference"),
            )
            for s in scorable
        ]
    )

    print(f"Judge   : {RAGAS_JUDGE_MODEL} via {OLLAMA_BASE_URL}")
    print(f"Metrics : {', '.join(names)}")
    print(f"Scoring : {len(scorable)} sample(s), {len(skipped)} skipped")
    print(f"Run     : generator={metadata['models'].get('generator')} "
          f"top_k={metadata['settings'].get('RERANK_TOP_K')}")

    context_warning = preflight_judge_context()

    if context_warning:
        print()
        print(f"WARNING: {context_warning}")

    print()

    started = time.time()

    result = evaluate(
        dataset=dataset,
        metrics=build_metrics(names, args.with_embeddings),
        llm=build_judge(),
        # Local models serve one request at a time in practice; firing
        # dozens at once makes every one of them slower rather than
        # finishing sooner.
        run_config=RunConfig(
            timeout=RAGAS_JUDGE_TIMEOUT_SECONDS,
            max_workers=RAGAS_MAX_WORKERS,
        ),
        show_progress=True,
    )

    elapsed = time.time() - started

    frame = result.to_pandas()

    columns = {
        name: METRIC_COLUMNS[name]
        for name in names
        if METRIC_COLUMNS[name] in frame.columns
    }

    rows = []

    for position, sample in enumerate(scorable):
        row = {
            "id": sample["id"],
            "category": sample["category"],
            "retrieved": len(sample["contexts"]),
        }

        for name, column in columns.items():
            value = float(frame[column][position])
            row[name] = None if math.isnan(value) else round(value, 3)

        rows.append(row)

    # ---------------------------------------------------------------
    # Per-case
    # ---------------------------------------------------------------

    print()
    header = f"{'case':<13}{'cat':<24}" + "".join(f"{n[:14]:>16}" for n in columns)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['id']:<13}{row['category'][:23]:<24}"
        for name in columns:
            value = row[name]
            line += f"{'judge failed' if value is None else f'{value:.3f}':>16}"
        print(line)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print("=" * 62)
    print(f"{len(rows)} scored in {elapsed / 60:.1f} min")
    print("-" * 62)

    summary = {}

    for name in columns:
        average, failures = mean_ignoring_nan([r[name] if r[name] is not None
                                               else float("nan") for r in rows])
        summary[name] = {
            "mean": None if math.isnan(average) else round(average, 3),
            "judge_failures": failures,
        }
        shown = "n/a" if math.isnan(average) else f"{average:.3f}"
        print(f"  {name:<22} {shown}"
              + (f"   ({failures} judge failure(s))" if failures else ""))

    print("-" * 62)

    by_category: dict[str, list] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    for category, group in sorted(by_category.items()):
        parts = []
        for name in columns:
            average, _ = mean_ignoring_nan(
                [r[name] if r[name] is not None else float("nan") for r in group]
            )
            parts.append(f"{name[:4]}={'n/a' if math.isnan(average) else f'{average:.2f}'}")
        print(f"  {category:<26} {'  '.join(parts)}  ({len(group)} case(s))")

    if skipped:
        print("-" * 62)
        print(f"  not scored ({len(skipped)}):")
        reasons: dict[str, int] = {}
        for sample in skipped:
            reasons[sample.get("not_scorable_because", "unknown")] = (
                reasons.get(sample.get("not_scorable_because", "unknown"), 0) + 1
            )
        for reason, count in sorted(reasons.items()):
            print(f"    {count:>2}  {reason}")

    print("=" * 62)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "judge": RAGAS_JUDGE_MODEL,
                    "metrics": names,
                    "run": metadata,
                    "seconds": round(elapsed, 1),
                    "summary": summary,
                    "cases": rows,
                    "not_scored": [
                        {"id": s["id"], "reason": s.get("not_scorable_because")}
                        for s in skipped
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
