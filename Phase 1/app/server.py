"""
app/server.py

WHAT THIS FILE DOES
--------------------
A minimal Flask web server with two routes:
  GET  /          -> serves the chat page (index.html)
  POST /api/chat  -> receives {"message": "..."}, runs it through
                      RagPipeline, and returns {"answer": "...", "sources": [...]}

WHY IT EXISTS AS A THIN LAYER
---------------------------------
The frontend is explicitly "not the main focus" per the project
requirements -- the RAG/retrieval system is. So this file does the
minimum needed to expose the pipeline over HTTP: no sessions, no
database, no auth. It creates ONE RagPipeline instance at startup
(so data isn't reloaded from Excel/JSON on every request) and calls
`.answer(question)` on each incoming message.

INPUTS  : HTTP requests from the browser
OUTPUTS : HTML page (GET /) and JSON answers (POST /api/chat)
"""

import json

from flask import Flask, Response, request, jsonify, render_template

from pipeline.rag_pipeline import RagPipeline
from config.settings import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

app = Flask(__name__)

# Created once at startup: loading all three data sources takes a
# moment, so we don't want to redo it on every single chat message.
pipeline = RagPipeline()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not question:
        return jsonify({"error": "Empty message"}), 400

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    try:
        result = pipeline.answer(question, session_id=session_id)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    sources = [
        {"source": r.source, "title": r.title} for r in result["retrieved_records"]
    ]

    return jsonify({"answer": result["answer"], "sources": sources})


def _sse(event: str, payload: dict) -> str:
    """One Server-Sent Events frame."""

    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.route("/api/chat/progress", methods=["POST"])
def chat_progress():
    """
    The same turn as /api/chat, reported as it happens.

    WHY A SECOND ROUTE RATHER THAN REPLACING THE FIRST
    --------------------------------------------------
    /api/chat is what the tests and any script hitting this service
    use, and a plain JSON POST is the right shape for them. The browser
    wants something different -- on this hardware an answer takes tens
    of seconds, and a silent wait that long reads as a hung app. Both
    routes run the SAME pipeline turn; only the delivery differs.

    Events emitted:
      stage  {stage, label, detail}  a step began
      done   {answer, sources}
      error  {message}

    The ANSWER is not streamed. It is sent once, in `done`, after the
    output guardrail has approved it -- see answer_with_progress() in
    pipeline/rag_pipeline.py for why. What streams is only which step
    the turn is on.
    """

    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not question:
        return jsonify({"error": "Empty message"}), 400

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    def events():
        try:
            for kind, payload in pipeline.answer_with_progress(
                question, session_id=session_id
            ):
                if kind == "stage":
                    yield _sse("stage", payload)

                elif kind == "result":
                    sources = [
                        {"source": r.source, "title": r.title}
                        for r in payload["retrieved_records"]
                    ]

                    yield _sse(
                        "done",
                        {
                            "answer": payload["answer"],
                            "sources": sources,
                        },
                    )

        except RuntimeError as e:
            # Ollama unreachable, timed out, or refused the model.
            yield _sse("error", {"message": str(e)})

        except Exception as e:  # noqa: BLE001
            # The stream has already begun, so there is no status code
            # left to set -- the only way to tell the browser is in-band.
            yield _sse(
                "error",
                {"message": f"Unexpected server error: {e}"},
            )

    return Response(
        events(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
