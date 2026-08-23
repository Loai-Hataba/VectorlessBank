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

from flask import Flask, request, jsonify, render_template

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


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
