"""
run.py

WHAT THIS FILE DOES
--------------------
The single command to start the whole application:
    python run.py

It just imports the Flask app object from app/server.py and runs it.
Kept separate from app/server.py itself so that server.py can also be
imported by tests or other scripts without automatically starting a
web server.

REQUIRES: Ollama running locally (`ollama serve`) with the model from
          config/settings.py already pulled.
"""

from app.server import app
from config.settings import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

if __name__ == "__main__":
    print(f"Starting BM Retail Banking Assistant on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
