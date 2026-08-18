"""
llm/llm_client.py

WHAT THIS FILE DOES
--------------------
A thin wrapper around Ollama's local HTTP API. Every other module in
this project calls `LLMClient().generate(...)` and never talks to
Ollama's HTTP endpoint directly.

WHY IT EXISTS
-------------
If we scattered `requests.post("http://localhost:11434/...")` calls
across the pipeline, agent, and evaluation code, switching to a
different model runtime later (e.g. a hosted API, or vLLM) would mean
hunting through every file. Instead, EVERY OTHER MODULE only depends
on this class's `generate(system_prompt, user_prompt)` method. Swap
what happens *inside* generate() and nothing else in the project needs
to change.

HOW OLLAMA WORKS (briefly)
----------------------------
Ollama runs a local server (started with `ollama serve`, or
automatically when you run `ollama run <model>`) that exposes a
simple JSON HTTP API at http://localhost:11434. We use the
`/api/chat` endpoint, which accepts a list of {role, content}
messages -- same shape as OpenAI/Anthropic chat APIs -- and returns
the model's reply.

INPUTS  : system_prompt (str), user_prompt (str)
OUTPUTS : the model's reply as a plain string

REQUIRES: Ollama installed and running locally (`ollama serve`),
          with the model in config.settings.OLLAMA_MODEL already
          pulled (`ollama pull llama3.1:8b`, or whichever you set).
"""

import requests

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_INDEXER_MODEL,
    OLLAMA_TRAVERSER_MODEL,
    OLLAMA_GENERATOR_MODEL,
    OLLAMA_INDEXER_TEMPERATURE,
    OLLAMA_TRAVERSER_TEMPERATURE,
    OLLAMA_GENERATOR_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
)


class LLMClient:

    MODEL_BY_ROLE = {
        "indexer": OLLAMA_INDEXER_MODEL,
        "traverser": OLLAMA_TRAVERSER_MODEL,
        "generator": OLLAMA_GENERATOR_MODEL,
    }

    TEMPERATURE_BY_ROLE = {
        "indexer": OLLAMA_INDEXER_TEMPERATURE,
        "traverser": OLLAMA_TRAVERSER_TEMPERATURE,
        "generator": OLLAMA_GENERATOR_TEMPERATURE,
    }

    def __init__(
        self,
        role: str = "generator",
        model: str = None,
        temperature: float = None,
    ):
        
        if role not in self.MODEL_BY_ROLE:
            valid_roles = ", ".join(self.MODEL_BY_ROLE.keys())
            raise ValueError(
                f"Unknown LLM role '{role}'. "
                f"Valid roles: {valid_roles}"
            )

        self.role = role

        self.model = model or self.MODEL_BY_ROLE[role]

        self.temperature = (
            temperature
            if temperature is not None
            else self.TEMPERATURE_BY_ROLE[role]
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format=None,
    ) -> str:
       
        url = f"{OLLAMA_BASE_URL}/api/chat"

        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            "stream": False,

            "options": {
                "temperature": self.temperature,
            },
        }

        if response_format is not None:
            payload["format"] = response_format

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Could not reach local Ollama at "
                f"{OLLAMA_BASE_URL}. "
                f"Make sure Ollama is running and the model "
                f"'{self.model}' is installed."
            ) from e

        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Ollama timed out after "
                f"{OLLAMA_TIMEOUT_SECONDS} seconds "
                f"while using model '{self.model}'."
            ) from e

        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Ollama returned an error for model "
                f"'{self.model}'. "
                f"Make sure the model is installed with "
                f"`ollama pull {self.model}`. "
                f"Details: {e}"
            ) from e

        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Error communicating with local Ollama: {e}"
            ) from e

        try:
            data = response.json()
            return data["message"]["content"]

        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(
                "Ollama returned an unexpected response format."
            ) from e