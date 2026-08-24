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
    OLLAMA_ROUTER_MODEL,
    OLLAMA_SUMMARIZER_MODEL,
    OLLAMA_GUARDRAIL_INPUT_MODEL,
    OLLAMA_GUARDRAIL_OUTPUT_MODEL,
    OLLAMA_GRADER_MODEL,
    OLLAMA_INDEXER_TEMPERATURE,
    OLLAMA_TRAVERSER_TEMPERATURE,
    OLLAMA_GENERATOR_TEMPERATURE,
    OLLAMA_ROUTER_TEMPERATURE,
    OLLAMA_SUMMARIZER_TEMPERATURE,
    OLLAMA_GUARDRAIL_INPUT_TEMPERATURE,
    OLLAMA_GUARDRAIL_OUTPUT_TEMPERATURE,
    OLLAMA_GRADER_TEMPERATURE,
    OLLAMA_INDEXER_NUM_CTX,
    OLLAMA_TRAVERSER_NUM_CTX,
    OLLAMA_GENERATOR_NUM_CTX,
    OLLAMA_ROUTER_NUM_CTX,
    OLLAMA_SUMMARIZER_NUM_CTX,
    OLLAMA_GUARDRAIL_INPUT_NUM_CTX,
    OLLAMA_GUARDRAIL_OUTPUT_NUM_CTX,
    OLLAMA_GRADER_NUM_CTX,
    OLLAMA_DEFAULT_NUM_CTX,
    OLLAMA_TIMEOUT_SECONDS,
)


class LLMClient:

    MODEL_BY_ROLE = {
        "indexer": OLLAMA_INDEXER_MODEL,
        "traverser": OLLAMA_TRAVERSER_MODEL,
        "generator": OLLAMA_GENERATOR_MODEL,
        "router": OLLAMA_ROUTER_MODEL,
        "summarizer": OLLAMA_SUMMARIZER_MODEL,
        "guardrail_input": OLLAMA_GUARDRAIL_INPUT_MODEL,
        "guardrail_output": OLLAMA_GUARDRAIL_OUTPUT_MODEL,
        "grader": OLLAMA_GRADER_MODEL,
    }

    TEMPERATURE_BY_ROLE = {
        "indexer": OLLAMA_INDEXER_TEMPERATURE,
        "traverser": OLLAMA_TRAVERSER_TEMPERATURE,
        "generator": OLLAMA_GENERATOR_TEMPERATURE,
        "router": OLLAMA_ROUTER_TEMPERATURE,
        "summarizer": OLLAMA_SUMMARIZER_TEMPERATURE,
        "guardrail_input": OLLAMA_GUARDRAIL_INPUT_TEMPERATURE,
        "guardrail_output": OLLAMA_GUARDRAIL_OUTPUT_TEMPERATURE,
        "grader": OLLAMA_GRADER_TEMPERATURE,
    }

    # Context window per role. This MUST be sent explicitly: Ollama
    # otherwise falls back to a small default (4096 for llama3.1:8b)
    # and silently truncates any longer prompt. See the long comment on
    # OLLAMA_*_NUM_CTX in config/settings.py for the measurement that
    # made this necessary.
    NUM_CTX_BY_ROLE = {
        "indexer": OLLAMA_INDEXER_NUM_CTX,
        "traverser": OLLAMA_TRAVERSER_NUM_CTX,
        "generator": OLLAMA_GENERATOR_NUM_CTX,
        "router": OLLAMA_ROUTER_NUM_CTX,
        "summarizer": OLLAMA_SUMMARIZER_NUM_CTX,
        "guardrail_input": OLLAMA_GUARDRAIL_INPUT_NUM_CTX,
        "guardrail_output": OLLAMA_GUARDRAIL_OUTPUT_NUM_CTX,
        "grader": OLLAMA_GRADER_NUM_CTX,
    }

    def __init__(
        self,
        role: str = "generator",
        model: str = None,
        temperature: float = None,
        num_ctx: int = None,
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

        # .get() with a fallback, not [role], on purpose: a role added to
        # MODEL_BY_ROLE but not here should degrade to a working default
        # rather than raise KeyError when the client is constructed. That
        # is not hypothetical -- it happened when the five Phase 2 roles
        # were added to the model and temperature tables only, which made
        # LLMClient(role="router") and every other new role unusable.
        self.num_ctx = (
            num_ctx
            if num_ctx is not None
            else self.NUM_CTX_BY_ROLE.get(role, OLLAMA_DEFAULT_NUM_CTX)
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

                # Without this Ollama truncates the prompt to its own
                # small default and never says so.
                "num_ctx": self.num_ctx,
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