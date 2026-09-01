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

import json
import re
import time

import requests

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_INDEXER_MODEL,
    OLLAMA_TRAVERSER_MODEL,
    OLLAMA_GENERATOR_MODEL,
    OLLAMA_INDEXER_TEMPERATURE,
    OLLAMA_TRAVERSER_TEMPERATURE,
    OLLAMA_GENERATOR_TEMPERATURE,
    OLLAMA_INDEXER_NUM_CTX,
    OLLAMA_TRAVERSER_NUM_CTX,
    OLLAMA_GENERATOR_NUM_CTX,
    OLLAMA_DEFAULT_NUM_CTX,
    OLLAMA_INDEXER_NUM_PREDICT,
    OLLAMA_TRAVERSER_NUM_PREDICT,
    OLLAMA_GENERATOR_NUM_PREDICT,
    OLLAMA_DEFAULT_NUM_PREDICT,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_THINK,
    OLLAMA_TIMEOUT_SECONDS,
    LLM_PROVIDER,
    VALID_LLM_PROVIDERS,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL_BY_ROLE,
    GEMINI_THINKING_BUDGET,
    GEMINI_MIN_OUTPUT_TOKENS,
    GEMINI_TIMEOUT_SECONDS,
    GEMINI_MAX_RETRIES,
)


class LLMClient:

    # The three roles. See "THREE ROLES, NOT NINE" in
    # config/settings.py for why the other six are gone and how the
    # old names still resolve.
    MODEL_BY_ROLE = {
        "indexer": OLLAMA_INDEXER_MODEL,
        "traverser": OLLAMA_TRAVERSER_MODEL,
        "generator": OLLAMA_GENERATOR_MODEL,
    }

    # Old role name -> the role that now does that work. Kept so that
    # every existing call site, and the other worktrees, keep running:
    # an unknown role raises at construction time, which would turn a
    # rename into a crash in code this change never touched.
    ROLE_ALIASES = {
        "router": "traverser",
        "querier": "traverser",
        "summarizer": "generator",
        "grader": "generator",
        "guardrail_input": "generator",
        "guardrail_output": "generator",
    }

    TEMPERATURE_BY_ROLE = {
        "indexer": OLLAMA_INDEXER_TEMPERATURE,
        "traverser": OLLAMA_TRAVERSER_TEMPERATURE,
        "generator": OLLAMA_GENERATOR_TEMPERATURE,
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
    }

    # Ceiling on reply length per role. Without num_predict Ollama
    # imposes no limit at all, so a classifier that owes us eight
    # tokens of JSON can spend hundreds. See the reasoning on
    # OLLAMA_*_NUM_PREDICT in config/settings.py. -1 means unbounded.
    #
    # .get() with a fallback for the same reason as NUM_CTX_BY_ROLE
    # below it: a role added to MODEL_BY_ROLE but forgotten here must
    # degrade to "unbounded", never raise at construction time.
    NUM_PREDICT_BY_ROLE = {
        "indexer": OLLAMA_INDEXER_NUM_PREDICT,
        "traverser": OLLAMA_TRAVERSER_NUM_PREDICT,
        "generator": OLLAMA_GENERATOR_NUM_PREDICT,
    }

    def __init__(
        self,
        role: str = "generator",
        model: str = None,
        temperature: float = None,
        num_ctx: int = None,
        num_predict: int = None,
        provider: str = None,
    ):
        
        # A retired role name resolves to the role that took over its
        # work, rather than raising.
        role = self.ROLE_ALIASES.get(role, role)

        if role not in self.MODEL_BY_ROLE:
            valid_roles = ", ".join(self.MODEL_BY_ROLE.keys())
            raise ValueError(
                f"Unknown LLM role '{role}'. "
                f"Valid roles: {valid_roles}"
            )

        self.role = role

        self.provider = (provider or LLM_PROVIDER).strip().lower()

        if self.provider not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider '{self.provider}'. "
                f"Valid providers: {', '.join(sorted(VALID_LLM_PROVIDERS))}"
            )

        if self.provider == "gemini":
            self.model = model or GEMINI_MODEL_BY_ROLE[role]
        else:
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

        # Populated after each call; None until then.
        self.last_metrics = None

        self.num_predict = (
            num_predict
            if num_predict is not None
            else self.NUM_PREDICT_BY_ROLE.get(
                role, OLLAMA_DEFAULT_NUM_PREDICT
            )
        )

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format,
        stream: bool,
    ) -> dict:
        """
        The one place an Ollama request body is assembled, so the
        model, options and keep_alive are stated once -- a drift here
        would silently reintroduce the num_ctx reload thrash this
        module was tuned to remove.
        """

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

            "stream": stream,

            # Keeps the model resident between roles instead of letting
            # Ollama's 5-minute idle timer evict it, which would make
            # the next question pay a full ~15s reload.
            "keep_alive": OLLAMA_KEEP_ALIVE,

            "options": {
                "temperature": self.temperature,

                # Without this Ollama truncates the prompt to its own
                # small default and never says so.
                "num_ctx": self.num_ctx,

                # Ceiling, not a target. Ollama is otherwise unbounded.
                "num_predict": self.num_predict,
            },
        }

        if response_format is not None:
            payload["format"] = response_format

        # See OLLAMA_THINK in config/settings.py: without this a
        # thinking model spends its whole reply budget reasoning and
        # returns nothing at all.
        if OLLAMA_THINK in ("true", "false"):
            payload["think"] = OLLAMA_THINK == "true"

        return payload

    @staticmethod
    def _unfence(text: str) -> str:
        """
        Strip a ```json ... ``` wrapper from a reply.

        Only ever applied when JSON was REQUESTED, so a fence can only
        be wrapper noise -- this must never touch prose, where a code
        block might be the actual content.

        Needed because "format": "json" is not honoured uniformly:
        qwen3.5:4b returns

            ```json\n{"selected_node_ids": ["node_0004"]}\n```

        which is correct JSON inside a markdown fence, and which
        json.loads rejects. The traversal parser raises ValueError on
        that and takes the whole turn down. Stripping it here fixes
        every role at once rather than teaching each parser the same
        trick.
        """

        if not text:
            return text

        stripped = text.strip()

        if not stripped.startswith("```"):
            return text

        match = re.match(
            r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```$",
            stripped,
            re.DOTALL,
        )

        return match.group(1) if match else text

    def _record_metrics(self, final: dict, wall_s: float) -> None:
        """
        Store what Ollama reported about the call just made.

        WHY THIS EXISTS
        ---------------
        The pipeline log used to record what each stage DECIDED but
        never what it COST, so every latency question had to be
        answered by subtracting second-granular timestamps by hand --
        and that subtraction cannot separate "the model was reloaded"
        from "the prompt was enormous", which are the two costs worth
        telling apart on this hardware. Ollama already reports both;
        we were simply throwing the numbers away.

        All durations arrive in nanoseconds.
        """

        def secs(key):
            return round(final.get(key, 0) / 1e9, 2)

        prompt_tokens = final.get("prompt_eval_count") or 0
        output_tokens = final.get("eval_count") or 0
        read_s = secs("prompt_eval_duration")
        write_s = secs("eval_duration")

        self.last_metrics = {
            "role": self.role,
            "model": self.model,
            "num_ctx": self.num_ctx,
            "wall_s": round(wall_s, 2),
            # Non-zero here means the model was evicted and reloaded --
            # the single most useful number in this dict.
            "load_s": secs("load_duration"),
            "prompt_tokens": prompt_tokens,
            "read_s": read_s,
            "output_tokens": output_tokens,
            "write_s": write_s,
            "read_tok_s": round(prompt_tokens / read_s) if read_s else None,
            "write_tok_s": round(output_tokens / write_s) if write_s else None,
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format=None,
    ) -> str:
        """Ask the configured provider for a whole reply."""

        if self.provider == "gemini":
            return self._gemini_generate(
                system_prompt, user_prompt, response_format
            )

        return self._ollama_generate(
            system_prompt, user_prompt, response_format
        )

    def _ollama_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format=None,
    ) -> str:

        url = f"{OLLAMA_BASE_URL}/api/chat"

        payload = self._build_payload(
            system_prompt, user_prompt, response_format, stream=False
        )

        started = time.time()

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
            content = data["message"]["content"]

        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(
                "Ollama returned an unexpected response format."
            ) from e

        self._record_metrics(data, time.time() - started)

        if response_format == "json":
            content = self._unfence(content)

        return content

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------
    #
    # Same two public methods, same return types, same RuntimeError on
    # failure -- so every caller in the pipeline is unchanged and the
    # provider really is one setting.
    #
    # WHAT DOES NOT CARRY OVER
    # ------------------------
    # num_ctx is meaningless here: there is no KV cache to size and no
    # card to overflow, which is the whole reason this backend is
    # thirty times faster on the traversal prompt. It is accepted and
    # ignored rather than removed, so switching providers needs no
    # change at the call sites.

    def _gemini_url(self, method: str) -> str:
        return f"{GEMINI_BASE_URL}/models/{self.model}:{method}"

    def _gemini_body(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format,
    ) -> dict:

        config = {
            "temperature": self.temperature,
            # The local caps exist to stop a small model rambling.
            # Applying them here truncates mid-JSON instead, and with
            # thinking enabled a tight cap can consume the whole budget
            # before a single visible token is produced.
            "maxOutputTokens": max(
                GEMINI_MIN_OUTPUT_TOKENS,
                self.num_predict if self.num_predict > 0 else 0,
            ),
        }

        if response_format == "json":
            config["responseMimeType"] = "application/json"

        # Empty means "send no thinkingConfig", which is the default
        # and the only setting every model accepts -- see the measured
        # note on GEMINI_THINKING_BUDGET in config/settings.py.
        if GEMINI_THINKING_BUDGET != "":
            config["thinkingConfig"] = {
                "thinkingBudget": int(GEMINI_THINKING_BUDGET)
            }

        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "generationConfig": config,
        }

    def _gemini_post(self, url: str, body: dict, stream: bool):
        """
        POST with one quiet retry on the transient failures a hosted
        API actually has: rate limiting and short-lived 5xx. Anything
        else -- a bad key, a retired model -- is permanent, and
        retrying it just doubles the time before the user sees the
        error.
        """

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set, but LLM_PROVIDER is "
                "'gemini'. Set the key in your environment, or set "
                "LLM_PROVIDER=ollama to run locally."
            )

        last_error = None

        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url,
                    headers={
                        "x-goog-api-key": GEMINI_API_KEY,
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=GEMINI_TIMEOUT_SECONDS,
                    stream=stream,
                )

                if response.status_code in (429, 500, 503, 504):
                    last_error = (
                        f"HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )

                    if attempt < GEMINI_MAX_RETRIES:
                        time.sleep(1.5 * (attempt + 1))
                        continue

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Gemini returned HTTP "
                        f"{response.status_code} for model "
                        f"'{self.model}': {response.text[:300]}"
                    )

                return response

            except requests.exceptions.Timeout as e:
                last_error = f"timed out after {GEMINI_TIMEOUT_SECONDS}s"

                if attempt < GEMINI_MAX_RETRIES:
                    continue

                raise RuntimeError(
                    f"Gemini timed out after "
                    f"{GEMINI_TIMEOUT_SECONDS} seconds while using "
                    f"model '{self.model}'."
                ) from e

            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Error communicating with Gemini: {e}"
                ) from e

        raise RuntimeError(
            f"Gemini request failed for model '{self.model}' after "
            f"{GEMINI_MAX_RETRIES + 1} attempts. Last error: "
            f"{last_error}"
        )

    @staticmethod
    def _gemini_text(payload: dict) -> str:
        """
        Pull the reply text out of a response.

        Defensive because the shape is genuinely optional: a filtered
        or empty candidate has no parts at all, and reaching blindly
        into [0] turns that into an IndexError far from the cause.
        """

        candidates = payload.get("candidates") or []

        if not candidates:
            return ""

        parts = (candidates[0].get("content") or {}).get("parts") or []

        return "".join(part.get("text", "") for part in parts)

    def _gemini_record_metrics(self, payload: dict, wall_s: float) -> None:
        """
        Same metrics dict as the Ollama path, so pipeline_logger and
        every reader of the log keep working across a provider switch.

        load_s is always 0 here and that is meaningful, not missing:
        there is no model to evict and reload, which is precisely the
        cost this backend removes.
        """

        usage = payload.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount") or 0
        output_tokens = usage.get("candidatesTokenCount") or 0

        self.last_metrics = {
            "role": self.role,
            "model": self.model,
            "provider": "gemini",
            "num_ctx": None,
            "wall_s": round(wall_s, 2),
            "load_s": 0.0,
            "prompt_tokens": prompt_tokens,
            # The API reports totals, not a read/write split, so the
            # per-phase timings the Ollama path logs are not available.
            "read_s": None,
            "output_tokens": output_tokens,
            "write_s": None,
            "read_tok_s": None,
            "write_tok_s": (
                round(output_tokens / wall_s) if wall_s else None
            ),
            "thinking_tokens": usage.get("thoughtsTokenCount") or 0,
        }

    def _gemini_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format=None,
    ) -> str:

        body = self._gemini_body(
            system_prompt, user_prompt, response_format
        )

        started = time.time()

        response = self._gemini_post(
            self._gemini_url("generateContent"), body, stream=False
        )

        try:
            payload = response.json()
        except ValueError as e:
            raise RuntimeError(
                "Gemini returned an unexpected response format."
            ) from e

        self._gemini_record_metrics(payload, time.time() - started)

        text = self._gemini_text(payload)

        if response_format == "json":
            text = self._unfence(text)

        if not text:
            # Almost always a thinking budget that ate the whole output
            # allowance, or a safety filter. Say which, because the
            # callers all fail open and would otherwise hide it.
            reason = (payload.get("candidates") or [{}])[0].get(
                "finishReason", "unknown"
            )
            raise RuntimeError(
                f"Gemini returned no text for role '{self.role}' "
                f"(finishReason={reason}). If this is MAX_TOKENS, "
                f"raise GEMINI_MIN_OUTPUT_TOKENS."
            )

        return text
