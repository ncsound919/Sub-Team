"""
LLM Client — OpenRouter-backed inference with graceful degradation.

This module provides a thin, optional wrapper around the OpenAI-compatible
client that targets OpenRouter.  It is used by the four CPU pipeline agents
and the CrossDisciplinaryAgent to optionally augment deterministic outputs
with LLM-generated insights.

Design principles
-----------------
* **Opt-in** – every agent defaults to ``use_llm=False`` so the deterministic
  path is always available, even without an API key.
* **Graceful degradation** – if the ``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY``
  env-var is absent, the client returns ``None`` from every call instead of
  raising.  Callers treat ``None`` as "no LLM augmentation available".
* **Thin wrapper** – the client does not cache, batch, or retry on its own.
  Error handling returns ``None`` so callers stay simple.
* **Thread-safe** – the singleton client is lazily initialised behind a lock.
* **Model selection** – defaults to ``openai/gpt-4o-mini`` (cheap, fast) but
  callers can override via the ``model`` parameter.

Environment variables
---------------------
Either of the following is accepted (``OPENROUTER_API_KEY`` takes precedence
when ``OPENAI_BASE_URL`` points to OpenRouter):

    OPENAI_API_KEY          – standard OpenAI key (used as Bearer token)
    OPENROUTER_API_KEY      – OpenRouter-specific key (same format)

    OPENAI_BASE_URL         – API base, defaults to https://openrouter.ai/api/v1
    SUB_TEAM_LLM_MODEL      – override the default model
"""

from __future__ import annotations

__all__ = ["llm_complete", "llm_available", "reset_client"]

import logging
import os
import threading
from typing import Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of openai — avoids a hard dependency at import time
# ---------------------------------------------------------------------------

# None = not checked yet, True = available, False = permanently unavailable
_openai_available: Optional[bool] = None
_client = None                        # openai.OpenAI instance (created once)
_lock = threading.Lock()


def _get_client():
    """Return a cached OpenAI-compatible client, or None if unavailable."""
    global _openai_available, _client

    # Fast path: already determined unavailable
    if _openai_available is False:
        return None

    # Fast path: already initialised
    if _client is not None:
        return _client

    # Slow path: initialise under lock (double-checked locking)
    with _lock:
        # Re-check after acquiring lock
        if _openai_available is False:
            return None
        if _client is not None:
            return _client

        # Resolve API key (OPENROUTER_API_KEY preferred, fallback to OPENAI_API_KEY)
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            _openai_available = False
            return None

        base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
        )

        try:
            from openai import OpenAI  # type: ignore[import-untyped]
            _client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=30.0,
                max_retries=2,
            )
            _openai_available = True
            return _client
        except ImportError:
            _openai_available = False
            return None


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def llm_complete(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Optional[str]:
    """
    Send a chat-completion request and return the assistant's text.

    Parameters
    ----------
    system_prompt : str
        Role/context instructions for the model.
    user_prompt : str
        The actual request.
    model : str, optional
        OpenRouter model identifier.  Defaults to ``SUB_TEAM_LLM_MODEL`` env
        var, or ``openai/gpt-4o-mini``.
    max_tokens : int
        Maximum tokens in the completion (default 1024).
    temperature : float
        Sampling temperature (default 0.2 for reproducibility).

    Returns
    -------
    str or None
        Assistant's reply text, or ``None`` if the client is unavailable or
        an error occurs.
    """
    # Input validation
    if not system_prompt or not user_prompt:
        return None

    client = _get_client()
    if client is None:
        return None

    # Read model lazily so env var changes after import are respected
    chosen_model = model or os.environ.get("SUB_TEAM_LLM_MODEL", _DEFAULT_MODEL)

    try:
        response = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not response.choices:
            _log.debug("LLM returned empty choices for model=%s", chosen_model)
            return None
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception:  # noqa: BLE001  — degrade gracefully
        _log.debug("LLM call failed for model=%s", chosen_model, exc_info=True)
        return None


def llm_available() -> bool:
    """Return True if an LLM client can be constructed (API key present)."""
    return _get_client() is not None


def reset_client() -> None:
    """
    Reset the cached client (useful in tests to inject env-var changes).
    """
    global _openai_available, _client
    with _lock:
        _openai_available = None
        _client = None
