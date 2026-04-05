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
* **Model selection** – defaults to ``openai/gpt-4o-mini`` (cheap, fast) but
  callers can override via the ``model`` parameter.

Environment variables
---------------------
Either of the following is accepted (``OPENAI_API_KEY`` takes precedence):

    OPENAI_API_KEY          – standard OpenAI key (used as Bearer token)
    OPENROUTER_API_KEY      – OpenRouter-specific key (same format)

    OPENAI_BASE_URL         – API base, defaults to https://openrouter.ai/api/v1
    SUB_TEAM_LLM_MODEL      – override the default model
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy import of openai — avoids a hard dependency at import time
# ---------------------------------------------------------------------------

_openai_available: Optional[bool] = None
_client = None                        # openai.OpenAI instance (created once)


def _get_client():
    """Return a cached OpenAI-compatible client, or None if unavailable."""
    global _openai_available, _client

    if _openai_available is False:
        return None

    if _client is not None:
        return _client

    # Resolve API key (OPENAI_API_KEY takes precedence over OPENROUTER_API_KEY)
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not api_key:
        _openai_available = False
        return None

    base_url = os.environ.get(
        "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
    )

    try:
        from openai import OpenAI  # type: ignore[import]
        _client = OpenAI(api_key=api_key, base_url=base_url)
        _openai_available = True
        return _client
    except ImportError:
        _openai_available = False
        return None


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.environ.get("SUB_TEAM_LLM_MODEL", "openai/gpt-4o-mini")


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
    client = _get_client()
    if client is None:
        return None

    chosen_model = model or _DEFAULT_MODEL

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
        return response.choices[0].message.content
    except Exception:  # noqa: BLE001  — degrade silently
        return None


def llm_available() -> bool:
    """Return True if an LLM client can be constructed (API key present)."""
    return _get_client() is not None


def reset_client() -> None:
    """
    Reset the cached client (useful in tests to inject env-var changes).
    """
    global _openai_available, _client
    _openai_available = None
    _client = None
