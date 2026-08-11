"""OpenAI-compatible client with bounded retries for remote model servers."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import openai


logger = logging.getLogger("osworld_cogagent.client")


class EmptyCompletionError(RuntimeError):
    pass


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content)


def call_openai_compatible(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    resolved_url = base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    resolved_key = api_key or os.getenv("OPENAI_API_KEY") or "dummy"
    timeout = float(os.getenv("OSWORLD_OPENAI_TIMEOUT", "120"))
    retries = max(1, int(os.getenv("OSWORLD_MAX_RETRY_TIMES", "5")))
    client = openai.OpenAI(
        base_url=resolved_url,
        api_key=resolved_key,
        timeout=timeout,
        max_retries=0,
    )

    retryable = tuple(
        item
        for item in (
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "InternalServerError", None),
            EmptyCompletionError,
        )
        if isinstance(item, type)
    )
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            choices = getattr(result, "choices", None)
            if not choices or getattr(choices[0], "message", None) is None:
                raise EmptyCompletionError("Server returned no completion message")
            text = _content_text(getattr(choices[0].message, "content", None)).strip()
            if not text:
                raise EmptyCompletionError("Server returned an empty completion")
            return text
        except retryable as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = min(2 ** (attempt - 1), 8)
            logger.warning(
                "Model request failed (%d/%d): %s; retrying in %ss",
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"Model request failed after {retries} attempts: {last_error}") from last_error
