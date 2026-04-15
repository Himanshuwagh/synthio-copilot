# llm_client.py — Single chat entrypoint for Ollama, OpenAI, or Anthropic.
# Provides both a blocking `chat()` (used for all pipeline steps except the
# final synthesizer) and an async streaming `stream_chat()` generator (used
# by the /chat/stream SSE endpoint to stream the synthesizer token-by-token).

from __future__ import annotations

from typing import AsyncGenerator, Optional

from config import get_llm_settings


def chat(system: str, user: str, max_tokens: Optional[int] = None) -> str:
    s = get_llm_settings()

    if s.provider == "ollama":
        return _chat_ollama(s, system, user, max_tokens)
    if s.provider == "openai":
        return _chat_openai(s, system, user, max_tokens)
    if s.provider == "anthropic":
        return _chat_anthropic(s, system, user, max_tokens)
    return f"[LLM error: unknown provider {s.provider!r}]"


async def stream_chat(
    system: str, user: str, max_tokens: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """Async generator that yields string tokens as they arrive from the LLM."""
    s = get_llm_settings()
    if s.provider == "ollama":
        async for token in _stream_ollama(s, system, user, max_tokens):
            yield token
    elif s.provider == "openai":
        async for token in _stream_openai(s, system, user, max_tokens):
            yield token
    elif s.provider == "anthropic":
        async for token in _stream_anthropic(s, system, user, max_tokens):
            yield token
    else:
        yield f"[LLM error: unknown provider {s.provider!r}]"


def _chat_ollama(s, system: str, user: str, max_tokens: Optional[int]) -> str:
    import ollama

    # Default httpx timeout is unbounded; multi-step agent calls need a long read timeout.
    _timeout = 600.0
    if s.ollama_host:
        client = ollama.Client(host=s.ollama_host, timeout=_timeout)
    else:
        client = ollama.Client(timeout=_timeout)
    opts = {"temperature": s.temperature}
    if max_tokens is not None:
        opts["num_predict"] = max_tokens
    try:
        response = client.chat(
            model=s.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=opts,
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def _chat_openai(s, system: str, user: str, max_tokens: Optional[int]) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        return "[LLM error: openai package not installed. Run: pip install openai]"

    client = OpenAI(api_key=s.openai_api_key)
    try:
        kwargs = {
            "model": s.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": s.temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        r = client.chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def _chat_anthropic(s, system: str, user: str, max_tokens: Optional[int]) -> str:
    try:
        import anthropic
    except ImportError:
        return "[LLM error: anthropic package not installed. Run: pip install anthropic]"

    client = anthropic.Anthropic(api_key=s.anthropic_api_key)
    mt = 8192 if max_tokens is None else max(64, min(max_tokens, 8192))
    try:
        r = client.messages.create(
            model=s.model,
            max_tokens=mt,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=s.temperature,
        )
        parts = []
        for block in r.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts).strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# ── Streaming implementations ─────────────────────────────────────────────────

async def _stream_ollama(s, system: str, user: str, max_tokens: Optional[int]):
    try:
        import ollama
    except ImportError:
        yield "[LLM error: ollama package not installed. Run: pip install ollama]"
        return
    opts = {"temperature": s.temperature}
    if max_tokens is not None:
        opts["num_predict"] = max_tokens
    try:
        host_kwargs = {"host": s.ollama_host} if s.ollama_host else {}
        client = ollama.AsyncClient(timeout=600.0, **host_kwargs)
        # ollama.AsyncClient.chat with stream=True returns an async iterator
        async for chunk in await client.chat(
            model=s.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=opts,
            stream=True,
        ):
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
    except Exception as e:
        yield f"[LLM error: {e}]"


async def _stream_openai(s, system: str, user: str, max_tokens: Optional[int]):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield "[LLM error: openai package not installed. Run: pip install openai]"
        return
    try:
        client = AsyncOpenAI(api_key=s.openai_api_key)
        kwargs = {
            "model": s.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": s.temperature,
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"[LLM error: {e}]"


async def _stream_anthropic(s, system: str, user: str, max_tokens: Optional[int]):
    try:
        import anthropic
    except ImportError:
        yield "[LLM error: anthropic package not installed. Run: pip install anthropic]"
        return
    mt = 8192 if max_tokens is None else max(64, min(max_tokens, 8192))
    try:
        client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key)
        async with client.messages.stream(
            model=s.model,
            max_tokens=mt,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=s.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"[LLM error: {e}]"
