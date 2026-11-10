"""
Kosh — upstream core (Day 6).

Both paths retry the *open* (via resilience.open_with_retry) and never the
body. Once tokens are flowing, a retry would regenerate everything already
produced — doubling cost for the user and load for the shared GPU.

Streaming responses are now TEE'd into the cache: every chunk goes to the
client the instant it arrives, and simultaneously into a buffer. The entry
is stored only after the stream completes cleanly. We never buffer-then-
forward, because that would destroy streaming to save a round trip.
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from kosh.cache import ExactCache, parse_sse
from kosh.ledger import RequestRecord
from kosh.semantic import SemanticCache
from kosh.resilience import CircuitBreaker, UpstreamResult, open_with_retry

# Stop proxies and browsers from buffering an SSE stream.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _error_response(result: UpstreamResult, request_id: str) -> JSONResponse:
    """Map a failed upstream attempt onto an honest HTTP response."""
    if result.error == "circuit_open":
        return JSONResponse(
            status_code=503,
            content={
                "error": "circuit_open",
                "detail": "Upstream is failing; Kosh is not calling it right now.",
                "kosh_request_id": request_id,
            },
            headers={"Retry-After": "30"},
        )

    return JSONResponse(
        status_code=result.status or 502,
        content={
            "error": result.error or "upstream_error",
            "upstream_status": result.status,
            "attempts": result.attempts,
            "kosh_request_id": request_id,
            "body": result.body.decode("utf-8", "replace")[:2000],
        },
    )


def _sse_from_lines(lines: list[str]) -> bytes:
    return b"".join((line + "\n\n").encode("utf-8") for line in lines)


def _cache_hit_stream(entry) -> StreamingResponse:
    """Replay a cached response as SSE, byte for byte."""
    payload = _sse_from_lines(entry.lines)
    return StreamingResponse(
        iter([payload]),
        media_type="text/event-stream",
        headers={
            **SSE_HEADERS,
            "x-kosh-cache": "hit",
            "x-kosh-attempts": "0",
        },
    )


async def stream_upstream(
    client,
    url: str,
    body: dict,
    headers: dict,
    request_id: str,
    breaker: CircuitBreaker,
    extra_headers: dict[str, str] | None = None,
    cache: ExactCache | None = None,
    cache_key: str | None = None,
    record: RequestRecord | None = None,
    semantic: SemanticCache | None = None,
    prompt_text: str = "",
) -> StreamingResponse | JSONResponse:
    """Proxy a streaming chat-completions request, byte for byte."""
    # --- cache lookup (before we touch the upstream at all) -----------------
    if cache is not None and cache_key:
        entry = cache.get(cache_key)
        if entry is not None:
            if record is not None:
                record.cache = "hit"
                record.completion_tokens = entry.completion_tokens
                record.saved_tokens = entry.completion_tokens
                record.attempts = 0
                record.latency_ms = 0.0
            response = _cache_hit_stream(entry)
            for name, value in (extra_headers or {}).items():
                response.headers[name] = value
            response.headers["x-kosh-cache"] = "hit"
            return response

    # --- near-duplicate lookup (only if the exact cache missed) -------------
    if semantic is not None and semantic.enabled and prompt_text:
        # exclude_key exists so we never serve an entry the exact cache
        # already let expire. With the exact cache off there is no TTL to
        # honour, so a byte-identical repeat is a genuine duplicate and
        # should hit.
        near = semantic.find(
            prompt_text, exclude_key=cache_key if cache is not None else None
        )
        if near is not None:
            if record is not None:
                record.cache = "near"
                record.completion_tokens = max(0, len(near.text) // 4)
                record.saved_tokens = max(0, len(near.text) // 4)
                record.attempts = 0
                record.latency_ms = 0.0
            response = _cache_hit_stream(near)
            for name, value in (extra_headers or {}).items():
                response.headers[name] = value
            response.headers["x-kosh-cache"] = "near"
            return response

    result = await open_with_retry(client, url, body, headers, breaker)
    if result.response is None:
        return _error_response(result, request_id)

    upstream = result.response

    async def generate() -> AsyncIterator[bytes]:
        buffer = bytearray()
        started = time.perf_counter()
        saw_bytes = False
        completed = False
        try:
            # Raw byte passthrough. We do NOT re-frame the SSE: the provider
            # already emits `data: {...}\n\n`, keepalives and `data: [DONE]`.
            # Parsing and rebuilding would only introduce bugs.
            async for chunk in upstream.aiter_bytes():
                if chunk:
                    saw_bytes = True
                # Tee on cache_key, NOT on `cache`: the near-duplicate store
                # needs the same bytes. Gating this on the exact cache meant
                # KOSH_CACHE=0 left the buffer empty and every streaming
                # store was rejected with finish_reason="".
                if cache_key:
                    buffer.extend(chunk)      # client first, always
                yield chunk
            completed = True
        except httpx.HTTPError as exc:
            # Upstream died mid-stream. Tell the client instead of leaving it
            # waiting on a stream that will never finish. Never cache a
            # fragment — we have no idea if it is complete.
            if cache is not None:
                cache.stats.rejects += 1
            payload = json.dumps(
                {
                    "error": "upstream_disconnected",
                    "detail": repr(exc),
                    "kosh_request_id": request_id,
                }
            )
            if record is not None:
                record.error = "upstream_disconnected"
                record.latency_ms = (time.perf_counter() - started) * 1000
            yield f"data: {payload}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        finally:
            # The store lives here, NOT in an `else:` clause, and this is the
            # single most important comment in the streaming path.
            #
            # An `else:` block only runs if the try body completes without an
            # exception. When the ASGI layer decides the response is finished
            # it closes this generator, which throws GeneratorExit at the
            # last `yield`. GeneratorExit is not an httpx.HTTPError, so it
            # propagates, `else:` is skipped, and the store never runs — no
            # streamed response is ever cached. Whether that happens depends
            # on platform timing, so the same code cached everything on one
            # machine and nothing on another.
            #
            # A stream is safe to cache only when we know it terminated, and
            # there are two ways to know:
            #   1. the upstream iterator finished normally (completed), or
            #   2. the buffer already carries the provider's terminal frame.
            # Case 2 is what covers the closed-early path.
            raw = bytes(buffer)
            terminated = completed or b"data: [DONE]" in raw

            if saw_bytes and terminated:
                lines, text, finish_reason = parse_sse(raw)
                if cache_key:
                    if cache is not None:
                        cache.store(cache_key, lines, text, finish_reason)
                    # Independent of the exact cache on purpose: KOSH_CACHE=0
                    # must not silently disable near-duplicate storage too.
                    if semantic is not None and semantic.enabled and prompt_text:
                        semantic.store(cache_key, prompt_text, lines, finish_reason)
                if record is not None:
                    record.completion_tokens = max(0, len(text) // 4)
                    record.latency_ms = (time.perf_counter() - started) * 1000
                    if finish_reason in ("", "error"):
                        record.error = ("incomplete" if not finish_reason
                                        else finish_reason)
            elif not saw_bytes and record is not None:
                # A 200 with zero bytes is not a clean completion. It arrives
                # as an empty body and, because the buffer never filled, as a
                # cache miss on every later identical request — which looks
                # exactly like "caching is broken". Record the real fault.
                record.error = "empty_stream"
            # Runs on success, on error AND on client disconnect. Without this
            # we leak a provider concurrency slot on every abandoned request.
            await upstream.aclose()

    response = StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            **SSE_HEADERS,
            "x-kosh-request-id": request_id,
            "x-kosh-attempts": str(result.attempts),
            "x-kosh-cache": "miss",
            **(extra_headers or {}),
        },
    )
    return response


async def collect_upstream(
    client,
    url: str,
    body: dict,
    headers: dict,
    request_id: str,
    breaker: CircuitBreaker,
    extra_headers: dict[str, str] | None = None,
    cache: ExactCache | None = None,
    cache_key: str | None = None,
    record: RequestRecord | None = None,
    semantic: SemanticCache | None = None,
    prompt_text: str = "",
) -> JSONResponse:
    """Non-streaming path: open with retry, then read the whole body."""
    if cache is not None and cache_key:
        entry = cache.get(cache_key)
        if entry is not None:
            if record is not None:
                record.cache = "hit"
                record.completion_tokens = entry.completion_tokens
                record.saved_tokens = entry.completion_tokens
                record.attempts = 0
                record.latency_ms = 0.0
            # Serve the ORIGINAL body verbatim. A cache hit must be
            # indistinguishable from the response that filled the cache —
            # same id, same timestamps, same usage. Anything less breaks the
            # drop-in contract for anyone logging response ids.
            if entry.payload is None:
                payload = {
                    "id": f"chatcmpl-kosh-{entry.key[:12]}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": entry.text},
                        "finish_reason": entry.finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": entry.completion_tokens,
                        "total_tokens": entry.completion_tokens,
                    },
                }
            else:
                payload = entry.payload
            return JSONResponse(
                content=payload,
                headers={
                    "x-kosh-cache": "hit",
                    "x-kosh-latency-ms": "0.0",
                    "x-kosh-attempts": "0",
                    **(extra_headers or {}),
                },
            )

    # --- near-duplicate lookup (only if the exact cache missed) -------------
    if semantic is not None and semantic.enabled and prompt_text:
        # exclude_key exists so we never serve an entry the exact cache
        # already let expire. With the exact cache off there is no TTL to
        # honour, so a byte-identical repeat is a genuine duplicate and
        # should hit.
        near = semantic.find(
            prompt_text, exclude_key=cache_key if cache is not None else None
        )
        if near is not None:
            if record is not None:
                record.cache = "near"
                record.completion_tokens = max(0, len(near.text) // 4)
                record.saved_tokens = max(0, len(near.text) // 4)
                record.attempts = 0
                record.latency_ms = 0.0
            return JSONResponse(
                content={
                    "id": f"chatcmpl-kosh-near-{near.key[:12]}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": near.text},
                        "finish_reason": near.finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": max(0, len(near.text) // 4),
                        "total_tokens": max(0, len(near.text) // 4),
                    },
                },
                headers={
                    "x-kosh-cache": "near",
                    "x-kosh-latency-ms": "0.0",
                    "x-kosh-attempts": "0",
                    **(extra_headers or {}),
                },
            )

    started = time.perf_counter()
    result = await open_with_retry(client, url, body, headers, breaker)
    if result.response is None:
        if record is not None:
            record.status = result.status or 502
            record.attempts = result.attempts
            record.error = result.error
        return _error_response(result, request_id)

    upstream = result.response
    try:
        raw = await upstream.aread()
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": "upstream_disconnected",
                "detail": repr(exc),
                "kosh_request_id": request_id,
            },
        )
    finally:
        await upstream.aclose()

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=502,
            content={
                "error": "invalid_upstream_json",
                "kosh_request_id": request_id,
                "body": raw.decode("utf-8", "replace")[:2000],
            },
        )

    # Cache only what we can prove is complete.
    # Guarded on cache_key alone, NOT on `cache`: the semantic cache is an
    # independent store and must keep working when KOSH_CACHE=0.
    if cache_key:
        finish_reason = ""
        choices = payload.get("choices") or []
        if choices:
            finish_reason = choices[0].get("finish_reason") or ""
        if finish_reason in ("stop", "tool_calls", "function_call"):
            text = ""
            if choices and isinstance(choices[0].get("message"), dict):
                text = choices[0]["message"].get("content") or ""
            replay = [
                "data: " + json.dumps({
                    "id": payload.get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": payload.get("created", 0),
                    "model": payload.get("model", ""),
                    "choices": [{"index": 0, "delta": {"content": text},
                                 "finish_reason": None}],
                }, separators=(",", ":")),
                "data: " + json.dumps({
                    "id": payload.get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": payload.get("created", 0),
                    "model": payload.get("model", ""),
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": finish_reason}],
                }, separators=(",", ":")),
                "data: [DONE]",
            ]
            if cache is not None:
                cache.store(cache_key, replay, text, finish_reason,
                            payload=payload)
            if semantic is not None and semantic.enabled and prompt_text:
                semantic.store(cache_key, prompt_text, replay, finish_reason)

    if record is not None:
        usage = payload.get("usage") or {}
        record.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        record.completion_tokens = int(usage.get("completion_tokens") or 0)
        record.latency_ms = elapsed_ms
        record.attempts = result.attempts
        choices = payload.get("choices") or []
        if choices and choices[0].get("finish_reason") == "length":
            record.error = "truncated"

    return JSONResponse(
        content=payload,
        headers={
            "x-kosh-request-id": request_id,
            "x-kosh-latency-ms": str(elapsed_ms),
            "x-kosh-attempts": str(result.attempts),
            "x-kosh-cache": "miss",
            **(extra_headers or {}),
        },
    )
