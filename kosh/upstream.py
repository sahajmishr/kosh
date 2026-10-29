"""
Kosh — upstream core (Day 3).

Both paths retry the *open* (via resilience.open_with_retry) and never the
body. Once tokens are flowing, a retry would regenerate everything already
produced — doubling cost for the user and load for the shared GPU.
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

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


async def stream_upstream(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    request_id: str,
    breaker: CircuitBreaker,
) -> StreamingResponse | JSONResponse:
    """Proxy a streaming chat-completions request, byte for byte."""
    result = await open_with_retry(client, url, body, headers, breaker)
    if result.response is None:
        return _error_response(result, request_id)

    upstream = result.response

    async def generate() -> AsyncIterator[bytes]:
        try:
            # Raw byte passthrough. We do NOT re-frame the SSE: the provider
            # already emits `data: {...}\n\n`, keepalives and `data: [DONE]`.
            # Parsing and rebuilding would only introduce bugs.
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except httpx.HTTPError as exc:
            # Upstream died mid-stream. Tell the client instead of leaving it
            # waiting on a stream that will never finish.
            payload = json.dumps(
                {
                    "error": "upstream_disconnected",
                    "detail": repr(exc),
                    "kosh_request_id": request_id,
                }
            )
            yield f"data: {payload}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        finally:
            # Runs on success, on error AND on client disconnect. Without this
            # we leak a provider concurrency slot on every abandoned request.
            await upstream.aclose()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            **SSE_HEADERS,
            "x-kosh-request-id": request_id,
            "x-kosh-attempts": str(result.attempts),
        },
    )


async def collect_upstream(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    request_id: str,
    breaker: CircuitBreaker,
) -> JSONResponse:
    """Non-streaming path: open with retry, then read the whole body."""
    started = time.perf_counter()
    result = await open_with_retry(client, url, body, headers, breaker)
    if result.response is None:
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

    return JSONResponse(
        content=payload,
        headers={
            "x-kosh-request-id": request_id,
            "x-kosh-latency-ms": str(elapsed_ms),
            "x-kosh-attempts": str(result.attempts),
        },
    )
