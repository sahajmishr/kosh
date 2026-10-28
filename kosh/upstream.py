"""
Kosh — upstream streaming core (Day 2).

The one genuinely hard part of a gateway: moving a server-sent-events stream
from the upstream provider to our client without corrupting it, without
leaking connections, and without hiding errors.

Three rules this file exists to enforce:

1. Decide the HTTP status BEFORE the first byte is streamed.
   Once a StreamingResponse has started, the status code is already on the
   wire — you cannot change it. So a 402/429/503 from the provider must be
   caught and returned as JSON, never as a stream.

2. Never let a stream outlive its client.
   If our client disconnects mid-generation, Starlette cancels this generator.
   The `finally` block closes the upstream response so the provider's slot is
   released. Skipping this leaks provider concurrency — a real, documented
   failure mode on shared-GPU inference servers.

3. Never let a dead upstream hang our client.
   If the provider drops mid-stream we emit a terminal error event plus
   [DONE], so the client's SDK raises instead of waiting forever.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

# Headers that stop proxies and browsers from buffering an SSE stream.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def stream_upstream(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    request_id: str,
) -> StreamingResponse | JSONResponse:
    """Proxy a streaming chat-completions request, byte for byte."""

    request = client.build_request("POST", url, json=body, headers=headers)

    # --- open the upstream stream -----------------------------------------
    try:
        upstream = await client.send(request, stream=True)
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": "upstream_timeout", "kosh_request_id": request_id},
        )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": "upstream_unreachable",
                "detail": repr(exc),
                "kosh_request_id": request_id,
            },
        )

    # --- rule 1: status before streaming ----------------------------------
    if upstream.status_code != 200:
        raw = await upstream.aread()
        await upstream.aclose()
        return JSONResponse(
            status_code=upstream.status_code,
            content={
                "error": "upstream_error",
                "upstream_status": upstream.status_code,
                "kosh_request_id": request_id,
                "body": raw.decode("utf-8", "replace")[:2000],
            },
        )

    async def generate() -> AsyncIterator[bytes]:
        chunks = 0
        try:
            # Raw byte passthrough. We deliberately do NOT re-frame the SSE:
            # the provider already emits `data: {...}\n\n`, keepalive comments
            # and the `data: [DONE]\n\n` terminator. Parsing and rebuilding it
            # would only introduce bugs.
            async for chunk in upstream.aiter_bytes():
                chunks += 1
                yield chunk
        except httpx.HTTPError as exc:
            # --- rule 3: upstream died mid-stream -------------------------
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
            # --- rule 2: runs on success, error AND client disconnect ------
            await upstream.aclose()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "x-kosh-request-id": request_id},
    )
