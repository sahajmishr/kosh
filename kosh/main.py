"""
Kosh — Day 1.

Goal: a dumb proxy that works. Non-streaming only.

What this deliberately does NOT do yet:
  - no streaming      (Day 2 — SSE passthrough, the hard part)
  - no retries        (Day 3)
  - no cache          (Day 6)
  - no ledger         (Day 7)
  - no request shaping(Day 5 — the differentiator)

The only job today: accept POST /v1/chat/completions and return exactly what
BharatCode would have returned, so that pointing an OpenAI SDK at Kosh works.
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

load_dotenv()

UPSTREAM_BASE = os.getenv("BHARATCODE_BASE_URL", "https://bharatcode.ai/api/model/v1")
UPSTREAM_URL = f"{UPSTREAM_BASE}/chat/completions"
DEFAULT_MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")

# Separate timeouts matter: connect fast, read slow (a 27B model can think).
TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ONE client for the whole app lifetime. Creating a client per request
    # leaks connections and kills performance — a classic asyncio mistake.
    app.state.upstream = httpx.AsyncClient(timeout=TIMEOUT)
    yield
    await app.state.upstream.aclose()


app = FastAPI(title="Kosh", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_URL, "model": DEFAULT_MODEL}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"

    # 1. Auth — we pass the caller's own BharatCode key straight through.
    #    Kosh never owns a key. This keeps it safe to host later.
    api_key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization: Bearer <BHARATCODE_API_KEY>",
        )

    # 2. Read the body as raw JSON. Day 1 does not validate or reshape it.
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON")

    # Streaming lands on Day 2. Failing loudly beats failing silently.
    if body.get("stream"):
        raise HTTPException(
            status_code=501,
            detail="stream=true is not implemented yet (Day 2: SSE passthrough)",
        )

    body.setdefault("model", DEFAULT_MODEL)

    # 3. Call upstream.
    client: httpx.AsyncClient = request.app.state.upstream
    try:
        upstream_response = await client.post(
            UPSTREAM_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream unreachable: {exc!r}")

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    # 4. Pass the upstream status code through unchanged.
    #    Never paper over a 429/503 — that is exactly the signal Day 3 acts on.
    if upstream_response.status_code != 200:
        return JSONResponse(
            status_code=upstream_response.status_code,
            content={
                "error": "upstream_error",
                "upstream_status": upstream_response.status_code,
                "kosh_request_id": request_id,
                "body": upstream_response.text[:2000],
            },
        )

    payload = upstream_response.json()

    # 5. Minimal observability. The real ledger starts Day 7.
    print(
        f"[{request_id}] model={body.get('model')} "
        f"status={upstream_response.status_code} "
        f"latency_ms={elapsed_ms} "
        f"usage={payload.get('usage', {})}"
    )

    return JSONResponse(
        content=payload,
        headers={
            "x-kosh-request-id": request_id,
            "x-kosh-latency-ms": str(elapsed_ms),
        },
    )