"""
Kosh — Day 2. Drop-in, provider-agnostic gateway with SSE streaming.

Progress:
  Day 1  non-streaming proxy                       [done]
  Day 2  SSE streaming passthrough                 [done]
  Day 3  retry + jittered backoff + circuit breaker  [next]
  Day 4  replay mode (KOSH_REPLAY=1)
  Day 5  request shaping / prefix hygiene           <- the differentiator
  Day 6  exact cache (SHA-256 canonical)
  Day 7  SQLite ledger + /stats dashboard
  Day 8  MVP FREEZE + benchmark
  Day 9-11 stretch, docs, submit
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

from kosh.upstream import stream_upstream

load_dotenv()

# --- Provider config -------------------------------------------------------
# Any OpenAI-compatible endpoint:
#   BharatCode : https://bharatcode.ai/api/model/v1   (default)
#   Ollama     : http://localhost:11434/v1
#   Groq       : https://api.groq.com/openai/v1
UPSTREAM_BASE = os.getenv("KOSH_UPSTREAM_BASE", "https://bharatcode.ai/api/model/v1")
UPSTREAM_URL = f"{UPSTREAM_BASE}/chat/completions"
DEFAULT_MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")

# Set -> one shared upstream key (dev / fallback mode).
# Empty -> caller's own key is passed through (BharatCode mode).
SHARED_UPSTREAM_KEY = os.getenv("KOSH_UPSTREAM_KEY", "")

# Separate timeouts: connect fast, read slow. A 27B model can think, and a
# long stream legitimately stays open for minutes.
TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ONE client for the whole app lifetime. Creating a client per request
    # leaks connections — the classic asyncio mistake.
    app.state.upstream = httpx.AsyncClient(timeout=TIMEOUT)
    yield
    await app.state.upstream.aclose()


app = FastAPI(title="Kosh", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "upstream": UPSTREAM_URL,
        "model": DEFAULT_MODEL,
        "key_mode": "shared" if SHARED_UPSTREAM_KEY else "passthrough",
    }


def _resolve_key(request: Request) -> str:
    """Shared key if configured, otherwise the caller's own key."""
    if SHARED_UPSTREAM_KEY:
        return SHARED_UPSTREAM_KEY
    key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not key:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization: Bearer <API key>",
        )
    return key


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"

    api_key = _resolve_key(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON")

    body.setdefault("model", DEFAULT_MODEL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    client: httpx.AsyncClient = request.app.state.upstream

    # --- streaming path (Day 2) -------------------------------------------
    if body.get("stream"):
        print(f"[{request_id}] stream=true model={body.get('model')}")
        return await stream_upstream(client, UPSTREAM_URL, body, headers, request_id)

    # --- non-streaming path (Day 1) ----------------------------------------
    try:
        upstream_response = await client.post(UPSTREAM_URL, json=body, headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream unreachable: {exc!r}")

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    # Pass upstream status through untouched. Day 3 acts on 429/503.
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
    print(
        f"[{request_id}] model={body.get('model')} "
        f"latency_ms={elapsed_ms} usage={payload.get('usage', {})}"
    )
    return JSONResponse(
        content=payload,
        headers={
            "x-kosh-request-id": request_id,
            "x-kosh-latency-ms": str(elapsed_ms),
        },
    )
