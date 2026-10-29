"""
Kosh — Day 3. Drop-in, provider-agnostic gateway with resilience.

Progress:
  Day 1  non-streaming proxy                        [done]
  Day 2  SSE streaming passthrough                  [done]
  Day 3  retry + jittered backoff + circuit breaker [done]
  Day 4  replay mode (KOSH_REPLAY=1)                [next]
  Day 5  request shaping / prefix hygiene           <- the differentiator
  Day 6  exact cache (SHA-256 canonical)
  Day 7  SQLite ledger + /stats dashboard
  Day 8  MVP FREEZE + benchmark
  Day 9-11 stretch, docs, submit
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from kosh.resilience import CircuitBreaker
from kosh.upstream import collect_upstream, stream_upstream

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

TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ONE client for the whole app lifetime. Creating a client per request
    # leaks connections — the classic asyncio mistake.
    app.state.upstream = httpx.AsyncClient(timeout=TIMEOUT)
    # Shared across all requests: the breaker is global state by design,
    # because the upstream's health is a property of the upstream, not of
    # any single caller.
    app.state.breaker = CircuitBreaker(
        threshold=int(os.getenv("KOSH_BREAKER_THRESHOLD", "5")),
        reset_after=float(os.getenv("KOSH_BREAKER_RESET", "30")),
    )
    yield
    await app.state.upstream.aclose()


app = FastAPI(title="Kosh", version="0.3.0", lifespan=lifespan)


@app.get("/health")
async def health():
    breaker: CircuitBreaker = app.state.breaker
    return {
        "status": "ok",
        "upstream": UPSTREAM_URL,
        "model": DEFAULT_MODEL,
        "key_mode": "shared" if SHARED_UPSTREAM_KEY else "passthrough",
        "breaker": {
            "state": breaker.state,
            "failures": breaker.failures,
            "opened_count": breaker.opened_count,
        },
    }


def _resolve_key(request: Request) -> str:
    if SHARED_UPSTREAM_KEY:
        return SHARED_UPSTREAM_KEY
    key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not key:
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <API key>")
    return key


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
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
    breaker: CircuitBreaker = request.app.state.breaker

    if body.get("stream"):
        return await stream_upstream(client, UPSTREAM_URL, body, headers, request_id, breaker)

    return await collect_upstream(client, UPSTREAM_URL, body, headers, request_id, breaker)
