"""
Kosh — Day 4. Drop-in, provider-agnostic gateway with resilience + replay.

Progress:
  Day 1  non-streaming proxy                        [done]
  Day 2  SSE streaming passthrough                  [done]
  Day 3  retry + jittered backoff + circuit breaker [done]  22 tests
  Day 4  replay mode (KOSH_REPLAY=1)                [done]
  Day 5  request shaping / prefix hygiene           <- the differentiator
  Day 6  exact cache (SHA-256 canonical)
  Day 7  SQLite ledger + /stats dashboard
  Day 8  MVP FREEZE + benchmark
  Day 9-11 stretch, docs, submit

THE ONE-LINE DEMO
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")

With KOSH_REPLAY=1 that line works with no API key, no account, no network
and no GPU. Anyone can evaluate Kosh in under a minute.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from kosh.replay import PROFILES, ReplayConfig, ReplayTransport, config_from_env
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


def create_app(replay: ReplayConfig | None = None) -> FastAPI:
    """Build the app.

    A factory rather than a module-level singleton, so tests and benchmarks
    can create an app with a fresh circuit breaker and a specific fault
    profile. Breaker state is global by design within one app — but two
    benchmark scenarios must not contaminate each other.
    """
    if replay is None:
        replay = config_from_env()

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
        # Replay REPLACES THE TRANSPORT, never the logic: retry, jitter and
        # the breaker all run unchanged, they just never reach the network.
        app.state.replay = replay
        app.state.transport = (
            ReplayTransport(replay) if replay.enabled else app.state.upstream
        )
        yield
        await app.state.upstream.aclose()

    app = FastAPI(title="Kosh", version="0.4.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        breaker: CircuitBreaker = app.state.breaker
        return {
            "status": "ok",
            "upstream": UPSTREAM_URL,
            "model": DEFAULT_MODEL,
            "key_mode": "shared" if SHARED_UPSTREAM_KEY else "passthrough",
            "replay": {
                "enabled": replay.enabled,
                "profile": replay.profile,
                "fixtures": len(replay.fixtures),
                "speed": replay.speed,
            },
            "breaker": {
                "state": breaker.state,
                "failures": breaker.failures,
                "opened_count": breaker.opened_count,
            },
        }

    @app.get("/")
    async def root():
        """Enough information to use Kosh without reading the README."""
        return {
            "name": "Kosh",
            "version": "0.4.0",
            "endpoints": ["/v1/chat/completions", "/health", "/fixtures"],
            "drop_in": "OpenAI(base_url='http://localhost:8000/v1', api_key='...')",
            "replay": {
                "enabled": replay.enabled,
                "profile": replay.profile,
                "available_profiles": sorted(PROFILES),
            },
        }

    @app.get("/fixtures")
    async def fixtures():
        """What replay is serving. Makes reproducibility inspectable."""
        if not replay.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "profile": replay.profile,
            "script": replay.script.outcomes,
            "count": len(replay.fixtures),
            "speed": replay.speed,
            "samples": [
                {"chars": len(f.text), "ttft_ms": round(f.ttft_ms, 1)}
                for f in replay.fixtures[:5]
            ],
        }

    def _resolve_key(request: Request) -> str:
        # Replay needs no key at all — that is the point of it.
        if replay.enabled:
            return "replay"
        if SHARED_UPSTREAM_KEY:
            return SHARED_UPSTREAM_KEY
        key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not key:
            raise HTTPException(
                status_code=401, detail="Missing Authorization: Bearer <API key>"
            )
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
        client = request.app.state.transport
        breaker: CircuitBreaker = request.app.state.breaker

        if body.get("stream"):
            return await stream_upstream(
                client, UPSTREAM_URL, body, headers, request_id, breaker
            )

        return await collect_upstream(
            client, UPSTREAM_URL, body, headers, request_id, breaker
        )

    return app


app = create_app()
