"""
Kosh — drop-in, provider-agnostic efficiency gateway for LLM APIs.

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="bc-...")

That one line is the whole integration. Everything else Kosh does — shaping,
caching, retry, circuit breaking, accounting — happens behind it.

Pipeline (nine stages; seven are live):
    1 auth      2 shape      3 cache      4 route
    5 admit     6 call+retry 7 stream     8 record

THE THESIS
    The cheapest token is the one you never send — and on a shared GPU,
    every token you don't send is latency somebody else gets back.

Version 0.7.0 · BharatCode Build League Round 1
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.background import BackgroundTask

from kosh.cache import ExactCache, cache_from_env
from kosh.dashboard import render as render_dashboard
from kosh.ledger import RequestRecord, ledger_from_env
from kosh.replay import PROFILES, ReplayConfig, ReplayTransport, config_from_env
from kosh.resilience import CircuitBreaker
from kosh.shaping import MODES as SHAPE_MODES
from kosh.shaping import flatten, shape_request
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

# Set -> one shared upstream key. Empty -> caller's key passes through.
SHARED_UPSTREAM_KEY = os.getenv("KOSH_UPSTREAM_KEY", "")

TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)


def _shape_mode() -> str:
    """report by default: measure and tell, never silently rewrite."""
    mode = os.getenv("KOSH_SHAPE", "report").strip().lower()
    if mode not in SHAPE_MODES:
        raise ValueError(f"unknown KOSH_SHAPE {mode!r}; choose from {SHAPE_MODES}")
    return mode


def _estimate_prompt_tokens(body: dict) -> int:
    """Rough, and only used for the savings ledger — never for billing."""
    try:
        return max(0, len(flatten(body.get("messages") or [])) // 4)
    except Exception:
        return 0


def create_app(replay: ReplayConfig | None = None) -> FastAPI:
    """Build the app.

    A factory rather than a module-level singleton, so tests and benchmarks
    can create an app with a fresh circuit breaker and a given fault profile.
    """
    if replay is None:
        replay = config_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ONE client for the whole app lifetime. Creating one per request
        # leaks connections — the classic asyncio mistake.
        app.state.upstream = httpx.AsyncClient(timeout=TIMEOUT)
        # Global by design: upstream health is a property of the upstream,
        # not of any single caller.
        app.state.breaker = CircuitBreaker(
            threshold=int(os.getenv("KOSH_BREAKER_THRESHOLD", "5")),
            reset_after=float(os.getenv("KOSH_BREAKER_RESET", "30")),
        )
        # Replay replaces the TRANSPORT, never the logic: retry, jitter and
        # the breaker all run unchanged, they just never reach the network.
        app.state.replay = replay
        app.state.transport = (
            ReplayTransport(replay) if replay.enabled else app.state.upstream
        )
        app.state.cache = cache_from_env()
        app.state.ledger = ledger_from_env()
        app.state.started = time.time()
        yield
        await app.state.upstream.aclose()
        app.state.ledger.close()

    app = FastAPI(title="Kosh", version="0.7.0", lifespan=lifespan)

    # ------------------------------------------------------------ routes --

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
            },
            "shaping": {"mode": _shape_mode()},
            "breaker": {
                "state": breaker.state,
                "failures": breaker.failures,
                "opened_count": breaker.opened_count,
            },
            "cache": {**app.state.cache.stats.as_dict(),
                      "entries": len(app.state.cache)},
        }

    @app.get("/")
    async def root():
        return {
            "name": "Kosh",
            "version": "0.7.0",
            "endpoints": {
                "/v1/chat/completions": "OpenAI-compatible drop-in",
                "/dashboard": "live HTML dashboard",
                "/stats": "JSON aggregates from the ledger",
                "/health": "breaker, cache and replay state",
                "/fixtures": "what replay is serving",
            },
            "drop_in": "OpenAI(base_url='http://localhost:8000/v1', api_key='...')",
            "replay": {"enabled": replay.enabled,
                       "profiles": sorted(PROFILES)},
        }

    @app.get("/stats")
    async def stats():
        """Aggregates. This is the receipt."""
        return {
            "totals": app.state.ledger.totals(),
            "latency_uncached": app.state.ledger.latency_percentiles(),
            "cache": app.state.cache.stats.as_dict(),
            "breaker": {
                "state": app.state.breaker.state,
                "opened_count": app.state.breaker.opened_count,
            },
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        ledger = app.state.ledger
        return render_dashboard(
            totals=ledger.totals(),
            latency=ledger.latency_percentiles(),
            recent=ledger.recent(30),
            cache_stats=app.state.cache.stats.as_dict(),
            uptime_s=time.time() - getattr(app.state, "started", time.time()),
        )

    @app.get("/fixtures")
    async def fixtures():
        if not replay.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "profile": replay.profile,
            "script": replay.script.outcomes,
            "count": len(replay.fixtures),
            "speed": replay.speed,
        }

    def _resolve_key(request: Request) -> str:
        if replay.enabled:
            return "replay"          # replay needs no key at all
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
        cache: ExactCache = request.app.state.cache
        ledger = request.app.state.ledger

        # STAGE 2 — shape. Report mode only measures; it never rewrites.
        shaped, report = shape_request(body, _shape_mode())
        extra = report.as_headers()

        # STAGE 3 — exact cache, keyed on the SHAPED request.
        use_cache = cache.cacheable(shaped)
        cache_key = ExactCache.key_for(shaped) if use_cache else None

        record = RequestRecord(
            request_id=request_id,
            model=shaped.get("model", ""),
            prompt_tokens=_estimate_prompt_tokens(shaped),
            prefix_verdict=report.verdict,
            prefix_cacheable=report.cacheable_pct,
        )

        # Written after the response completes — for streaming, that means
        # after the last token, so latency and token counts are real.
        background = BackgroundTask(ledger.record, record)

        if shaped.get("stream"):
            response = await stream_upstream(
                client, UPSTREAM_URL, shaped, headers, request_id, breaker,
                extra, cache, cache_key, record,
            )
        else:
            response = await collect_upstream(
                client, UPSTREAM_URL, shaped, headers, request_id, breaker,
                extra, cache, cache_key, record,
            )

        if hasattr(response, "background"):
            response.background = background
        return response

    return app


app = create_app()
