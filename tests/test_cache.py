"""
Day 6 exact-cache tests — no network, no API key.

The dangerous failure mode for a cache is a FALSE HIT: returning an answer
to a question nobody asked. These tests are mostly about proving that
cannot happen, because a silent wrong answer is far worse than a miss.
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from kosh.cache import (
    CacheEntry,
    ExactCache,
    cache_from_env,
    estimate_tokens,
    parse_sse,
)
from kosh.main import create_app
from kosh.replay import (
    BUSY,
    OK,
    TRUNCATED,
    FaultScript,
    ReplayConfig,
    _builtin_fixtures,
)

FIXTURES = _builtin_fixtures()


def make_config(outcomes: list[str] | None = None) -> ReplayConfig:
    return ReplayConfig(
        enabled=True,
        fixtures=FIXTURES,
        script=FaultScript(outcomes=list(outcomes or [])),
        profile="test",
        speed=0.0,
    )


def body(text: str = "hello", **extra) -> dict:
    base = {
        "model": "qwen-3.8-27b",
        "messages": [{"role": "user", "content": text}],
    }
    base.update(extra)
    return base


# -------------------------------------------------------------- keying ------

def test_stream_flag_is_not_part_of_the_key():
    """SSE and JSON are transports for the same answer — one cache entry."""
    assert ExactCache.key_for(body("hi", stream=True)) == \
           ExactCache.key_for(body("hi", stream=False))


def test_different_prompts_get_different_keys():
    assert ExactCache.key_for(body("hi")) != ExactCache.key_for(body("bye"))


def test_identical_bodies_get_identical_keys():
    assert ExactCache.key_for(body("hi")) == ExactCache.key_for(body("hi"))


def test_temperature_is_part_of_the_key():
    assert ExactCache.key_for(body("hi", temperature=0.2)) != \
           ExactCache.key_for(body("hi", temperature=0.9))


# -------------------------------------------------------------- parsing -----

def test_parse_sse_extracts_text_and_finish_reason():
    raw = b"".join(
        (line + "\n\n").encode() for line in [
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )
    lines, text, reason = parse_sse(raw)
    assert text == "Hello world"
    assert reason == "stop"
    assert len(lines) == 4


def test_parse_sse_detects_truncation():
    raw = b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
    _lines, _text, reason = parse_sse(raw)
    assert reason == "length"


def test_parse_sse_flags_injected_error_events():
    raw = b'data: {"error":"upstream_disconnected"}\n\ndata: [DONE]\n\n'
    _lines, _text, reason = parse_sse(raw)
    assert reason == "error"


def test_parse_sse_reports_unterminated_streams():
    raw = b'data: {"choices":[{"delta":{"content":"cut off"},"finish_reason":null}]}\n\n'
    _lines, _text, reason = parse_sse(raw)
    assert reason == "", "no finish_reason means we must not cache it"


# ------------------------------------------------------------- storing ------

def test_store_then_get():
    cache = ExactCache()
    key = ExactCache.key_for(body("hi"))
    cache.store(key, ['data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                      "data: [DONE]"], "answer", "stop")
    entry = cache.get(key)
    assert entry is not None
    assert entry.text == "answer"


def test_truncated_responses_are_never_cached():
    """The single most important correctness rule in this file."""
    cache = ExactCache()
    key = ExactCache.key_for(body("hi"))
    stored = cache.store(key, ["data: ..."], "partial answer", "length")
    assert stored is False
    assert cache.get(key) is None
    assert cache.stats.rejects == 1


def test_error_responses_are_never_cached():
    cache = ExactCache()
    assert cache.store("k", ["data: ..."], "", "error") is False
    assert cache.stats.rejects == 1


def test_unterminated_responses_are_never_cached():
    cache = ExactCache()
    assert cache.store("k", ["data: ..."], "text", "") is False


def test_empty_responses_are_never_cached():
    cache = ExactCache()
    assert cache.store("k", [], "text", "stop") is False
    assert cache.store("k", ["data: ..."], "", "stop") is False


# ---------------------------------------------------------------- stats -----

def test_hit_rate_and_token_savings():
    cache = ExactCache()
    key = ExactCache.key_for(body("hi"))
    cache.store(key, ["data: [DONE]"], "x" * 400, "stop")
    cache.get(key)          # hit
    cache.get(key)          # hit
    cache.get("missing")    # miss
    assert cache.stats.hits == 2
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == pytest.approx(66.7, abs=0.1)
    assert cache.stats.tokens_saved == 200  # 400 chars / 2 hits


def test_estimate_tokens_is_rough_and_documented():
    assert estimate_tokens("a" * 400) == 100


# -------------------------------------------------------------- eviction ----

def test_lru_eviction_respects_max_entries():
    cache = ExactCache(max_entries=3)
    keys = []
    for index in range(4):
        key = ExactCache.key_for(body(f"prompt-{index}"))
        keys.append(key)
        cache.store(key, ["data: [DONE]"], f"answer-{index}", "stop")
    assert len(cache) == 3
    assert cache.get(keys[0]) is None, "oldest entry must be evicted"
    assert cache.stats.evictions == 1


def test_recency_refreshes_on_hit():
    cache = ExactCache(max_entries=2)
    a = ExactCache.key_for(body("a"))
    b = ExactCache.key_for(body("b"))
    c = ExactCache.key_for(body("c"))
    cache.store(a, ["data: [DONE]"], "a", "stop")
    cache.store(b, ["data: [DONE]"], "b", "stop")
    cache.get(a)                                        # a becomes most recent
    cache.store(c, ["data: [DONE]"], "c", "stop")       # should evict b, not a
    assert cache.get(b) is None
    assert cache.get(a) is not None


def test_ttl_expiry():
    """Age the entry manually rather than sleeping — deterministic."""
    cache = ExactCache(ttl_seconds=60.0)
    key = ExactCache.key_for(body("hi"))
    cache.store(key, ["data: [DONE]"], "answer", "stop")
    cache._data[key].created_at -= 120.0          # backdate past the TTL
    assert cache.get(key) is None
    assert cache.stats.expiries == 1


def test_ttl_of_zero_means_never_expire():
    cache = ExactCache(ttl_seconds=0.0)
    key = ExactCache.key_for(body("hi"))
    cache.store(key, ["data: [DONE]"], "answer", "stop")
    cache._data[key].created_at -= 10_000.0
    assert cache.get(key) is not None


# ------------------------------------------------------------- gating -------

def test_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("KOSH_CACHE", "0")
    assert ExactCache().cacheable(body("hi")) is False


def test_high_temperature_can_be_excluded(monkeypatch):
    monkeypatch.setenv("KOSH_CACHE_MAX_TEMPERATURE", "0.0")
    cache = cache_from_env()
    assert cache.cacheable(body("hi", temperature=0.0)) is True
    assert cache.cacheable(body("hi", temperature=0.9)) is False


def test_cacheable_reads_env(monkeypatch):
    monkeypatch.setenv("KOSH_CACHE", "1")
    assert ExactCache().cacheable(body("hi")) is True


# --------------------------------------------------- through the app -------

@pytest.fixture
async def client_factory():
    created: list[httpx.AsyncClient] = []

    def build(config: ReplayConfig) -> tuple[httpx.AsyncClient, FastAPI]:
        app = create_app(config)
        http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        created.append(http)
        return http, app

    yield build
    for http in created:
        await http.aclose()


@pytest.mark.asyncio
async def test_second_identical_request_is_a_cache_hit(client_factory, monkeypatch):
    monkeypatch.setenv("KOSH_CACHE", "1")
    http, app = client_factory(make_config([OK, OK, OK]))
    payload = body("cache me please")

    async with app.router.lifespan_context(app):
        first = await http.post("/v1/chat/completions", json=payload)
        second = await http.post("/v1/chat/completions", json=payload)

    assert first.headers["x-kosh-cache"] == "miss"
    assert second.headers["x-kosh-cache"] == "hit"
    assert second.json() == first.json(), "a hit must be byte-identical"


@pytest.mark.asyncio
async def test_cache_hit_makes_no_upstream_call(client_factory, monkeypatch):
    monkeypatch.setenv("KOSH_CACHE", "1")
    http, app = client_factory(make_config([OK, OK]))
    payload = body("no second call")

    async with app.router.lifespan_context(app):
        await http.post("/v1/chat/completions", json=payload)
        transport = app.state.transport
        calls_after_first = transport.calls
        await http.post("/v1/chat/completions", json=payload)

    assert transport.calls == calls_after_first, "hit must not touch the upstream"


@pytest.mark.asyncio
async def test_streaming_second_request_is_a_hit(client_factory, monkeypatch):
    monkeypatch.setenv("KOSH_CACHE", "1")
    http, app = client_factory(make_config([OK, OK, OK]))
    payload = body("stream me twice", stream=True)

    async with app.router.lifespan_context(app):
        first = await http.post("/v1/chat/completions", json=payload)
        second = await http.post("/v1/chat/completions", json=payload)

    assert first.headers["x-kosh-cache"] == "miss"
    assert second.headers["x-kosh-cache"] == "hit"
    assert second.text == first.text


@pytest.mark.asyncio
async def test_truncated_upstream_is_not_cached(client_factory, monkeypatch):
    """Replay a truncated response: it must not poison the cache."""
    monkeypatch.setenv("KOSH_CACHE", "1")
    http, app = client_factory(make_config([TRUNCATED, TRUNCATED]))
    payload = body("truncate me", stream=True)

    async with app.router.lifespan_context(app):
        first = await http.post("/v1/chat/completions", json=payload)
        second = await http.post("/v1/chat/completions", json=payload)

    assert first.status_code == 200
    assert second.headers["x-kosh-cache"] == "miss", "a truncation must not be served"
    assert app.state.cache.stats.rejects >= 1


@pytest.mark.asyncio
async def test_errors_are_not_cached(client_factory, monkeypatch):
    monkeypatch.setenv("KOSH_CACHE", "1")
    http, app = client_factory(make_config([BUSY, BUSY]))
    payload = body("this will fail")

    async with app.router.lifespan_context(app):
        await http.post("/v1/chat/completions", json=payload)

    assert app.state.cache.stats.stores == 0
    assert len(app.state.cache) == 0


@pytest.mark.asyncio
async def test_stats_endpoint_reports_hit_rate(client_factory, monkeypatch):
    monkeypatch.setenv("KOSH_CACHE", "1")
    http, app = client_factory(make_config([OK, OK, OK]))
    payload = body("stats please")

    async with app.router.lifespan_context(app):
        await http.post("/v1/chat/completions", json=payload)
        await http.post("/v1/chat/completions", json=payload)
        stats = (await http.get("/stats")).json()

    assert stats["cache"]["hits"] == 1
    assert stats["cache"]["misses"] == 1
    assert stats["cache"]["hit_rate_pct"] == 50.0
