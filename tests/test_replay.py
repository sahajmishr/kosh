"""
Day 4 replay tests — no network, no API key, deterministic.

These prove two different things:

1. Replay itself works: it is deterministic, it serves recorded bytes, and
   it injects the failure modes BharatCode actually exhibits.

2. Replay drives the REAL resilience code. ReplayTransport replaces the
   transport, so retry, jitter and the circuit breaker run unchanged. That
   is what makes these tests worth anything — replaying around the logic
   would test nothing.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from kosh.main import create_app
from kosh.replay import (
    BUSY,
    DISCONNECT,
    OK,
    PROFILES,
    RATE_LIMITED,
    TRUNCATED,
    FaultScript,
    ReplayConfig,
    ReplayTransport,
    _builtin_fixtures,
    canonical_key,
    truncate_lines,
)

# Tests must not depend on whatever happens to be in the fixtures directory,
# so we pin the built-in corpus.
FIXTURES = _builtin_fixtures()


def make_config(outcomes: list[str] | None = None, speed: float = 0.0) -> ReplayConfig:
    return ReplayConfig(
        enabled=True,
        fixtures=FIXTURES,
        script=FaultScript(outcomes=list(outcomes or [])),
        profile="test",
        speed=speed,
    )


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch):
    """Neuter retry sleeps so the suite stays fast.

    These tests assert control flow — attempt counts, breaker state, call
    counts — never timing. Real backoff would add ~30 s for nothing. Timing
    itself is covered in test_resilience.py.
    """
    async def _sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)


@pytest.fixture
async def client_factory():
    """Yield a factory building an app + httpx client over ASGI transport."""
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


async def _post(http: httpx.AsyncClient, prompt: str, stream: bool = True, **extra):
    body = {"model": "qwen-3.8-27b", "messages": [{"role": "user", "content": prompt}]}
    body.update(extra)
    if stream:
        return await http.post("/v1/chat/completions", json={**body, "stream": True})
    return await http.post("/v1/chat/completions", json=body)


# --------------------------------------------------------- determinism ------

def test_same_prompt_maps_to_same_fixture():
    a = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    b = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    assert canonical_key(a) == canonical_key(b)


def test_different_prompts_map_differently():
    a = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    b = {"model": "m", "messages": [{"role": "user", "content": "goodbye"}]}
    assert canonical_key(a) != canonical_key(b)


def test_key_ignores_irrelevant_fields():
    """A differing temperature changes the answer, but the model choice and
    the messages are what identify the request."""
    a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True}
    b = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    assert canonical_key(a) == canonical_key(b)


# -------------------------------------------------------- transport ---------

@pytest.mark.asyncio
async def test_replay_transport_never_uses_the_network():
    transport = ReplayTransport(make_config([OK]))
    assert not isinstance(transport, httpx.AsyncClient)
    request = transport.build_request("POST", "http://x", json={}, headers={})
    response = await transport.send(request, stream=True)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_transport_reports_every_outcome():
    """The outcomes list is what the benchmark reads to prove injection."""
    transport = ReplayTransport(make_config([BUSY, RATE_LIMITED, OK]))
    request = transport.build_request("POST", "http://x", json={}, headers={})
    for _ in range(3):
        await transport.send(request, stream=True)
    assert transport.outcomes == [BUSY, RATE_LIMITED, OK]
    assert transport.calls == 3


@pytest.mark.asyncio
async def test_timeout_outcome_raises_like_a_real_timeout():
    from kosh.replay import TIMEOUT

    transport = ReplayTransport(make_config([TIMEOUT]))
    request = transport.build_request("POST", "http://x", json={}, headers={})
    with pytest.raises(httpx.TimeoutException):
        await transport.send(request, stream=True)


# -------------------------------------------------- replay through the app --

@pytest.mark.asyncio
async def test_replay_needs_no_api_key(client_factory):
    """The whole point: anyone can run this with no account."""
    http, app = client_factory(make_config([OK]))
    async with app.router.lifespan_context(app):
        response = await _post(http, "hello")
        assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_replay_is_deterministic(client_factory):
    http, app = client_factory(make_config([OK, OK]))
    async with app.router.lifespan_context(app):
        first = await _post(http, "the same prompt twice")
        second = await _post(http, "the same prompt twice")
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


@pytest.mark.asyncio
async def test_replayed_bytes_match_the_recorded_lines(client_factory):
    """Contract test: replay emits the recorded SSE byte for byte."""
    config = make_config([OK])
    body = {
        "model": "qwen-3.8-27b",
        "messages": [{"role": "user", "content": "contract check"}],
        "stream": True,
    }
    # Ask replay which fixture it will serve, then assert it serves exactly it.
    expected = config.select(body)

    http, app = client_factory(config)
    async with app.router.lifespan_context(app):
        response = await http.post("/v1/chat/completions", json=body)

    delivered = [line for line in response.text.splitlines() if line.startswith("data:")]
    assert delivered == expected.lines


@pytest.mark.asyncio
async def test_non_streaming_replay_returns_valid_openai_json(client_factory):
    http, app = client_factory(make_config([OK]))
    async with app.router.lifespan_context(app):
        response = await _post(http, "hello", stream=False)

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"]


# ------------------------------------------- replay drives real resilience --

@pytest.mark.asyncio
async def test_retry_recovers_where_a_single_attempt_would_fail(client_factory):
    """Two 503s then success: one attempt fails, Kosh succeeds on the third."""
    http, app = client_factory(make_config([BUSY, BUSY, OK]))
    async with app.router.lifespan_context(app):
        response = await _post(http, "retry me")

    assert response.status_code == 200
    assert int(response.headers["x-kosh-attempts"]) == 3


@pytest.mark.asyncio
async def test_sustained_outage_opens_the_breaker(client_factory, monkeypatch):
    """Three failures with threshold 3 -> open, then zero further calls."""
    monkeypatch.setenv("KOSH_BREAKER_THRESHOLD", "3")
    http, app = client_factory(make_config([BUSY]))
    async with app.router.lifespan_context(app):
        await _post(http, "first")
        breaker = app.state.breaker
        transport = app.state.transport
        assert breaker.state == "open"
        assert breaker.opened_count == 1
        calls_after_first = transport.calls

        second = await _post(http, "second")

    assert second.status_code == 503
    assert second.json()["error"] == "circuit_open"
    assert transport.calls == calls_after_first, "an open breaker must call nothing"


@pytest.mark.asyncio
async def test_open_breaker_advertises_retry_after(client_factory, monkeypatch):
    monkeypatch.setenv("KOSH_BREAKER_THRESHOLD", "1")
    http, app = client_factory(make_config([BUSY, BUSY]))
    async with app.router.lifespan_context(app):
        await _post(http, "trip it")
        response = await _post(http, "now blocked")

    assert response.status_code == 503
    assert response.headers.get("retry-after") == "30"


# --------------------------------------------------- nasty failure modes ----

@pytest.mark.asyncio
async def test_truncation_is_a_200_that_lies(client_factory):
    """HTTP 200, but finish_reason 'length'. Status-code checks miss this."""
    http, app = client_factory(make_config([TRUNCATED]))
    async with app.router.lifespan_context(app):
        response = await _post(http, "truncate me")

    assert response.status_code == 200, "this is exactly the trap"
    chunks = [
        json.loads(line[5:])
        for line in response.text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]
    finish = [c for c in chunks if c["choices"][0].get("finish_reason")]
    assert finish, "stream must terminate"
    assert finish[-1]["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_disconnect_is_reported_not_silently_swallowed(client_factory):
    """Client must learn the stream died, not hang until its own timeout."""
    http, app = client_factory(make_config([DISCONNECT]))
    async with app.router.lifespan_context(app):
        response = await _post(http, "die on me")

    assert response.status_code == 200
    body = response.text
    assert "upstream_disconnected" in body
    assert "data: [DONE]" in body, "stream must still be closed cleanly"


# ------------------------------------------------------------- profiles -----

def test_profiles_are_all_known_outcomes():
    known = {OK, BUSY, RATE_LIMITED, "500", "timeout", TRUNCATED, DISCONNECT}
    for name, script in PROFILES.items():
        unknown = set(script) - known
        assert not unknown, f"profile {name} has unknown outcomes {unknown}"


def test_outage_profile_never_succeeds():
    assert PROFILES["outage"] == [BUSY]
    assert OK not in PROFILES["outage"]


def test_truncate_lines_keeps_some_content():
    lines = FIXTURES[0].lines
    cut = truncate_lines(lines)
    assert len(cut) < len(lines)
    assert cut[-1] == "data: [DONE]"
    assert "length" in cut[-2]

