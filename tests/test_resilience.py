"""
Day 3 fault-injection tests — pure logic, no network, no API key.

These run in milliseconds and are among the most impressive things in the
repo: they prove resilience behaviour without ever touching the shared GPU.

    pytest                          (from project root)
    pytest tests/test_resilience.py -v
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from kosh.resilience import (
    FATAL_STATUS,
    MAX_DELAY,
    RETRYABLE_STATUS,
    CircuitBreaker,
    backoff_delay,
    open_with_retry,
    parse_retry_after,
)


# ------------------------------------------------------------- sleep fakes ---
#
# asyncio.sleep is *awaited*, so the replacement must be a coroutine function.
# A plain `lambda d: None` returns None and blows up on `await None`.


async def _no_sleep(delay: float) -> None:
    """Async no-op standing in for asyncio.sleep, so retries run instantly."""
    return None


def _patch_sleep(monkeypatch, sink: list[float]) -> None:
    """Replace asyncio.sleep with a coroutine that records every delay."""

    async def fake_sleep(delay: float) -> None:
        sink.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


# ---------------------------------------------------------------- backoff ---

def test_backoff_is_always_within_bounds():
    for attempt in range(6):
        for _ in range(200):
            delay = backoff_delay(attempt)
            assert 0.0 <= delay <= MAX_DELAY


def test_backoff_grows_exponentially():
    """Ceilings must double: 2, 4, 8, then capped at 16."""
    ceilings = [max(backoff_delay(a) for _ in range(400)) for a in range(5)]
    assert ceilings[0] <= 2.0
    assert ceilings[1] > ceilings[0]
    assert ceilings[2] > ceilings[1]
    assert ceilings[3] <= MAX_DELAY
    assert ceilings[4] <= MAX_DELAY, "must stay capped, never grow past 16"


def test_full_jitter_actually_desynchronises():
    """Two clients rejected at the same instant must not retry together.

    This is the whole point of full jitter: without it the herd reassembles
    on every retry wave.
    """
    delays = {backoff_delay(2) for _ in range(50)}
    assert len(delays) > 45, "jitter is not spreading retries"


# ----------------------------------------------------------- retry-after ---

def test_parse_retry_after():
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after("-3") == 0.0, "negative must clamp to zero"
    assert parse_retry_after(None) is None
    assert parse_retry_after("nonsense") is None
    # An HTTP-date form is not supported; returning None means "use backoff".
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


# ---------------------------------------------------------------- breaker ---

def test_breaker_starts_closed():
    breaker = CircuitBreaker()
    assert breaker.allows() is True
    assert breaker.state == "closed"


def test_breaker_opens_after_threshold():
    breaker = CircuitBreaker(threshold=3)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allows() is False


def test_breaker_stays_closed_below_threshold():
    breaker = CircuitBreaker(threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    assert breaker.allows() is True


def test_breaker_half_opens_after_cooldown():
    breaker = CircuitBreaker(threshold=1, reset_after=0.0)
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allows() is True          # cooldown elapsed -> one probe
    assert breaker.state == "half_open"


def test_breaker_reopens_if_probe_fails():
    breaker = CircuitBreaker(threshold=1, reset_after=0.0)
    breaker.record_failure()
    assert breaker.allows() is True
    assert breaker.state == "half_open"
    breaker.record_failure()                 # the probe failed
    assert breaker.state == "open"


def test_breaker_closes_on_success():
    breaker = CircuitBreaker(threshold=1)
    breaker.record_failure()
    assert breaker.state == "open"
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.failures == 0


def test_breaker_closes_after_successful_probe():
    breaker = CircuitBreaker(threshold=1, reset_after=0.0)
    breaker.record_failure()
    assert breaker.allows() is True
    assert breaker.state == "half_open"
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.opened_count == 1


# --------------------------------------------------------- classification ---

def test_status_classification_is_disjoint():
    assert not (RETRYABLE_STATUS & FATAL_STATUS)
    assert 402 in FATAL_STATUS, "a payment error must never be retried"
    assert 503 in RETRYABLE_STATUS, "GPU busy is the key one to retry"
    assert 429 in RETRYABLE_STATUS
    assert 400 in FATAL_STATUS


# ------------------------------------------------- retry behaviour (fake) ---

class _FakeResponse:
    def __init__(self, status: int, body: bytes = b'{"error":"busy"}', headers=None):
        self.status_code = status
        self.headers: dict[str, str] = headers or {}
        self._body = body
        self.closed = False

    async def aread(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    """Returns a scripted sequence of upstream responses."""

    def __init__(self, statuses: list[int], headers: dict[str, str] | None = None):
        self.statuses = statuses
        self.default_headers = headers
        self.calls = 0
        self.responses: list[_FakeResponse] = []

    def build_request(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return object()

    async def send(self, request, stream: bool = False):  # noqa: ANN001
        self.calls += 1
        status = self.statuses[min(self.calls - 1, len(self.statuses) - 1)]
        response = _FakeResponse(status, headers=dict(self.default_headers or {}))
        self.responses.append(response)
        return response


@pytest.mark.asyncio
async def test_retries_503_then_succeeds(monkeypatch):
    """Two busy responses, then success. Three attempts, two backoffs."""
    client = _FakeClient([503, 503, 200])
    sleeps: list[float] = []
    _patch_sleep(monkeypatch, sleeps)

    result = await open_with_retry(client, "http://x", {}, {}, CircuitBreaker())

    assert result.status == 200
    assert result.attempts == 3
    assert client.calls == 3
    assert len(sleeps) == 2, "back off between attempts, never after the last"


@pytest.mark.asyncio
async def test_does_not_retry_402(monkeypatch):
    """A payment error is final. Retrying burns capacity for nothing."""
    client = _FakeClient([402])
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    result = await open_with_retry(client, "http://x", {}, {}, CircuitBreaker())

    assert result.status == 402
    assert client.calls == 1, "402 must fail fast on the first attempt"


@pytest.mark.asyncio
async def test_402_does_not_open_the_breaker(monkeypatch):
    """Our bad request is not evidence the upstream is unhealthy."""
    client = _FakeClient([402])
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    breaker = CircuitBreaker(threshold=1)

    await open_with_retry(client, "http://x", {}, {}, breaker)

    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_circuit_open_short_circuits_without_calling(monkeypatch):
    client = _FakeClient([200])
    breaker = CircuitBreaker(threshold=1)
    breaker.record_failure()
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    result = await open_with_retry(client, "http://x", {}, {}, breaker)

    assert result.error == "circuit_open"
    assert result.status == 503
    assert client.calls == 0, "an open breaker must not touch the upstream at all"


@pytest.mark.asyncio
async def test_every_failed_response_is_closed(monkeypatch):
    """Leaked responses are leaked provider slots. Assert none leak."""
    client = _FakeClient([503, 503, 503, 503])
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    result = await open_with_retry(client, "http://x", {}, {}, CircuitBreaker())

    assert result.response is None
    assert len(client.responses) == 4
    assert all(r.closed for r in client.responses), "every failed response must be closed"


@pytest.mark.asyncio
async def test_successful_response_is_left_open_for_caller(monkeypatch):
    """On success the caller owns the response. We must not close it."""
    client = _FakeClient([200])
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    result = await open_with_retry(client, "http://x", {}, {}, CircuitBreaker())

    assert result.response is not None
    assert result.response.closed is False, "caller must be able to stream it"


@pytest.mark.asyncio
async def test_honours_retry_after_header(monkeypatch):
    """When the server tells us how long to wait, we wait exactly that."""
    client = _FakeClient([429, 200], headers={"retry-after": "3"})
    sleeps: list[float] = []
    _patch_sleep(monkeypatch, sleeps)

    result = await open_with_retry(client, "http://x", {}, {}, CircuitBreaker())

    assert result.status == 200
    assert 3.0 in sleeps, "must honour Retry-After over its own backoff guess"


@pytest.mark.asyncio
async def test_no_pointless_sleep_after_last_attempt(monkeypatch):
    """Wasting a sleep after the final attempt only delays the error."""
    client = _FakeClient([503])
    sleeps: list[float] = []
    _patch_sleep(monkeypatch, sleeps)

    result = await open_with_retry(
        client, "http://x", {}, {}, CircuitBreaker(), max_attempts=2
    )

    assert result.response is None
    assert client.calls == 2
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_timeout_is_retried(monkeypatch):
    client = _FakeClient([200])
    calls = {"n": 0}

    async def raise_once(request, stream: bool = False):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.TimeoutException("too slow")
        client.calls += 1
        return _FakeResponse(200)

    client.send = raise_once
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    result = await open_with_retry(client, "http://x", {}, {}, CircuitBreaker())

    assert result.status == 200
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts(monkeypatch):
    client = _FakeClient([503])
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    result = await open_with_retry(
        client, "http://x", {}, {}, CircuitBreaker(), max_attempts=3
    )

    assert result.response is None
    assert client.calls == 3
    assert result.attempts == 3
    assert result.error == "upstream_error"
