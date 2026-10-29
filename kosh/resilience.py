"""
Kosh — resilience (Day 3).

Two things go wrong against a shared GPU:

1. It returns 503 when busy and 429 when we are too fast. If every client
   retries instantly, they retry in lockstep and make the queue worse.
   That is a thundering herd, and it is the classic way a degraded service
   tips into an outage.

2. It can be unavailable for a sustained window. Hammering it anyway wastes
   our own slots and only delays the error the user will get regardless.

FIX 1: exponential backoff with FULL jitter, so retries desynchronise.
FIX 2: a circuit breaker that stops calling a failing upstream for a
       cooldown, then probes once (half-open) before resuming.

THE RULE THAT SHAPES THIS FILE
Retry only BEFORE any bytes are consumed. Once a stream has started, you
cannot retry it without regenerating every token already produced. So both
the streaming and non-streaming paths retry the *open*, never the body.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import httpx

# Transient, server-side, or explicitly rate-limited -> worth retrying.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Our request is wrong. Retrying only burns shared capacity.
FATAL_STATUS = {400, 401, 402, 403, 404, 405, 406, 413, 422}

MAX_ATTEMPTS = 4
BASE_DELAY = 2.0      # ceilings: 2s -> 4s -> 8s -> 16s
MAX_DELAY = 16.0


def backoff_delay(attempt: int, base: float = BASE_DELAY, cap: float = MAX_DELAY) -> float:
    """Full jitter: uniform(0, min(cap, base * 2**attempt)).

    Full jitter rather than equal jitter: it spreads retries across the whole
    window instead of clustering them at the ceiling. That desynchronisation
    is the entire point — it is what stops a thundering herd.
    """
    ceiling = min(cap, base * (2**attempt))
    return random.uniform(0.0, ceiling)


def parse_retry_after(value: str | None) -> float | None:
    """Honour Retry-After when the server sends it. Never guess instead."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


@dataclass
class CircuitBreaker:
    """closed -> open -> half_open -> closed.

    closed    : traffic flows
    open      : everything fails fast, no upstream calls at all
    half_open : one probe allowed; success closes, failure reopens
    """

    threshold: int = 5
    reset_after: float = 30.0
    failures: int = 0
    opened_at: float | None = None
    state: str = "closed"
    opened_count: int = 0

    def allows(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            elapsed = time.monotonic() - (self.opened_at or 0.0)
            if elapsed >= self.reset_after:
                self.state = "half_open"   # allow exactly one probe
                return True
            return False
        return True  # half_open: the single probe

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None
        self.state = "closed"

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.threshold:
            if self.state != "open":
                self.opened_count += 1
            self.state = "open"
            self.opened_at = time.monotonic()


@dataclass
class UpstreamResult:
    response: httpx.Response | None
    status: int
    attempts: int
    error: str | None
    body: bytes = b""


async def open_with_retry(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    breaker: CircuitBreaker,
    max_attempts: int = MAX_ATTEMPTS,
) -> UpstreamResult:
    """Open an upstream request with breaker + jittered backoff.

    Returns an OPEN, UNREAD streaming response on success — the caller owns
    closing it. On failure the response is None and `body` holds whatever the
    upstream last said.
    """
    last_status = 0
    last_body = b""
    last_error = "upstream_failed"

    for attempt in range(max_attempts):
        if not breaker.allows():
            return UpstreamResult(None, 503, attempt, "circuit_open", b"circuit breaker open")

        if attempt > 0:
            # Sleep between attempts only — never after the final one.
            await asyncio.sleep(backoff_delay(attempt - 1))

        try:
            request = client.build_request("POST", url, json=body, headers=headers)
            response = await client.send(request, stream=True)
        except httpx.TimeoutException:
            breaker.record_failure()
            last_status, last_body, last_error = 504, b"upstream timeout", "timeout"
            continue
        except httpx.RequestError as exc:
            breaker.record_failure()
            last_status, last_body, last_error = 502, repr(exc).encode(), "unreachable"
            continue

        if response.status_code == 200:
            breaker.record_success()
            # Deliberately NOT closed: the caller streams or reads it.
            return UpstreamResult(response, 200, attempt + 1, None)

        # Non-200: read and close before we can retry.
        raw = await response.aread()
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        await response.aclose()
        last_status, last_body, last_error = response.status_code, raw, "upstream_error"

        if response.status_code not in RETRYABLE_STATUS:
            # Our fault, not theirs — do not punish the breaker.
            breaker.record_success()
            break

        breaker.record_failure()

        if retry_after is not None:
            # The server told us exactly how long. Respect it over our guess.
            await asyncio.sleep(min(retry_after, MAX_DELAY))

    return UpstreamResult(None, last_status, max_attempts, last_error, last_body)
