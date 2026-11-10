"""
Kosh — admission control (Stage 5). A token bucket.

Rejects a request BEFORE it touches the upstream, with an honest 429 and a
Retry-After, rather than letting it queue behind eight other requests and
timing out 180 seconds later.

WHY A TOKEN BUCKET AND NOT A FIXED WINDOW
-----------------------------------------
A fixed window ("100 per minute") lets a client spend 100 in the last second
of one window and 100 in the first second of the next: 200 requests in two
seconds, while the counter claims they are within limits. A token bucket
refills continuously, so the ceiling is real.

THE ONE INTERESTING DESIGN DECISION
-----------------------------------
Checking and consuming are separate operations, so that a CACHE HIT COSTS
NOTHING. A hit never reaches the GPU, so charging it against the budget
would penalise exactly the behaviour we want to encourage. The sequence is:

    allowed = bucket.check(key)     # do we have headroom?
    ...
    bucket.consume(key)             # only if the request actually went out

This is the first place where two Kosh features compose into something
neither does alone: caching makes your budget go further.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucket:
    """Per-key token bucket. Thread-safe: handlers run on a thread pool."""

    def __init__(self, rate_per_minute: float = 0.0, burst: float = 0.0):
        self.rate_per_minute = rate_per_minute
        self.burst = burst
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        # Counters. Only mutate under _lock — every read path is also a
        # write path (lazy refill), so there is no unlocked read to protect.
        self.checked = 0
        self.admitted = 0
        self.rejected = 0

    @property
    def enabled(self) -> bool:
        return self.rate_per_minute > 0

    @staticmethod
    def _refill_rate(rate_per_minute: float) -> float:
        return rate_per_minute / 60.0

    def _peek(self, key: str) -> tuple[_Bucket, float]:
        """Return the bucket after applying refill, plus seconds to 1 token."""
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self.burst, updated=now)
            self._buckets[key] = bucket

        elapsed = now - bucket.updated
        if elapsed > 0:
            bucket.tokens = min(
                self.burst,
                bucket.tokens + elapsed * self._refill_rate(self.rate_per_minute),
            )
            bucket.updated = now

        if bucket.tokens >= 1.0:
            return bucket, 0.0
        deficit = 1.0 - bucket.tokens
        return bucket, deficit / self._refill_rate(self.rate_per_minute)

    def check(self, key: str) -> tuple[bool, float]:
        """Do we have headroom? Returns (allowed, retry_after_seconds).

        Does not consume. Call consume() only if the request really went out.
        """
        if not self.enabled:
            return True, 0.0
        with self._lock:
            _bucket, wait = self._peek(key)
            allowed = wait <= 0.0
            self.checked += 1
            if allowed:
                self.admitted += 1
            else:
                self.rejected += 1
            return allowed, round(wait, 3)

    def consume(self, key: str, tokens: float = 1.0) -> None:
        """Spend from the bucket. Only for requests that reached upstream."""
        if not self.enabled:
            return
        with self._lock:
            bucket, _wait = self._peek(key)
            bucket.tokens = max(0.0, bucket.tokens - tokens)

    def remaining(self, key: str) -> float:
        if not self.enabled:
            return float("inf")
        with self._lock:
            bucket, _wait = self._peek(key)
            return round(bucket.tokens, 2)

    def stats(self) -> dict:
        """Live counters for /stats. Zero-cost: reads under the lock."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "rate_per_minute": self.rate_per_minute,
                "burst": self.burst,
                "checked": self.checked,
                "admitted": self.admitted,
                "rejected": self.rejected,
                "keys_tracked": len(self._buckets),
            }

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)


def bucket_from_env() -> TokenBucket:
    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, ""))
        except (TypeError, ValueError):
            return default

    rate = _float("KOSH_RATE_LIMIT", 0.0)
    burst = _float("KOSH_RATE_BURST", 0.0) or max(1.0, rate)
    return TokenBucket(rate_per_minute=rate, burst=burst)
