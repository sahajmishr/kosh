"""
Kosh — exact cache (Day 6).

The cheapest token is the one you never send. A cache is the bluntest
possible way to achieve that, so it has to be exactly right: a false hit
returns an answer to a question nobody asked.

KEYED ON THE CANONICAL REQUEST
------------------------------
The key is the SHA-256 of the *shaped* request — model, messages,
max_tokens, temperature. Two requests that differ only in whitespace or
key order are the same request and must hit.

CRUCially, `stream` is NOT part of the key. Whether the caller wants SSE or
JSON is a transport choice; the answer is identical. Caching them
separately would halve the hit rate for no reason.

WHAT MUST NEVER BE CACHED
-------------------------
1. Truncated responses (finish_reason "length"). The response is a 200, but
   the answer is incomplete. Caching it would serve a wrong answer forever.
2. Errors. Obviously.
3. Streams that died mid-flight. Whatever we accumulated is a fragment.
4. Anything we could not parse. When in doubt, do not cache.

A cache that is sometimes wrong is worse than no cache, because the failure
is silent. Every rejection is counted in `stats.rejects` so the behaviour is
visible rather than assumed.

STREAMING
---------
We do not buffer the response before forwarding it — that would destroy
streaming, which is the whole point of the SSE work on Day 2. Instead we
tee: every chunk goes to the client immediately AND into a buffer. The entry
is stored only after the stream completes cleanly.

So the first caller always pays full latency and gets a miss. That is
correct: you cannot serve from a cache you have not filled yet.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable


# ----------------------------------------------------------------- entry ---

@dataclass
class CacheEntry:
    """A completed response, stored as the exact SSE lines that produced it."""

    key: str
    lines: list[str]
    text: str
    finish_reason: str
    created_at: float
    hits: int = 0
    completion_tokens: int = 0
    # The original non-streaming JSON. Kept verbatim so a hit is byte-identical
    # to the response that filled the cache — that is the drop-in contract.
    payload: dict | None = None

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    rejects: int = 0          # completed, but deliberately not cached
    tokens_saved: int = 0
    evictions: int = 0
    expiries: int = 0

    @property
    def hit_rate(self) -> float:
        if self.lookups == 0:
            return 0.0
        return round(100.0 * self.hits / self.lookups, 1)

    def as_dict(self) -> dict:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "rejects": self.rejects,
            "tokens_saved": self.tokens_saved,
            "evictions": self.evictions,
            "expiries": self.expiries,
            "hit_rate_pct": self.hit_rate,
        }


# ------------------------------------------------------------------ parse ---

def parse_sse(raw: bytes) -> tuple[list[str], str, str]:
    """Split a streamed body into (SSE lines, assistant text, finish_reason).

    Returns finish_reason "" when the stream never terminated cleanly, which
    is how we detect a fragment that must not be cached.
    """
    body = raw.decode("utf-8", "replace")
    lines = [line for line in body.splitlines() if line.strip()]
    text_parts: list[str] = []
    finish_reason = ""

    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        # An error event injected by our own upstream layer.
        if "error" in chunk:
            return lines, "", "error"
        for choice in chunk.get("choices") or []:
            delta_content = (choice.get("delta") or {}).get("content")
            if delta_content:
                text_parts.append(delta_content)
            reason = choice.get("finish_reason")
            if reason:
                finish_reason = reason

    return lines, "".join(text_parts), finish_reason


def estimate_tokens(text: str) -> int:
    """Rough count, used only for savings reporting — never for billing."""
    return max(0, len(text) // 4)


# ------------------------------------------------------------------ cache ---

class ExactCache:
    """Bounded LRU cache of completed responses."""

    def __init__(
        self,
        max_entries: int = 1000,
        ttl_seconds: float = 3600.0,
        max_temperature: float = 2.0,
    ):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        # Caching a sampled response means two callers get the same sample,
        # which defeats the point of setting temperature. Default is
        # permissive (cache everything); set to 0.0 for strict determinism.
        self.max_temperature = max_temperature
        self._data: dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    @staticmethod
    def key_for(body: dict) -> str:
        """Hash the semantically meaningful request. `stream` is excluded."""
        payload = {
            "model": body.get("model", ""),
            "messages": body.get("messages", []),
            "max_tokens": body.get("max_tokens"),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "seed": body.get("seed"),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _purge_expired(self) -> None:
        if self.ttl_seconds <= 0:
            return
        expired = [
            key for key, entry in self._data.items()
            if entry.age_seconds > self.ttl_seconds
        ]
        for key in expired:
            del self._data[key]
            self.stats.expiries += 1

    def get(self, key: str) -> CacheEntry | None:
        self.stats.lookups += 1
        self._purge_expired()

        entry = self._data.get(key)
        if entry is None:
            self.stats.misses += 1
            return None

        entry.hits += 1
        self.stats.hits += 1
        self.stats.tokens_saved += estimate_tokens(entry.text)
        # Refresh recency for the LRU ordering.
        self._data.pop(key)
        self._data[key] = entry
        return entry

    def cacheable(self, body: dict) -> bool:
        """Should this request consult the cache at all?"""
        if os.getenv("KOSH_CACHE", "1").lower() not in {"1", "true", "yes"}:
            return False
        try:
            temperature = float(body.get("temperature") or 0.0)
        except (TypeError, ValueError):
            return False
        return temperature <= self.max_temperature

    def store(self, key: str, lines: Iterable[str], text: str,
              finish_reason: str, payload: dict | None = None) -> bool:
        """Store a completed response. Returns True if it was kept.

        Rejects anything incomplete. A silent wrong answer is far more
        expensive than a missed cache hit.
        """
        lines = list(lines)
        if finish_reason == "length":
            # Truncated: HTTP 200, but the answer is incomplete. Caching it
            # would serve a wrong answer to every future caller.
            self.stats.rejects += 1
            return False
        if finish_reason in ("", "error"):
            self.stats.rejects += 1
            return False
        if not lines or not text:
            self.stats.rejects += 1
            return False

        if len(self._data) >= self.max_entries:
            # Evict least recently used; dict preserves insertion order.
            oldest = next(iter(self._data))
            del self._data[oldest]
            self.stats.evictions += 1

        self._data[key] = CacheEntry(
            key=key,
            lines=lines,
            text=text,
            finish_reason=finish_reason,
            created_at=time.monotonic(),
            completion_tokens=estimate_tokens(text),
            payload=payload,
        )
        self.stats.stores += 1
        return True

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def cache_from_env() -> ExactCache:
    def _int(name: str, default: int) -> int:
        try:
            return int(float(os.getenv(name, "")))
        except (TypeError, ValueError):
            return default

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, ""))
        except (TypeError, ValueError):
            return default

    return ExactCache(
        max_entries=_int("KOSH_CACHE_MAX_ENTRIES", 1000),
        ttl_seconds=_float("KOSH_CACHE_TTL", 3600.0),
        max_temperature=_float("KOSH_CACHE_MAX_TEMPERATURE", 2.0),
    )
