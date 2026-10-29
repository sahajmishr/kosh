"""
Kosh — deterministic replay (Day 4).

Serves recorded responses instead of calling an upstream. Zero tokens, zero
API keys, zero dependence on a shared GPU being idle.

WHY THIS IS A FEATURE AND NOT A TEST HELPER
-------------------------------------------
Every other project in this space can only be verified by someone who
already has API access. Kosh can be verified by anyone, offline, with no
account: point an OpenAI client at it and the entire system runs — including
its failure behaviour — without consuming a single token of shared capacity.

That matters because it removes the biggest obstacle between a judge and a
real evaluation.

THE DESIGN RULE
---------------
Replay replaces the TRANSPORT, not the LOGIC.

ReplayTransport impersonates httpx.AsyncClient — it implements only the two
methods open_with_retry actually calls (build_request and send). So the real
retry loop, the real jittered backoff and the real circuit breaker run
completely unchanged. Replaying *around* the logic would exercise none of
it, and every test would prove nothing.

WHAT IS INJECTABLE
------------------
Every fault below is a real, documented BharatCode failure mode:

  429        rate limited
  503        model busy (their documented "wait a few seconds and retry")
  500        server error
  timeout    hangs, so the client's own timeout fires
  truncate   HTTP 200 but finish_reason "length" — looks like success, isn't
  disconnect dies mid-stream, after some tokens have already been emitted

`truncate` is the nastiest and the most valuable: the response is a 200, so
naive client code believes it succeeded and silently uses a truncated
answer. Practising against it now is what lets the Day 7 ledger flag it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# ------------------------------------------------------------ fault words ---
OK = "ok"
RATE_LIMITED = "429"
BUSY = "503"
SERVER_ERROR = "500"
TIMEOUT = "timeout"
TRUNCATED = "truncate"
DISCONNECT = "disconnect"

# Faults delivered as an HTTP status, before any bytes are emitted. These are
# exactly the ones the retry layer can recover from.
STATUS_FAULTS: dict[str, tuple[int, dict[str, str]]] = {
    RATE_LIMITED: (429, {"retry-after": "2"}),
    BUSY: (503, {"retry-after": "5"}),
    SERVER_ERROR: (500, {}),
}

# Scripted profiles. Cycled per request, so a run is reproducible.
PROFILES: dict[str, list[str]] = {
    "clean": [],
    "flaky": [OK, OK, OK, BUSY, OK, OK, RATE_LIMITED, OK],
    "storm": [BUSY, BUSY, OK, BUSY, RATE_LIMITED, BUSY, OK, BUSY],
    "outage": [BUSY],
    "hostile": [
        TIMEOUT, BUSY, TRUNCATED, DISCONNECT,
        RATE_LIMITED, BUSY, SERVER_ERROR, OK,
    ],
}


# ------------------------------------------------------------- fixtures ----

@dataclass
class Fixture:
    """One recorded response: raw SSE lines plus the timing they arrived with."""

    lines: list[str]
    text: str
    ttft_ms: float
    total_ms: float
    chunks: int
    prompt: str = ""

    @property
    def gap_ms(self) -> float:
        """Mean gap between chunks — reproduces inter-token latency."""
        if self.chunks <= 1:
            return 0.0
        span = max(0.0, self.total_ms - self.ttft_ms)
        return span / self.chunks


def _extract_text(lines: list[str]) -> str:
    """Pull the assistant text back out of recorded SSE chunks."""
    parts: list[str] = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                parts.append(content)
    return "".join(parts)


def fixture_from_row(row: dict) -> Fixture | None:
    """Build a Fixture from one line of a smoke.py jsonl file."""
    lines = [line for line in row.get("lines", []) if line]
    if not lines:
        return None
    return Fixture(
        lines=lines,
        text=_extract_text(lines),
        ttft_ms=float(row.get("ttft_ms") or 0.0),
        total_ms=float(row.get("total_ms") or 0.0),
        chunks=int(row.get("chunks") or len(lines)),
        prompt=row.get("prompt", ""),
    )


# Used only when no recorded fixtures exist, so replay never hard-fails on a
# fresh clone. Deliberately small and obviously synthetic.
_BUILTIN_PROMPTS = [
    "What is the capital of India?",
    "Explain HTTP 503 in one sentence.",
    "Write a Python function to reverse a string.",
    "What does idempotent mean in APIs?",
    "Name two causes of high time-to-first-token.",
]


def _builtin_fixtures() -> list[Fixture]:
    fixtures: list[Fixture] = []
    for index, prompt in enumerate(_BUILTIN_PROMPTS):
        text = (
            f"[replay {index}] This is a synthetic Kosh replay response. "
            f"No tokens were generated and no GPU was involved. "
            f"The prompt asked: {prompt}"
        )
        lines = [
            "data: " + json.dumps({
                "id": f"chatcmpl-replay{index}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "qwen-3.8-27b",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }, separators=(",", ":"))
        ]
        for word in text.split(" "):
            lines.append("data: " + json.dumps({
                "id": f"chatcmpl-replay{index}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "qwen-3.8-27b",
                "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
            }, separators=(",", ":")))
        lines.append("data: " + json.dumps({
            "id": f"chatcmpl-replay{index}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "qwen-3.8-27b",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }, separators=(",", ":")))
        lines.append("data: [DONE]")
        fixtures.append(Fixture(
            lines=lines,
            text=text,
            ttft_ms=700.0,
            total_ms=1400.0,
            chunks=len(lines),
            prompt=prompt,
        ))
    return fixtures


def load_fixtures(directory: Path = FIXTURES_DIR) -> list[Fixture]:
    """Load every .jsonl in fixtures/. Missing or empty -> built-in corpus."""
    fixtures: list[Fixture] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for row_line in handle:
                    row_line = row_line.strip()
                    if not row_line:
                        continue
                    try:
                        row = json.loads(row_line)
                    except json.JSONDecodeError:
                        continue
                    fixture = fixture_from_row(row)
                    if fixture is not None:
                        fixtures.append(fixture)
    return fixtures or _builtin_fixtures()


# -------------------------------------------------------------- matching ---

def canonical_key(body: dict) -> str:
    """Stable hash of the semantically meaningful parts of a request.

    Two identical prompts must map to the same fixture on every machine and
    every run. That stability is what makes benchmark numbers reproducible
    for someone else.
    """
    payload = {
        "model": body.get("model", ""),
        "messages": body.get("messages", []),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class FaultScript:
    """Cyclic fault schedule. An empty list means every request succeeds."""

    outcomes: list[str] = field(default_factory=list)
    cursor: int = 0

    def next(self) -> str:
        if not self.outcomes:
            return OK
        outcome = self.outcomes[self.cursor % len(self.outcomes)]
        self.cursor += 1
        return outcome

    def reset(self) -> None:
        self.cursor = 0


@dataclass
class ReplayConfig:
    enabled: bool = False
    fixtures: list[Fixture] = field(default_factory=list)
    script: FaultScript = field(default_factory=FaultScript)
    profile: str = "clean"
    # 0 = instant (tests). 1 = recorded timing. 10 = ten times faster (demo).
    speed: float = 0.0

    def select(self, body: dict) -> Fixture:
        if not self.fixtures:
            self.fixtures = load_fixtures()
        index = int(canonical_key(body), 16) % len(self.fixtures)
        return self.fixtures[index]


def config_from_env() -> ReplayConfig:
    """Build config from KOSH_REPLAY* environment variables."""
    enabled = os.getenv("KOSH_REPLAY", "0").lower() in {"1", "true", "yes"}
    profile = os.getenv("KOSH_REPLAY_PROFILE", "clean")
    if profile not in PROFILES:
        raise ValueError(
            f"unknown KOSH_REPLAY_PROFILE {profile!r}; choose from {sorted(PROFILES)}"
        )
    return ReplayConfig(
        enabled=enabled,
        profile=profile,
        script=FaultScript(outcomes=list(PROFILES[profile])),
        speed=float(os.getenv("KOSH_REPLAY_SPEED", "0")),
        fixtures=load_fixtures() if enabled else [],
    )


# -------------------------------------------------------- the fake client ---

@dataclass
class _ReplayRequest:
    method: str
    url: str
    json: dict
    headers: dict


class _ReplayResponse:
    """Just enough of httpx.Response for upstream.py to stream or read it."""

    def __init__(
        self,
        status_code: int,
        lines: list[str],
        *,
        config: ReplayConfig,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        fixture: Fixture | None = None,
        die_after: int | None = None,
        truncated: bool = False,
    ):
        self.status_code = status_code
        self.headers: dict[str, str] = dict(headers or {})
        self._lines = lines
        self._config = config
        self._body = body
        self._fixture = fixture
        self._die_after = die_after
        self._truncated = truncated
        self.closed = False

    async def aread(self) -> bytes:
        if self._body is not None:
            return self._body
        text = _extract_text(self._lines)
        payload = {
            "id": "chatcmpl-replay",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen-3.8-27b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if self._truncated else "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return json.dumps(payload).encode("utf-8")

    async def aclose(self) -> None:
        self.closed = True

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        try:
            for index, line in enumerate(self._lines):
                await self._pace(index)
                if self._die_after is not None and index >= self._die_after:
                    raise httpx.RemoteProtocolError("upstream closed the stream")
                yield (line + "\n\n").encode("utf-8")
        finally:
            self.closed = True

    async def _pace(self, index: int) -> None:
        """Reproduce recorded timing when speed > 0."""
        if self._config.speed <= 0 or self._fixture is None:
            return
        ms = self._fixture.ttft_ms if index == 0 else self._fixture.gap_ms
        if ms > 0:
            await asyncio.sleep(ms / 1000.0 / self._config.speed)


class ReplayTransport:
    """Drop-in stand-in for httpx.AsyncClient that never touches the network.

    Implements only what open_with_retry uses: build_request and send.
    Everything downstream — retry, jitter, breaker, SSE passthrough — is the
    real production code path.
    """

    def __init__(self, config: ReplayConfig):
        self.config = config
        self.calls = 0
        self.outcomes: list[str] = []

    def build_request(self, method: str, url: str, json: dict | None = None,
                      headers: dict | None = None) -> _ReplayRequest:
        return _ReplayRequest(method, url, json or {}, headers or {})

    async def send(self, request: _ReplayRequest, stream: bool = False) -> _ReplayResponse:
        self.calls += 1
        outcome = self.config.script.next()
        self.outcomes.append(outcome)

        if outcome == TIMEOUT:
            raise httpx.TimeoutException("replay: injected timeout")

        if outcome in STATUS_FAULTS:
            status, headers = STATUS_FAULTS[outcome]
            return _ReplayResponse(
                status,
                [],
                config=self.config,
                headers=headers,
                body=json.dumps({"error": f"replay injected {outcome}"}).encode("utf-8"),
            )

        fixture = self.config.select(request.json)
        lines = list(fixture.lines)

        if outcome == TRUNCATED:
            return _ReplayResponse(
                200, truncate_lines(lines), config=self.config,
                fixture=fixture, truncated=True,
            )

        if outcome == DISCONNECT:
            # A third of the way in — enough to prove partial output happened.
            return _ReplayResponse(
                200, lines, config=self.config,
                fixture=fixture, die_after=max(1, len(lines) // 3),
            )

        return _ReplayResponse(200, lines, config=self.config, fixture=fixture)

    async def aclose(self) -> None:
        return None


def truncate_lines(lines: list[str]) -> list[str]:
    """Cut the stream short and finish with finish_reason "length".

    Reproduces BharatCode's documented behaviour: HTTP 200, but the answer is
    incomplete because it ran out of max_tokens. Client code that only checks
    the status code will believe it succeeded.
    """
    stop_index = None
    for index in range(len(lines) - 1, -1, -1):
        if '"finish_reason":null' not in lines[index] and '"finish_reason":null' not in lines[index]:
            if '"stop"' in lines[index]:
                stop_index = index
                break
    if stop_index is None:
        stop_index = max(1, len(lines) - 2)

    kept = lines[:stop_index]
    kept = kept[: max(1, int(len(kept) * 0.66))]

    finish = json.dumps({
        "id": "chatcmpl-replay",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "qwen-3.8-27b",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
    }, separators=(",", ":"))
    return kept + [f"data: {finish}", "data: [DONE]"]
