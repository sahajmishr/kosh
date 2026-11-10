"""
Kosh — request shaping / prefix hygiene (Day 5).

THE INSIGHT
-----------
BharatCode runs vLLM, and vLLM has Automatic Prefix Caching on by default.
It hashes the prompt in 16-token blocks and skips prefill for any block it
has already computed. If a hundred students send the same 2,000-token system
prompt, the first pays for prefill and the other ninety-nine get it free.

But APC matches on an EXACT prefix, and only from the start. One changed
character at token 3 invalidates everything after it. A system prompt that
opens with "Current date: 2026-10-29T14:22:07" therefore gets a 0% cache hit
rate on every request, forever, and nobody notices — the requests succeed.

THE GAP
-------
Every published fix is infrastructure-side: route the request to the replica
that already holds the prefix. GKE Inference Gateway, llm-d and Alibaba ACK
GIE all do this, and they report 57x-92% improvements. All of them need
multiple replicas, Kubernetes, and control of the serving layer.

BharatCode has ONE GPU with three slots. There is nowhere to route.

So Kosh fixes it from the only side it can: the request itself. That is
prefix hygiene, and it is the reason this project exists.

THE RULE
--------
Measure and report by default. Rewrite only what is provably safe.

Silently editing a user's prompt is the fastest way to destroy trust in a
proxy. So shaping has three modes and the default is the one that changes
nothing:

  report     analyse and report via headers; body untouched      (default)
  canonical  normalise whitespace so identical requests are byte-identical
  reorder    additionally move volatile markers to the end of the prefix

Only `reorder` can change model behaviour, and it moves only machine-
generated markers (timestamps, UUIDs), never prose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Rough, and clearly labelled as rough. Good enough for a cacheability score;
# never used for billing (the ledger uses the upstream's own counts).
CHARS_PER_TOKEN = 4.0

# Volatile markers: machine-generated strings that change every request and
# so destroy any prefix they appear in. Deliberately conservative — a false
# positive here would mean rewriting text we should not touch.
VOLATILE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("iso_timestamp", re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")),
    ("uuid", re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")),
    ("epoch_seconds", re.compile(r"\b1[6-9]\d{8}\b")),          # 1.6e9 .. 1.9e9
    ("session_id", re.compile(r"\b(?:session|request|trace|correlation)[-_ ]?id\b\s*[:=]?\s*\S+",
                              re.IGNORECASE)),
    ("random_hex", re.compile(r"\b[0-9a-f]{16,}\b")),
]

MODES = ("off", "report", "canonical", "reorder")


# ------------------------------------------------------------ canonical ----

def normalise_whitespace(text: str) -> str:
    """Collapse trailing whitespace and normalise newlines.

    Purely cosmetic: two prompts that differ only in trailing spaces are the
    same prompt, but they hash differently and so miss every cache.
    """
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def canonicalize_messages(messages: list[dict]) -> list[dict]:
    """Whitespace-normalise every string content block, preserving order."""
    out: list[dict] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = normalise_whitespace(content)
        elif isinstance(content, list):
            item["content"] = [
                {**part, "text": normalise_whitespace(part["text"])}
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
        out.append(item)
    return out


def flatten(messages: list[dict]) -> str:
    """Join all message content into the approximate token stream.

    This is an approximation of how the prompt is tokenised — the real
    tokenisation applies a chat template we do not have. But for locating
    *where* volatility sits in the prompt, it is accurate enough, and that
    is all we use it for.
    """
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


# -------------------------------------------------------------- analysis ---

@dataclass
class PrefixReport:
    """How cache-friendly is this prompt?"""

    total_chars: int = 0
    cacheable_chars: int = 0
    cacheable_pct: float = 0.0
    volatile_count: int = 0
    volatile_kinds: list[str] = field(default_factory=list)
    first_volatile_at: int | None = None          # char offset, None if clean
    first_volatile_kind: str | None = None
    verdict: str = "unknown"
    rewritten: bool = False

    def as_headers(self) -> dict[str, str]:
        return {
            "x-kosh-prefix-cacheable": f"{self.cacheable_pct:.1f}",
            "x-kosh-prefix-tokens": str(int(self.cacheable_chars / CHARS_PER_TOKEN)),
            "x-kosh-prefix-verdict": self.verdict,
            "x-kosh-volatiles": str(self.volatile_count),
            "x-kosh-volatile-kinds": ",".join(self.volatile_kinds) or "none",
            "x-kosh-shaped": "1" if self.rewritten else "0",
        }


def _verdict(pct: float, volatile_count: int) -> str:
    if volatile_count == 0:
        return "excellent"
    if pct >= 80.0:
        return "good"
    if pct >= 40.0:
        return "poor"
    return "hostile"


def find_volatile(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, kind) for every volatile marker, sorted by start."""
    found: list[tuple[int, int, str]] = []
    for kind, pattern in VOLATILE_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), match.end(), kind))
    found.sort(key=lambda item: item[0])

    # Drop overlaps: the earliest, longest match wins.
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in found:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end, kind))
    return merged


def analyse(messages: list[dict]) -> PrefixReport:
    """Score a prompt's prefix stability without changing it."""
    text = flatten(messages)
    total = len(text)
    if total == 0:
        return PrefixReport(verdict="excellent")

    spans = find_volatile(text)
    if not spans:
        return PrefixReport(
            total_chars=total,
            cacheable_chars=total,
            cacheable_pct=100.0,
            verdict="excellent",
        )

    first_start, _, _ = spans[0]
    pct = 100.0 * first_start / total
    kinds = sorted({kind for _, _, kind in spans})
    return PrefixReport(
        total_chars=total,
        cacheable_chars=first_start,
        cacheable_pct=round(pct, 1),
        volatile_count=len(spans),
        volatile_kinds=kinds,
        first_volatile_at=first_start,
        first_volatile_kind=spans[0][2],
        verdict=_verdict(pct, len(spans)),
    )


# ---------------------------------------------------------------- shape ----

def _expand_to_lines(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    """Grow each span to the whole line it sits on.

    Moving the bare token would leave its label behind: stripping
    "2026-10-29T14:22:07Z" out of "Current date: 2026-10-29T14:22:07Z"
    leaves a dangling "Current date: " still at the front — which is worse
    than doing nothing. So we move the entire line, label and value together.
    """
    lines: list[tuple[int, int]] = []
    for start, end, _kind in spans:
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)
        lines.append((line_start, line_end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(lines):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def reorder_messages(messages: list[dict]) -> list[dict]:
    """Move volatile lines out of the leading position of the first message.

    Only the FIRST text-bearing message is touched, and only lines
    containing machine-generated markers are moved — never prose. The lines
    are re-attached at the end of that message, verbatim, so no information
    is lost; they just stop sitting in front of everything the cache could
    have reused.
    """
    if not messages:
        return messages

    out = [dict(message) for message in messages]
    index = next(
        (i for i, m in enumerate(out) if isinstance(m.get("content"), str) and m["content"]),
        None,
    )
    if index is None:
        return out

    text = out[index]["content"]
    spans = find_volatile(text)
    if not spans:
        return out

    # Only worth doing if volatility sits in the first half of the prompt —
    # late volatility barely costs anything, because blocks before it still hit.
    if spans[0][0] >= len(text) / 2:
        return out

    moved: list[str] = []
    remaining = text
    # Back to front, so earlier offsets stay valid as we cut.
    for start, end in reversed(_expand_to_lines(text, spans)):
        segment = remaining[start:end].strip()
        if segment:
            moved.append(segment)
        remaining = remaining[:start] + remaining[end:]

    # Removing lines can leave runs of blank lines behind.
    remaining = re.sub(r"\n{3,}", "\n\n", remaining).strip()
    if not remaining:
        return out  # nothing stable left; leave the prompt alone

    moved.reverse()
    out[index]["content"] = remaining + "\n\n" + "\n".join(moved)
    return out


def shape_request(body: dict, mode: str) -> tuple[dict, PrefixReport]:
    """Apply shaping in the requested mode. Returns (body, report).

    `body` is a new dict; the caller's is never mutated.
    """
    if mode not in MODES:
        raise ValueError(f"unknown shaping mode {mode!r}; choose from {MODES}")

    shaped = dict(body)
    messages = list(shaped.get("messages") or [])
    report = analyse(messages)

    if mode == "off":
        return shaped, report

    if mode in ("canonical", "reorder"):
        messages = canonicalize_messages(messages)
        report = analyse(messages)

    if mode == "reorder":
        reordered = reorder_messages(messages)
        report.rewritten = flatten(reordered) != flatten(messages)
        messages = reordered
        report = analyse(messages)
        report.rewritten = True

    shaped["messages"] = messages
    return shaped, report
