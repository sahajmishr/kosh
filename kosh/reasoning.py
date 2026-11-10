"""
Kosh — reasoning policy (Stage 2).

WHY THIS EXISTS
---------------
On Day 1 a single request took 141.6 seconds while its neighbours took
11-37. The cause was not the GPU: qwen-3.8-27b emits <think> reasoning
tokens, and those count against max_tokens. The model spent its entire
output budget thinking and ran out before answering — so the request cost
forty times more than it should and returned a truncated answer with a
200 status.

Setting reasoning_effort="none" dropped that run to 19 seconds.

That was a measurement accident. This module makes it a policy: most
requests do not need a reasoning budget, and the ones that do announce
themselves.

THE RULE
--------
Only ever DOWNGRADE, never upgrade. If the classifier is unsure it leaves
the request exactly as the caller wrote it. A policy that adds reasoning
where none was wanted makes every request slower; a policy that removes it
where it was needed produces a worse answer. Of those two failure modes,
the second is worse — so the classifier errs toward keeping reasoning.

Signals that a task NEEDS reasoning: code, debugging, maths, derivation,
multi-step planning, explicit "why" questions, anything long.
Signals it does NOT: short factual lookups, definitions, formatting,
translation, summarisation of short text.
"""
from __future__ import annotations

import os
import re

NONE = "none"


class PolicyStats:
    """Live counters for the reasoning policy.

    Kept separate from the classifier itself so that classify() stays a
    pure function of its input — testable, no hidden state.
    """

    __slots__ = ("applied", "downgraded", "kept", "caller_set")

    def __init__(self) -> None:
        self.applied = 0      # policy ran on this request
        self.downgraded = 0   # ...and turned reasoning off
        self.kept = 0         # ...and left reasoning on
        self.caller_set = 0   # caller already chose; we did not touch it

    def record(self, action: str) -> None:
        self.applied += 1
        if action == NONE:
            self.downgraded += 1
        elif action == "caller_set":
            self.caller_set += 1
        else:
            self.kept += 1

    def as_dict(self) -> dict:
        return {
            "applied": self.applied,
            "downgraded": self.downgraded,
            "kept": self.kept,
            "caller_set": self.caller_set,
        }

# Deliberately conservative: these must be strong signals.
_REASONING_SIGNALS = re.compile(
    r"""
    \b( debug | implement | refactor | optimi[sz]e | algorithm |
        prove | derive | calculate | compute | solve | evaluate |
        step[\s-]by[\s-]step | time\s+complexity | big[\s-]?o |
        design | architect | trade[\s-]?off | why\s+does |
        what('s|\s+is)\s+wrong | fix\s+the | unit\s+test |
        regex | sql\s+query | recursion | deadlock | race\s+condition )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Cheap structural signals that a reasoning budget would be wasted.
_TRIVIAL_SIGNALS = re.compile(
    r"""
    \b( what\s+is\s+the\s+capital | who\s+(is|was)\s |
        define | translate | capital\s+of |
        what\s+does\s+\w+\s+(mean|stand\s+for) |
        list\s+\d+ | name\s+(two|three|the) )
    \b
    | ^[^.?!]{0,45}[.?]?$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def contains_code(text: str) -> bool:
    return ("```" in text
            or "def " in text
            or "function " in text
            or "class " in text
            or "import " in text
            or "SELECT " in text.upper())


def classify(prompt: str) -> str:
    """Return 'reason' or 'skip'. Conservative: 'reason' when unsure."""
    text = (prompt or "").strip()
    if not text:
        return "reason"                     # unknown -> do not touch it

    if contains_code(text):
        return "reason"
    if _REASONING_SIGNALS.search(text):
        return "reason"
    if len(text) > 400:
        return "reason"                     # long task, leave it alone
    if _TRIVIAL_SIGNALS.search(text) and len(text) <= 200:
        return "skip"

    return "reason"                         # default: do not touch it


def apply_policy(body: dict, prompt: str) -> tuple[dict, str]:
    """Set reasoning_effort when we are confident it is not needed.

    Returns (body, action) where action is 'none', 'kept' or 'caller_set'.
    Never overrides a value the caller chose explicitly.
    """
    if body.get("reasoning_effort") is not None:
        return body, "caller_set"

    if classify(prompt) != "skip":
        return body, "kept"

    shaped = dict(body)
    shaped["reasoning_effort"] = NONE
    return shaped, NONE


def policy_from_env() -> str:
    """'off' (default is 'auto' only if you opt in) — safest possible default."""
    return os.getenv("KOSH_REASONING", "off").strip().lower()
