"""
Day 5b — does BharatCode's vLLM report prefix-cache hits directly?

TTFT on a shared GPU is dominated by queue wait, which swamps the prefill
signal (see bench_prefix.py: within-phase spread was ~13s, between-phase
difference ~2.7s). Timing is the wrong instrument.

vLLM, however, reports prefix-cache hits straight in the response body:

    "usage": {"prompt_tokens_details": {"cached_tokens": N}}

If BharatCode exposes that field, we can measure prefix reuse exactly, with
no timing and no queue noise at all. This probe finds out in ~15 seconds.

    python scripts/probe_cached_tokens.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

KEY = os.getenv("BHARATCODE_API_KEY", "")
BASE = os.getenv("BHARATCODE_BASE", "https://bharatcode.ai/api/model/v1")
MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")

from scripts.bench_prefix import build_prefix, unique_prefix  # noqa: E402


def ask(system: str, question: str = "What is the first rule listed?") -> dict:
    """Non-streaming, so we can read the usage block."""
    response = httpx.post(
        f"{BASE}/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "max_tokens": 16,
            "reasoning_effort": "none",
        },
        headers={"Authorization": f"Bearer {KEY}"},
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()


def usage_of(payload: dict) -> dict:
    usage = payload.get("usage") or {}
    return usage


def main() -> None:
    if not KEY:
        sys.exit("Set BHARATCODE_API_KEY in .env first")

    prefix_tokens = 2500
    shared = build_prefix(prefix_tokens)

    print(f"probing {BASE} · model={MODEL} · prefix ~{prefix_tokens} tokens\n")

    steps = [
        ("1 warm  (cold cache)", shared),
        ("2 warm  (should hit)", shared),
        ("3 warm  (should hit)", shared),
        ("4 cold  (unique prefix)", unique_prefix(1, prefix_tokens)),
        ("5 hostile (1 char changed at front)", "X" + shared[1:]),
        ("6 warm  (again, should hit)", shared),
    ]

    seen_field = False
    for label, system in steps:
        try:
            payload = ask(system)
        except httpx.HTTPError as exc:
            print(f"{label:<38} ERROR {exc}")
            continue

        usage = usage_of(payload)
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")

        if "prompt_tokens_details" in usage:
            seen_field = True

        print(
            f"{label:<38} "
            f"prompt={usage.get('prompt_tokens', '?'):>6}  "
            f"cached={str(cached):>6}  "
            f"completion={usage.get('completion_tokens', '?'):>4}"
        )

        if not seen_field:
            # One full dump, so we can see what they actually return.
            print("\n--- full usage block from step 1 ---")
            print(json.dumps(usage, indent=2))
            print("\n'prompt_tokens_details' absent. BharatCode may be running an")
            print("older vLLM, or proxying through a layer that strips it.")
            print("Falling back to timing means accepting queue noise; see below.")
            break

    print()
    if seen_field:
        print("If steps 2/3/6 show cached_tokens near the prefix size while 4 and 5")
        print("show ~0, prefix caching is confirmed directly — no timing needed.")
        print("That is the result for the README.")
    else:
        print("No direct field. Options, in order of preference:")
        print("  1. Ask BharatCode directly whether APC is enabled (contact@).")
        print("     A confirmed 'yes' plus our header evidence is enough for the")
        print("     README; we then present prefix hygiene as prevention, not as")
        print("     a measured speedup.")
        print("  2. Measure with a MUCH larger prefix (15k-20k tokens) so prefill")
        print("     is a large fraction of TTFT, interleaved A/B/A/B.")
        print("  3. Report the negative result. 'We tried to measure this and the")
        print("     shared queue made it unmeasurable' is a legitimate, rare,")
        print("     and credible finding.")


if __name__ == "__main__":
    main()
