"""
Kosh smoke test — streaming edition (Day 2).

    python scripts/smoke.py --mode direct
    python scripts/smoke.py --mode kosh

Measures TTFT (time to first token) SEPARATELY from total latency. This
distinction matters all week:
  - TTFT is prefill-bound  -> what request shaping / prefix hygiene fixes
  - total is decode-bound  -> what caching eliminates entirely

Every response is saved raw to fixtures/ — that becomes replay mode on Day 4.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

DIRECT = os.getenv("KOSH_UPSTREAM_BASE", "https://bharatcode.ai/api/model/v1")
KOSH = "http://localhost:8000/v1"
KEY = os.getenv("BHARATCODE_API_KEY", "")
MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")
MAX_TOKENS = int(os.getenv("SMOKE_MAX_TOKENS", "64"))

# Fixed, committed, never edited. Same prompts forever = comparable numbers.
PROMPTS = [
    "Write a Python function that reverses a linked list.",
    "Explain what a Python generator is in three sentences.",
    "What is the difference between a list and a tuple in Python?",
    "Write a Python function that reverses a linked list.",        # duplicate
    "Find the bug: def add(a, b): return a - b",
    "Explain what a Python generator is in three sentences.",      # duplicate
    "Write pytest tests for a function that checks if a string is a palindrome.",
    "What does the GIL mean for threading in Python?",
]

FIXTURES = pathlib.Path("fixtures")
FIXTURES.mkdir(exist_ok=True)


def call(base: str, prompt: str) -> dict:
    """Stream one request, measuring TTFT separately from total time."""
    started = time.perf_counter()
    ttft_ms: float | None = None
    chunks = 0
    lines: list[str] = []

    with httpx.stream(
        "POST",
        f"{base}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": MAX_TOKENS,
            # qwen-3.8-27b emits <think> reasoning tokens that count against
            # max_tokens — that is what produced the 141s run. Off for dev:
            # far faster loops and no truncation. Day 9 turns this into a
            # per-task Kosh policy (a real feature, not just a convenience).
            "reasoning_effort": "none",
        },
        headers={"Authorization": f"Bearer {KEY}"},
        timeout=180.0,
    ) as response:
        status = response.status_code
        for line in response.iter_lines():
            if not line:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            chunks += 1
            lines.append(line)

    total_ms = (time.perf_counter() - started) * 1000
    return {
        "status": status,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "total_ms": round(total_ms, 1),
        "chunks": chunks,
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "kosh"], required=True)
    args = parser.parse_args()

    if not KEY:
        sys.exit("Set BHARATCODE_API_KEY in .env first")

    base = DIRECT if args.mode == "direct" else KOSH
    rows = []

    print(f"mode={args.mode}  model={MODEL}  max_tokens={MAX_TOKENS}\n")
    for index, prompt in enumerate(PROMPTS):
        result = call(base, prompt)
        ok = result["status"] == 200
        print(
            f"[{'OK ' if ok else 'ERR'}] {index:02d} "
            f"ttft={str(result['ttft_ms']):>8}ms "
            f"total={result['total_ms']:>8.1f}ms "
            f"chunks={result['chunks']:>4}  {prompt[:38]!r}"
        )
        rows.append({"index": index, "mode": args.mode, "prompt": prompt, **result})

    path = FIXTURES / f"{args.mode}_day2.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    ok = sum(1 for r in rows if r["status"] == 200)
    streamed = sum(1 for r in rows if r["chunks"] > 1)
    print(f"\n{ok}/{len(rows)} succeeded · {streamed}/{len(rows)} actually streamed")
    print(f"saved -> {path}")

    if streamed < len(rows):
        print("\nWARNING: some responses arrived in one chunk. Streaming is not working.")


if __name__ == "__main__":
    main()
