"""
Day 1 proof: the same request, direct vs through Kosh.

    python scripts/smoke.py --mode direct
    python scripts/smoke.py --mode kosh

Every response is saved raw to fixtures/ — that file becomes replay mode on Day 4,
so start collecting from your very first request. It costs nothing.
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

DIRECT = os.getenv("BHARATCODE_BASE_URL", "https://bharatcode.ai/api/model/v1")
KOSH = "http://localhost:8000/v1"
KEY = os.getenv("BHARATCODE_API_KEY", "")
MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")

# Fixed, committed, never edited. Same prompts forever = comparable numbers.
PROMPTS = [
    "Write a Python function that reverses a linked list.",
    "Explain what a Python generator is in three sentences.",
    "What is the difference between a list and a tuple in Python?",
    "Write a Python function that reverses a linked list.",          # duplicate
    "Find the bug: def add(a, b): return a - b",
    "Explain what a Python generator is in three sentences.",        # duplicate
    "Write pytest tests for a function that checks if a string is a palindrome.",
    "What does the GIL mean for threading in Python?",
]

FIXTURES = pathlib.Path("fixtures")
FIXTURES.mkdir(exist_ok=True)


def call(base: str, prompt: str) -> tuple[float, int, str]:
    started = time.perf_counter()
    response = httpx.post(
        f"{base}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 256,
        },
        headers={"Authorization": f"Bearer {KEY}"},
        timeout=180.0,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return elapsed_ms, response.status_code, response.text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "kosh"], required=True)
    args = parser.parse_args()

    if not KEY:
        sys.exit("Set BHARATCODE_API_KEY in .env first")

    base = DIRECT if args.mode == "direct" else KOSH
    rows = []

    for index, prompt in enumerate(PROMPTS):
        elapsed_ms, status, raw = call(base, prompt)
        flag = "OK " if status == 200 else "ERR"
        print(f"[{args.mode}] {flag} {index:02d} {status:>3} {elapsed_ms:>8.1f}ms  {prompt[:44]!r}")
        rows.append(
            {
                "index": index,
                "mode": args.mode,
                "prompt": prompt,
                "status": status,
                "latency_ms": elapsed_ms,
                "raw": raw,
            }
        )

    path = FIXTURES / f"{args.mode}_day1.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    ok = sum(1 for r in rows if r["status"] == 200)
    print(f"\n{ok}/{len(rows)} succeeded -> saved to {path}")


if __name__ == "__main__":
    main()