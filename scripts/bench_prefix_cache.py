"""
Day 5c — mapping BharatCode's prefix-cache ceiling.

probe_cached_tokens.py proved prefix caching is live:

    repeated 2,093-token prompt     -> cached_tokens = 800
    unique prompt                   -> cached_tokens = 0
    same prompt, 1 char changed     -> cached_tokens = 0

800 is not 2,093. 800 = 50 blocks x 16 tokens (vLLM's default block size).
Something caps it. This script finds out what, with two experiments.

EXPERIMENT 1 — ceiling sweep
    Warm prefixes of increasing length and record cached_tokens.
    If cached_tokens plateaus, there is a hard ceiling and we find its value.

EXPERIMENT 2 — marker position sweep  (the important one)
    Take one fixed long prefix and put a volatile marker at increasing depths.
    Record cached_tokens at each depth.

    If the ceiling is ~800 tokens, then a marker at token 1,000 should cost
    NOTHING (you only ever cached 800), while a marker at token 10 costs
    everything. That turns "put stable content first" from advice into a
    number, and it tells Kosh exactly how much cacheable runway a prompt has.

WHY THIS IS MEASURED, NOT TIMED
TTFT on this GPU swings 13 seconds within a single phase (bench_prefix.py).
cached_tokens is a direct counter reported by vLLM. It has no queue noise.

Usage:
    python scripts/bench_prefix_cache.py
    python scripts/bench_prefix_cache.py --sizes 250,500,1000,2000,4000,8000
    python scripts/bench_prefix_cache.py --marker-only
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

KEY = os.getenv("BHARATCODE_API_KEY", "")
BASE = os.getenv("BHARATCODE_BASE", "https://bharatcode.ai/api/model/v1")
MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")
FIXTURES = ROOT / "fixtures"

from scripts.bench_prefix import DOC_PARAGRAPHS, QUESTIONS  # noqa: E402

MARKER = "\n[Current date: 2026-10-29T14:22:07Z · session 7f3a-9c21]\n"


def ask(system: str, question: str) -> dict:
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
    payload = response.json()
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "cached_tokens": details.get("cached_tokens", 0),
    }


def build_prefix(target_tokens: int) -> str:
    """Rebuild locally so this script depends on nothing but .env."""
    approx_chars = int(target_tokens * 4)
    chunks: list[str] = []
    total = 0
    index = 0
    while total < approx_chars:
        para = DOC_PARAGRAPHS[index % len(DOC_PARAGRAPHS)]
        chunks.append(para)
        total += len(para) + 2
        index += 1
    return "\n\n".join(chunks)


def measure(system: str, reps: int = 3) -> dict:
    """One cold call to populate, then `reps` warm calls to measure."""
    ask(system, QUESTIONS[0])                      # populate, not measured
    time.sleep(0.4)
    readings = [ask(system, QUESTIONS[i % len(QUESTIONS)]) for i in range(reps)]
    return {
        "prompt_tokens": statistics.median(r["prompt_tokens"] for r in readings),
        "cached_tokens": statistics.median(r["cached_tokens"] for r in readings),
        "cached_all": [r["cached_tokens"] for r in readings],
    }


def experiment_ceiling(sizes: list[int], reps: int) -> list[dict]:
    print("EXPERIMENT 1 — does cached_tokens plateau?\n")
    print(f"{'prefix tokens':>14} {'prompt':>8} {'cached':>8} {'pct':>7}   readings")
    print("-" * 62)
    rows = []
    for size in sizes:
        system = build_prefix(size)
        try:
            result = measure(system, reps)
        except httpx.HTTPError as exc:
            print(f"{size:>14}   ERROR {exc}")
            continue
        pct = 100.0 * result["cached_tokens"] / max(1, result["prompt_tokens"])
        print(f"{size:>14} {int(result['prompt_tokens']):>8} "
              f"{int(result['cached_tokens']):>8} {pct:>6.1f}%   {result['cached_all']}")
        rows.append({"target_tokens": size, **result, "cached_pct": round(pct, 1)})
    return rows


def experiment_marker(depth_tokens: list[int], prefix_tokens: int, reps: int) -> list[dict]:
    print("\nEXPERIMENT 2 — where does a volatile marker stop mattering?\n")
    print(f"  (fixed prefix ~{prefix_tokens} tokens; marker inserted at increasing depth)")
    print(f"\n{'marker at token':>16} {'prompt':>8} {'cached':>8} {'cost':>8}")
    print("-" * 46)

    base = build_prefix(prefix_tokens)
    rows = []
    for depth in depth_tokens:
        char_pos = int(depth * 4)                        # ~4 chars per token
        char_pos = min(char_pos, len(base))
        # Insert on a line boundary so we do not split a sentence mid-word.
        nl = base.rfind("\n\n", 0, char_pos)
        if nl == -1:
            nl = char_pos
        system = base[:nl] + MARKER + base[nl:]

        try:
            result = measure(system, reps)
        except httpx.HTTPError as exc:
            print(f"{depth:>16}   ERROR {exc}")
            continue
        print(f"{depth:>16} {int(result['prompt_tokens']):>8} "
              f"{int(result['cached_tokens']):>8}")
        rows.append({"marker_depth_tokens": depth, **result})

    # Cost is relative to the clean baseline at the same prefix size.
    if rows:
        clean = max(r["cached_tokens"] for r in rows)
        print(f"\n{'marker at token':>16} {'cached':>8} {'cost':>8}")
        print("-" * 34)
        for row in rows:
            cost = clean - row["cached_tokens"]
            row["cost_tokens"] = int(cost)
            print(f"{row['marker_depth_tokens']:>16} "
                  f"{int(row['cached_tokens']):>8} {int(cost):>8}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="250,500,1000,2000,4000,8000")
    parser.add_argument("--depths", default="0,50,200,500,900,1500,3000")
    parser.add_argument("--prefix-tokens", type=int, default=4000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--marker-only", action="store_true")
    args = parser.parse_args()

    if not KEY:
        sys.exit("Set BHARATCODE_API_KEY in .env first")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]

    est = (sum(sizes) * (args.reps + 1)) + (len(depths) * (args.prefix_tokens) * (args.reps + 1))
    print(f"BharatCode prefix-cache ceiling · model={MODEL}")
    print(f"est. input tokens ~{est:,} of your 1,000,000/day\n")

    result: dict = {"model": MODEL}

    if not args.marker_only:
        result["ceiling"] = experiment_ceiling(sizes, args.reps)

    result["marker"] = experiment_marker(depths, args.prefix_tokens, args.reps)

    print("\n" + "=" * 62)
    print("How to read this:")
    print("  Experiment 1: if 'cached' stops growing while 'prefix tokens' keeps")
    print("  growing, the plateau is the cache ceiling. Everything past it is")
    print("  recomputed every request no matter how stable it is.")
    print()
    print("  Experiment 2: 'cost' is how many cached tokens the marker destroyed.")
    print("  If cost falls to ~0 once the marker is past the ceiling, that proves")
    print("  the ceiling is what limits you — and that volatility beyond it is")
    print("  FREE. That is the number Kosh's shaping should optimise against.")

    path = FIXTURES / "bench_prefix_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
