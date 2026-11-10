"""
Day 6 — cache benchmark on replay. No API key, no network, no queue.

This exists because the live GPU is unusable right now (queue depth 8,
180-second timeouts). Rather than wait, we measure the cache against
recorded responses. Every number here is exact — there is no timing noise,
because there is no shared GPU involved.

THE WORKLOAD
  exact duplicates    byte-identical repeats          -> must hit
  near duplicates     same words, different whitespace -> must hit IF shaped
  adversarial         differ by ONE meaningful word    -> must NOT hit
  long-context        shared prefix, different question-> must NOT hit (yet)
  unique              nothing in common                -> must not hit

THE NUMBER THAT MATTERS MOST IS NOT THE HIT RATE
It is the FALSE-HIT rate on the adversarial slice. A cache that answers a
different question than the one asked is worse than no cache, and it fails
silently. That is why the slice exists and why it must read 0.0%.

Usage:
    python scripts/bench_cache.py
    python scripts/bench_cache.py --shape off      # show what shaping buys
    python scripts/bench_cache.py --requests 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kosh.main import create_app                                   # noqa: E402
from kosh.replay import ReplayConfig, _builtin_fixtures             # noqa: E402

BASE_QUESTIONS = [
    "What is the capital of India?",
    "Explain HTTP 503 in one sentence.",
    "Write a Python function to reverse a string.",
    "What does idempotent mean in APIs?",
    "Name two causes of high time-to-first-token.",
    "Summarise what a circuit breaker does.",
    "Why is jitter important when retrying?",
    "What is backpressure?",
]

# Pools are deliberately DISJOINT. Overlapping pools contaminate the
# results: a prompt reused across slices makes a legitimate hit look like a
# false hit, which is exactly how you fool yourself with a cache benchmark.
BASE_QUESTIONS = [
    "What is the capital of India?",
    "Explain HTTP 503 in one sentence.",
    "Write a Python function to reverse a string.",
    "What does idempotent mean in APIs?",
    "Name two causes of high time-to-first-token.",
    "Summarise what a circuit breaker does.",
    "Why is jitter important when retrying?",
    "What is backpressure?",
]

# Used only by the near-duplicate slice, so its first call is always a miss.
NEAR_QUESTIONS = [
    "Define a hash collision.",
    "What is tail latency?",
    "When should you use a queue?",
    "Describe exponential backoff.",
]

# Differ by ONE word. Never appear in any other slice.
ADVERSARIAL_PAIRS = [
    ("Who wrote the Ramayana?", "Who wrote the Mahabharata?"),
    ("Sort this list ascending.", "Sort this list descending."),
    ("Which port does HTTPS use?", "Which port does HTTP use?"),
]

LONG_PREFIX = (
    "You are a careful teaching assistant for an undergraduate algorithms "
    "course. Answer concisely, show reasoning in at most three steps, and "
    "never give a final answer without explaining how you reached it. "
) * 12

# What each slice SHOULD score, with a correctly implemented exact cache.
EXPECTED = {
    "exact duplicates": "100% (warm cache)",
    "near duplicates":  "100% shaped, 50% unshaped",
    "adversarial":      "0% - a hit here is a BUG",
    "long-context":     "0% - shared prefix, different question",
    "unique":           "0%",
}
SLICE_ORDER = ["exact duplicates", "near duplicates", "adversarial",
               "long-context", "unique"]


def build_workload(per_slice: int) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (warmup_prompts, measured_items).

    The warm-up is TARGETED, not a blanket first pass. Warming everything
    would make every slice score 100% — including "unique" — which is
    logically correct and completely meaningless.

    The design that actually tests something: warm the ORIGINAL, then
    measure the VARIANT that must not match it.
    """
    warmup: list[str] = []
    measured: list[tuple[str, str]] = []

    # Warm A, then ask A twice. Both should hit.
    for index in range(per_slice):
        question = BASE_QUESTIONS[index % len(BASE_QUESTIONS)]
        warmup.append(question)
        measured.append(("exact duplicates", question))
        measured.append(("exact duplicates", question))

    # Warm B (clean), then ask B and a whitespace-mangled B. The mangled one
    # only hits if shaping canonicalised it.
    for index in range(per_slice):
        # Index suffix guarantees each prompt is unique within the pass, so a
        # mangled variant cannot be cached by its own earlier appearance.
        question = f"{NEAR_QUESTIONS[index % len(NEAR_QUESTIONS)]} #{index}"
        warmup.append(question)
        measured.append(("near duplicates", question))
        measured.append(("near duplicates", f"  {question}  \n\n"))

    # Warm C, then ask C' which differs by ONE WORD. Must MISS. This is the
    # false-hit test: a hit means the cache answered a different question.
    for original, variant in ADVERSARIAL_PAIRS:
        warmup.append(original)
        measured.append(("adversarial", variant))

    # Warm one long-context prompt, then ask others sharing the same long
    # prefix but a different question. Must MISS: an exact cache keys on the
    # whole prompt, not the prefix. This is what Day 9's semantic cache is for.
    warmup.append(LONG_PREFIX + f"Variant 0: {BASE_QUESTIONS[0]}")
    for index in range(1, per_slice + 1):
        measured.append(("long-context", LONG_PREFIX + f"Variant {index}: "
                         + BASE_QUESTIONS[index % len(BASE_QUESTIONS)]))

    # Never warmed. Must miss.
    for index in range(per_slice * 2):
        measured.append(("unique", f"unique question {index} about topic {index * 7}"))

    return warmup, measured


async def run(shape_mode: str, warmup: list[str],
              workload: list[tuple[str, str]]) -> dict:
    os.environ["KOSH_SHAPE"] = shape_mode
    os.environ["KOSH_REPLAY"] = "1"
    os.environ["KOSH_CACHE"] = "1"
    config = ReplayConfig(enabled=True, fixtures=_builtin_fixtures(), speed=0.0)
    app = create_app(config)

    outcomes: list[dict] = []
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            # WARM-UP: populate the cache, then discard. Measuring cold-cache
            # misses would only measure how many prompts are new, which says
            # nothing about the cache.
            for prompt in warmup:
                await http.post("/v1/chat/completions", json={
                    "model": "qwen-3.8-27b",
                    "messages": [{"role": "user", "content": prompt}],
                })
            app.state.cache.stats = type(app.state.cache.stats)()
            app.state.transport.calls = 0

            for slice_name, prompt in workload:
                started = time.perf_counter()
                response = await http.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwen-3.8-27b",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                outcomes.append({
                    "slice": slice_name,
                    "cache": response.headers.get("x-kosh-cache", "?"),
                    "prompt": prompt,
                    "text": response.json()["choices"][0]["message"]["content"],
                    "ms": (time.perf_counter() - started) * 1000,
                })

    return {
        "outcomes": outcomes,
        "stats": app.state.cache.stats.as_dict(),
        "upstream_calls": app.state.transport.calls,
    }


def analyse(result: dict) -> dict:
    outcomes = result["outcomes"]
    total = len(outcomes)

    # A false hit is any hit on a slice that should never hit.
    never_hit = {"adversarial", "long-context", "unique"}
    false_hits = [o for o in outcomes if o["slice"] in never_hit and o["cache"] == "hit"]

    slices: dict[str, dict] = {}
    for name in {o["slice"] for o in outcomes}:
        rows = [o for o in outcomes if o["slice"] == name]
        hits = sum(1 for r in rows if r["cache"] == "hit")
        slices[name] = {
            "requests": len(rows),
            "hits": hits,
            "hit_pct": round(100.0 * hits / len(rows), 1) if rows else 0.0,
        }

    tokens_served = sum(len(o["text"]) // 4 for o in outcomes if o["cache"] == "hit")
    tokens_total = sum(len(o["text"]) // 4 for o in outcomes)

    return {
        "total_requests": total,
        "total_hits": sum(1 for o in outcomes if o["cache"] == "hit"),
        "hit_pct": round(100.0 * sum(1 for o in outcomes if o["cache"] == "hit") / total, 1),
        "false_hits": len(false_hits),
        "false_hit_pct": round(100.0 * len(false_hits) / total, 1),
        "false_hit_examples": [
            {"slice": o["slice"], "prompt": o["prompt"][:70]} for o in false_hits[:5]
        ],
        "tokens_served_from_cache": tokens_served,
        "tokens_total": tokens_total,
        "upstream_calls": result["upstream_calls"],
        "slices": slices,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--shape", default="canonical",
                        choices=["off", "report", "canonical", "reorder"])
    parser.add_argument("--compare", action="store_true",
                        help="also run with shaping off, to show its contribution")
    args = parser.parse_args()

    warmup, workload = build_workload(max(1, args.requests // 5))
    print(f"Kosh cache benchmark · replay · {len(workload)} requests "
          f"· shape={args.shape}")
    print("No API key. No network. Exact numbers, no queue noise.\n")

    result = analyse(await run(args.shape, warmup, workload))

    print(f"{'slice':<18} {'requests':>9} {'hits':>6} {'hit %':>7}   expected")
    print("-" * 74)
    for name in SLICE_ORDER:
        if name in result["slices"]:
            row = result["slices"][name]
            print(f"{name:<18} {row['requests']:>9} {row['hits']:>6} "
                  f"{row['hit_pct']:>6.1f}%   {EXPECTED[name]}")
    print("-" * 74)
    print(f"{'TOTAL':<18} {result['total_requests']:>9} "
          f"{result['total_hits']:>6} {result['hit_pct']:>6.1f}%")

    print(f"\nupstream calls made: {result['upstream_calls']} of "
          f"{result['total_requests']} requests "
          f"({result['total_requests'] - result['upstream_calls']} avoided)")
    saved_pct = round(100.0 * result['tokens_served_from_cache']
                      / max(1, result['tokens_total']), 1)
    print(f"tokens served from cache: {result['tokens_served_from_cache']:,} of "
          f"{result['tokens_total']:,} ({saved_pct}%)")

    print(f"\nFALSE HITS (must be zero): {result['false_hits']} "
          f"({result['false_hit_pct']}%)")
    if result["false_hits"]:
        print("  !! BUG — the cache answered a question nobody asked:")
        for example in result["false_hit_examples"]:
            print(f"     [{example['slice']}] {example['prompt']}")
    else:
        print("  Clean. Every miss that should be a miss is a miss.")

    if args.compare:
        print("\n" + "=" * 60)
        print("Shaping contribution\n")
        off = analyse(await run("off", warmup, workload))
        print(f"{'slice':<20} {'shaped':>8} {'unshaped':>9} {'delta':>8}")
        print("-" * 47)
        for name in SLICE_ORDER:
            if name in result["slices"]:
                a = result["slices"][name]["hit_pct"]
                b = off["slices"].get(name, {}).get("hit_pct", 0.0)
                print(f"{name:<20} {a:>7.1f}% {b:>8.1f}% {a - b:>+7.1f}")
        print("\n'near duplicates' is where shaping earns its keep: identical")
        print("words with different whitespace are the same question, and only")
        print("canonicalisation makes them hash the same.")

    path = ROOT / "fixtures" / "bench_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
