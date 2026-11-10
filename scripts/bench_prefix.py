"""
Day 5 — the prefix experiment (LIVE, uses real BharatCode tokens).

Does BharatCode's vLLM actually reuse prefill across requests?

vLLM ships with Automatic Prefix Caching enabled by default. If it is live,
then two requests sharing a long prefix should have dramatically different
time-to-first-token: the first pays for prefill, the second skips it.

Nobody in this competition will have measured this. It is also the single
most useful number BharatCode's operator could have, because it tells them
how much of their GPU's prefill compute is currently being wasted.

THE THREE PHASES
  warm     the same long prefix every time  -> should get faster after #1
  cold     a different long prefix each time -> every one is a full prefill
  hostile  the same prefix, but one character at the front changed
           -> should collapse back to cold, proving matching is from the start

WHY SEQUENTIAL
Concurrency interleaves requests and muddies the cache signal. This measures
one request at a time so the effect is unambiguous.

Usage:
    python scripts/bench_prefix.py
    python scripts/bench_prefix.py --reps 12 --prefix-tokens 6000
    python scripts/bench_prefix.py --mode kosh --rounds 3
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
KOSH = os.getenv("KOSH_BASE", "http://localhost:8000/v1")
MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")

FIXTURES = ROOT / "fixtures"

QUESTIONS = [
    "What is the first rule listed?",
    "Summarise section two in one sentence.",
    "Which item is marked deprecated?",
    "What timeout value is recommended?",
    "Name the two error codes mentioned.",
    "What does the appendix cover?",
    "Which field is required in every request?",
    "What is the stated retry policy?",
    "Who is the document intended for?",
    "What is the last recommendation?",
    "Which section is longest?",
    "What value should the header carry?",
]

# A synthetic but realistic-looking reference document. Long, stable, and
# identical across the warm phase — exactly the shape of the system prompts,
# API docs and code contexts that real applications send every single call.
DOC_PARAGRAPHS = [
    "The gateway accepts requests on /v1/chat/completions and expects a JSON "
    "body with a model field, an array of messages, and optional sampling "
    "parameters. Requests without a model field are rejected with HTTP 400.",

    "Responses are streamed using Server-Sent Events. Each chunk is a JSON "
    "object prefixed with 'data: ' and terminated by two newlines. The stream "
    "ends with a literal data: [DONE] line, which clients must handle.",

    "Clients should set the Authorization header to a bearer token. Tokens are "
    "scoped per account and must never be embedded in client-side code. Rotate "
    "them at least every ninety days, and immediately on any suspected leak.",

    "Rate limits are enforced per account. A HTTP 429 response includes a "
    "Retry-After header stating how long to wait. A HTTP 503 means the model "
    "is busy; wait a few seconds and retry with exponential backoff.",

    "The recommended connect timeout is ten seconds and the recommended read "
    "timeout is one hundred and eighty seconds, because inference on shared "
    "hardware can legitimately queue for a considerable period under load.",

    "Two error codes deserve special handling. HTTP 402 indicates a billing or "
    "entitlement problem and must never be retried. HTTP 504 indicates an "
    "upstream timeout and is safe to retry after a short delay.",

    "Prompts should place stable content first and volatile content last. "
    "Timestamps, request identifiers and session tokens at the front of a "
    "prompt prevent prefix reuse and measurably increase time to first token.",

    "The output token limit is thirty-two thousand tokens and includes any "
    "reasoning tokens the model emits internally. A response that ends with "
    "finish_reason 'length' was truncated and should be treated as incomplete.",
]


def build_prefix(target_tokens: int) -> str:
    """Repeat the document until it is roughly target_tokens long."""
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


def unique_prefix(seed: int, target_tokens: int) -> str:
    """Same size and shape, but guaranteed to differ from every other seed.

    The variation is spread through the text, not just at the front, so these
    prefixes share no reusable blocks with each other.
    """
    base = build_prefix(target_tokens)
    return f"[corpus revision {seed} · internal build {seed * 7919}]\n\n{base}"


def call(url: str, system: str, question: str, max_tokens: int = 16) -> dict:
    """One request. Returns TTFT separately from total, as always."""
    started = time.perf_counter()
    ttft = None
    chunks = 0
    status = 0
    try:
        with httpx.stream(
            "POST",
            f"{url}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ],
                "stream": True,
                "max_tokens": max_tokens,
                "reasoning_effort": "none",
            },
            headers={"Authorization": f"Bearer {KEY}"},
            timeout=180.0,
        ) as response:
            status = response.status_code
            for _line in response.iter_lines():
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                chunks += 1
    except httpx.HTTPError as exc:
        return {"status": 0, "error": repr(exc), "ttft_ms": None, "total_ms": None}

    total = (time.perf_counter() - started) * 1000
    return {
        "status": status,
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "total_ms": round(total, 1),
        "chunks": chunks,
    }


def run_phase(name: str, url: str, systems: list[str], reps: int) -> list[dict]:
    results = []
    for index in range(reps):
        system = systems[index % len(systems)]
        question = QUESTIONS[index % len(QUESTIONS)]
        row = call(url, system, question)
        row.update(phase=name, index=index)
        results.append(row)
        state = "OK " if row["status"] == 200 else "ERR"
        ttft = row["ttft_ms"]
        print(f"  [{state}] {name:<8} #{index:<3} ttft={str(ttft):>9}ms  chunks={row.get('chunks', 0)}")
    return results


def median(values: list[float]) -> float:
    return round(statistics.median(values), 1) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "kosh"], default="direct")
    parser.add_argument("--reps", type=int, default=8, help="requests per phase")
    parser.add_argument("--prefix-tokens", type=int, default=2500)
    parser.add_argument("--rounds", type=int, default=2,
                        help="repeat the whole experiment; median across rounds")
    args = parser.parse_args()

    if not KEY and args.mode == "direct":
        sys.exit("Set BHARATCODE_API_KEY in .env first")

    url = BASE if args.mode == "direct" else KOSH
    shared = build_prefix(args.prefix_tokens)
    hostile = ("X" if not shared.startswith("X") else "Y") + shared[1:]
    colds = [unique_prefix(seed, args.prefix_tokens) for seed in range(args.reps)]

    approx = args.reps * 3 * args.prefix_tokens * args.rounds
    print(f"Kosh prefix experiment · mode={args.mode} · {args.reps} reps/phase "
          f"· {args.rounds} rounds")
    print(f"prefix ~{args.prefix_tokens} tokens · est. input tokens ~{approx:,}")
    print(f"target {url}\n")
    print("NOTE: screenshot bharatcode.ai/livestats before and after — TTFT on a")
    print("      shared GPU is dominated by queue depth, which we do not control.\n")

    rounds = []
    for round_index in range(args.rounds):
        print(f"--- round {round_index + 1}/{args.rounds} ---")
        warm = run_phase("warm", url, [shared], args.reps)
        cold = run_phase("cold", url, colds, args.reps)
        hostile_rows = run_phase("hostile", url, [hostile], args.reps)
        rounds.append({"warm": warm, "cold": cold, "hostile": hostile_rows})
        print()

    def ttfts(phase: str, drop_first: bool = False) -> list[float]:
        values: list[float] = []
        for r in rounds:
            rows = r[phase]
            if drop_first:
                rows = rows[1:]
            values += [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
        return values

    warm_first = [r["warm"][0]["ttft_ms"] for r in rounds if r["warm"][0]["ttft_ms"]]
    warm_rest = ttfts("warm", drop_first=True)
    cold_all = ttfts("cold")
    hostile_all = ttfts("hostile")

    print("=" * 66)
    print(f"{'phase':<26} {'median TTFT':>12} {'mean':>10} {'n':>4}")
    print("-" * 66)
    for label, values in [
        ("warm  (first request)", warm_first),
        ("warm  (after cache fill)", warm_rest),
        ("cold  (unique prefix)", cold_all),
        ("hostile (1 char changed)", hostile_all),
    ]:
        mean = round(statistics.mean(values), 1) if values else 0.0
        print(f"{label:<26} {median(values):>9}ms {mean:>9}ms {len(values):>4}")

    print("-" * 66)
    if warm_rest and cold_all:
        ratio = round(statistics.median(cold_all) / statistics.median(warm_rest), 2)
        saved = round(100 * (1 - statistics.median(warm_rest) / statistics.median(cold_all)), 1)
        print(f"cold / warm  = {ratio}x     warm saves {saved}% of TTFT")

    if warm_first and warm_rest:
        print(f"warm first vs warm rest = "
              f"{round(statistics.median(warm_first) / statistics.median(warm_rest), 2)}x")

    verdict = "inconclusive"
    if warm_rest and cold_all and len(warm_rest) >= 3:
        if statistics.median(warm_rest) < 0.75 * statistics.median(cold_all):
            verdict = "PREFIX CACHING IS LIVE"
        else:
            verdict = "no measurable prefix reuse"
    print(f"\nverdict: {verdict}")
    print("\nCaveat: TTFT on a shared GPU moves with queue depth. If warm and cold")
    print("differ by less than the spread you see within a single phase, the result")
    print("is noise — re-run when livestats shows a shorter queue.")

    path = FIXTURES / "bench_prefix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "mode": args.mode,
        "reps": args.reps,
        "rounds": args.rounds,
        "prefix_tokens": args.prefix_tokens,
        "rounds_data": rounds,
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
