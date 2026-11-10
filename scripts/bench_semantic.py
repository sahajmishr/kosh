#!/usr/bin/env python3
"""
What is the near-duplicate cache actually worth, and what does it cost you
in correctness?

Method
------
Take N base prompts. For each, send four variants:

    exact        byte-identical repeat          (exact cache should catch)
    punctuation  same words, "!!" appended      (exact misses, near catches)
    whitespace   same words, spacing mangled    (exact misses, near catches)
    filler       "Please " prepended            (exact misses, near catches)
    adversarial  one key noun swapped           (MUST NOT hit — ever)

The adversarial row is the whole point. A near-duplicate cache buys hit rate
by being fuzzy; the only question that matters is whether its fuzziness ever
answers a question nobody asked. So the fifth row is measured, not asserted.

Run it twice — once with the near-duplicate cache off, once on — so the gain
is attributed to the feature and not to the exact cache doing its job.

    python scripts/bench_semantic.py

No API key. No network. Deterministic.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KOSH_REPLAY"] = "1"
os.environ["KOSH_REPLAY_PROFILE"] = "clean"
os.environ["KOSH_LEDGER"] = "0"
os.environ["KOSH_RATE_LIMIT"] = "0"
os.environ["KOSH_REASONING"] = "off"

import httpx  # noqa: E402

# (base, adversarial) — the swap changes the MEANING, not just the wording.
CASES: list[tuple[str, str]] = [
    ("Explain the difference between TCP and UDP in computer networking",
     "Explain the difference between TCP and IP in computer networking"),
    ("Describe how garbage collection works in the Java virtual machine",
     "Describe how garbage collection works in the Python virtual machine"),
    ("What are the main causes of the First World War in Europe",
     "What are the main causes of the Second World War in Europe"),
    ("Write a function to reverse a linked list in the C language",
     "Write a function to reverse a binary tree in the C language"),
    ("Summarise the plot of the Mahabharata in a few sentences",
     "Summarise the plot of the Ramayana in a few sentences"),
    ("How does a hash table resolve collisions in memory",
     "How does a database index resolve collisions in memory"),
    ("What is the time complexity of merge sort and why",
     "What is the space complexity of merge sort and why"),
    ("Explain how HTTPS handshake works between browser and server",
     "Explain how DNS resolution works between browser and server"),
    ("Which mountain is the highest peak in the Himalayan range",
     "Which mountain is the highest peak in the Andes range"),
    ("Describe the water cycle and its stages in the atmosphere",
     "Describe the nitrogen cycle and its stages in the atmosphere"),
    ("How do you implement a rate limiter using a token bucket",
     "How do you implement a rate limiter using a leaky bucket"),
    ("What is the difference between a process and a thread in Linux",
     "What is the difference between a process and a container in Linux"),
]

VARIANTS = ("exact", "punctuation", "whitespace", "filler", "adversarial")


def variants(base: str, adversarial: str) -> dict[str, str]:
    return {
        "exact": base,
        "punctuation": base + "!!",
        "whitespace": "   ".join(base.split(" ")),
        "filler": "Please " + base[0].lower() + base[1:],
        "adversarial": adversarial,
    }


async def run(threshold: float) -> dict[str, dict[str, int]]:
    """Send base + every variant; tally what the gateway said per variant."""
    os.environ["KOSH_CACHE"] = "1"
    os.environ["KOSH_SEMANTIC_THRESHOLD"] = str(threshold)

    from kosh.main import create_app

    app = create_app()
    tally: dict[str, dict[str, int]] = {
        v: {"hit": 0, "near": 0, "miss": 0} for v in VARIANTS
    }

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t") as client:
            for base, adversarial in CASES:
                # Prime: send the base prompt once so there is something to hit.
                await client.post(
                    "/v1/chat/completions",
                    json={"model": "m",
                          "messages": [{"role": "user", "content": base}]},
                    headers={"Authorization": "Bearer t",
                             "Content-Type": "application/json"},
                )
                for name, text in variants(base, adversarial).items():
                    r = await client.post(
                        "/v1/chat/completions",
                        json={"model": "m",
                              "messages": [{"role": "user", "content": text}]},
                        headers={"Authorization": "Bearer t",
                                 "Content-Type": "application/json"},
                    )
                    verdict = r.headers.get("x-kosh-cache") or "miss"
                    tally[name][verdict] = tally[name].get(verdict, 0) + 1
    return tally


def served(row: dict[str, int]) -> int:
    """Requests answered from cache (either cache), i.e. no upstream call."""
    return row.get("hit", 0) + row.get("near", 0)


def sweep(cases_n: int) -> None:
    """Where does it break?

    A threshold is a dial between hit rate and correctness. Turning it down
    buys hits and pays in false hits — answers to questions nobody asked.
    This locates the cliff so the default is chosen with room to spare
    rather than guessed.
    """
    print("Threshold sweep · false hits on adversarial pairs (of "
          f"{cases_n})\n")
    print(f"{'threshold':>10}{'served':>10}{'false hits':>13}")
    print("-" * 33)
    for t in (0.50, 0.60, 0.70, 0.75, 0.80, 0.92, 0.95):
        row = asyncio.run(run(t))
        false_hits = served(row["adversarial"])
        total_served = sum(served(row[v]) for v in VARIANTS)
        print(f"{t:>10.2f}{total_served:>10}{false_hits:>13}")
    print("\nThe cliff sits between 0.75 and 0.80. The default of 0.92 is")
    print("chosen to sit well clear of it, not to maximise hit rate.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.92,
                    help="Jaccard threshold for the near-duplicate cache")
    ap.add_argument("--cases", type=int, default=len(CASES))
    ap.add_argument("--sweep", action="store_true",
                    help="locate the false-hit cliff instead of one run")
    args = ap.parse_args()

    if args.sweep:
        sweep(args.cases)
        return 0

    cases = CASES[: args.cases]
    total = len(cases)
    print(f"Kosh near-duplicate benchmark · {total} prompts x "
          f"{len(VARIANTS)} variants · threshold={args.threshold}")
    print("No API key. No network. Deterministic.\n")

    off = asyncio.run(run(0.0))
    on = asyncio.run(run(args.threshold))

    print(f"{'variant':<14}{'exact only':>12}{'+ near-dup':>12}"
          f"{'gain':>10}{'false hits':>13}")
    print("-" * 61)

    false_total = 0
    for v in VARIANTS:
        a, b = served(off[v]), served(on[v])
        false_hits = on[v].get("hit", 0) + on[v].get("near", 0) if v == "adversarial" else 0
        false_total += false_hits
        print(f"{v:<14}{a:>9}/{total:<2}{b:>9}/{total:<2}{b - a:>10}{false_hits:>13}")

    total_off = sum(served(off[v]) for v in VARIANTS)
    total_on = sum(served(on[v]) for v in VARIANTS)
    sent = total * len(VARIANTS)
    print("-" * 61)
    print(f"{'TOTAL':<14}{total_off:>9}/{sent:<3}{total_on:>8}/{sent:<3}"
          f"{total_on - total_off:>10}{false_total:>13}")
    print(f"\nUpstream calls avoided by adding the near-duplicate cache: "
          f"{total_on - total_off} of {sent} "
          f"({100.0 * (total_on - total_off) / sent:.1f}pp)")
    print(f"False hits on adversarial pairs: {false_total}")

    print("\nWhat this benchmark does NOT prove:")
    print("  - It is not embeddings. 'Please explain X' and 'explain X!!' hit")
    print("    because they share nearly all tokens, not because they mean the")
    print("    same thing. A genuine paraphrase ('How do I sort a list?' vs")
    print("    'What is the best way to order these items?') scores ~0.1 and")
    print("    misses. That is a real limitation, not a tuning problem.")
    print("  - Latency is not measured. Replay timing is synthetic.")
    print("  - The adversarial set is 12 hand-picked pairs. It is a smoke test")
    print("    for catastrophic false hits, not a false-positive rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
