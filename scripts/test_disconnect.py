"""
Day 2 disconnect test.

Opens a stream through Kosh, reads two chunks, then closes the connection
WITHOUT draining it — exactly what Ctrl-C in a terminal or closing a browser
tab mid-answer does.

WHY THIS MATTERS
If upstream.py does not close the upstream response when the client goes
away, we leak a provider concurrency slot on every abandoned request. On a
server with three slots, a few abandoned streams degrade service for
everyone. Bharat-Gate lists this as one of its planted bug scenarios.

HOW TO JUDGE IT
Run the server in one terminal, this script in another. When it finishes,
look at the SERVER terminal.

    PASS -> no traceback. Ideally a clean close.
    FAIL -> any traceback, or "connection was closed" errors.

    python scripts/test_disconnect.py
"""
from __future__ import annotations

import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

KOSH = "http://localhost:8000/v1"
KEY = os.getenv("BHARATCODE_API_KEY", "")
MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")


def abandon(prompt: str, chunks_to_read: int) -> None:
    """Open a stream, read a few chunks, then walk away."""
    print(f"  opening stream: {prompt[:48]!r}")
    with httpx.stream(
        "POST",
        f"{KOSH}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 400,          # long enough that we leave early
            "reasoning_effort": "none",
        },
        headers={"Authorization": f"Bearer {KEY}"},
        timeout=180.0,
    ) as response:
        print(f"  status {response.status_code}, reading {chunks_to_read} chunks")
        read = 0
        for line in response.iter_lines():
            if not line:
                continue
            read += 1
            print(f"    chunk {read}: {line[:90]}")
            if read >= chunks_to_read:
                break
        print(f"  abandoning after {read} chunks (stream not drained)")


def main() -> None:
    if not KEY:
        sys.exit("Set BHARATCODE_API_KEY in .env first")

    print("Make sure the Kosh server is running on :8000")
    print("Watch the SERVER terminal while this runs.\n")

    for attempt in (1, 2, 3):
        print(f"--- abandon #{attempt} ---")
        abandon("Explain the CAP theorem in detail, with examples.", 2)
        time.sleep(1)

    print("\nDone. Now check the server terminal.")
    print("  PASS = no traceback")
    print("  FAIL = any traceback or 'connection was closed' error")


if __name__ == "__main__":
    main()
