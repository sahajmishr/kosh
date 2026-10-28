"""
Day 2 probe — does the upstream actually stream on /chat/completions?

Run BEFORE building on the assumption:

    python scripts/probe_stream.py

What you want:
    status: 200
    content-type: text/event-stream
    lines beginning with  data: {

If you get 400/406, or a single JSON blob with no `data:` lines, tell me —
we switch to their /responses endpoint and translate the frames instead.
"""
from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("KOSH_UPSTREAM_BASE", "https://bharatcode.ai/api/model/v1")
KEY = os.getenv("BHARATCODE_API_KEY", "")
MODEL = os.getenv("KOSH_MODEL", "qwen-3.8-27b")
URL = f"{BASE}/chat/completions"
MAX_LINES = 20


def main() -> None:
    if not KEY:
        sys.exit("Set BHARATCODE_API_KEY in .env first")

    print(f"probing {URL}\n  model={MODEL}\n")

    with httpx.stream(
        "POST",
        URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "stream": True,
            "max_tokens": 64,
        },
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        timeout=180.0,
    ) as response:
        print("status      :", response.status_code)
        print("content-type:", response.headers.get("content-type"))

        if response.status_code != 200:
            print("\nbody:", response.read().decode("utf-8", "replace")[:800])
            print("\nSTREAM FAILED — send this output back.")
            return

        streaming = "event-stream" in (response.headers.get("content-type") or "")
        printed = 0
        for line in response.iter_lines():
            if not line:
                continue
            print(f"  {line[:150]}")
            printed += 1
            if printed >= MAX_LINES:
                print("  ... (truncated)")
                break

    print("\nSTREAM OK" if streaming else "\nWARNING: 200 but not event-stream")


if __name__ == "__main__":
    main()
