"""
Kosh — 60-second demo. No API key, no network, no GPU.

    python scripts/demo.py

Starts Kosh in replay mode if it is not already running, then walks through
everything the project does, in the order that makes the argument:

    1  the drop-in contract        (a normal OpenAI call, unmodified)
    2  prefix hygiene              (one timestamp costs the whole cache)
    3  exact cache                 (same question, zero upstream calls)
    4  shaping                     (whitespace variants are one question)
    5  failure behaviour           (retry + breaker under an injected storm)
    6  the receipt                 (ledger totals)

At the end it prints the dashboard URL. That is the thing to look at.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = os.getenv("KOSH_BASE", "http://127.0.0.1:8000")
PORT = int(os.getenv("KOSH_PORT", "8000"))

try:
    import httpx
except ImportError:
    sys.exit("Install dependencies first:  pip install -r requirements.txt")


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("-" * len(title))


def show(label: str, value, note: str = "") -> None:
    suffix = f"  \033[2m{note}\033[0m" if note else ""
    print(f"  {label:<26} {value}{suffix}")


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_server(url: str, seconds: int = 25) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1):
                return True
        except Exception:
            time.sleep(0.3)
    return False


def start_server() -> subprocess.Popen | None:
    """Run Kosh in replay mode so the demo needs nothing from the network."""
    env = {**os.environ,
           "KOSH_REPLAY": "1",
           "KOSH_SHAPE": "canonical",
           "KOSH_LEDGER": "1",
           "PYTHONPATH": str(ROOT)}
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "kosh.main:app",
         "--port", str(PORT), "--log-level", "error"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def ask(client: httpx.Client, prompt: str, system: str | None = None,
        stream: bool = False) -> httpx.Response:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return client.post(
        f"{BASE}/v1/chat/completions",
        json={"model": "qwen-3.8-27b", "messages": messages,
              "stream": stream, "max_tokens": 64},
        timeout=60.0,
    )


def main() -> None:
    print("\033[1mKosh\033[0m — the cheapest token is the one you never send\n")

    server = None
    if not port_open(PORT):
        print("starting Kosh in replay mode...")
        server = start_server()
        if not wait_for_server(BASE):
            server.terminate()
            sys.exit("Kosh failed to start. Run:  uvicorn kosh.main:app")
    else:
        print(f"using Kosh already running on {BASE}")

    try:
        run()
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=5)


def run() -> None:
    with httpx.Client() as client:
        rule("1  The drop-in contract")
        print("  A completely unmodified OpenAI-shaped call:")
        print("    client = OpenAI(base_url='http://localhost:8000/v1', api_key='...')")
        first = ask(client, "What is a hash table?")
        show("status", first.status_code, "200 = the contract holds")
        show("answer", first.json()["choices"][0]["message"]["content"][:44] + "...")

        rule("2  Prefix hygiene")
        clean = ask(client, "What is a queue?")
        poisoned = ask(client, "What is a queue?",
                       system="Current date: 2026-10-29T14:22:07Z\nYou are a tutor.")
        c = float(clean.headers.get("x-kosh-prefix-cacheable", 0))
        p = float(poisoned.headers.get("x-kosh-prefix-cacheable", 0))
        show("clean prompt", f"{c}% cacheable  ({clean.headers.get('x-kosh-prefix-verdict')})")
        show("one timestamp added", f"{p}% cacheable  ({poisoned.headers.get('x-kosh-prefix-verdict')})")
        print(f"\n  \033[1m{c - p:.0f} percentage points of prefix, lost to 30 characters.\033[0m")
        print("  vLLM matches prefixes from the first token, so everything")
        print("  after that timestamp is recomputed on every single request.")

        rule("3  Exact cache")
        a = ask(client, "Explain jitter in one sentence.")
        b = ask(client, "Explain jitter in one sentence.")
        show("first call", a.headers.get("x-kosh-cache"), "goes upstream")
        show("identical second call", b.headers.get("x-kosh-cache"), "zero upstream calls")
        show("byte-identical", a.json() == b.json(), "same id, same everything")

        rule("4  Shaping")
        variant = ask(client, "   Explain jitter in one sentence.  \n\n")
        show("whitespace variant", variant.headers.get("x-kosh-cache"),
             "hits only because shaping canonicalised it")

        rule("5  Failure behaviour")
        print("  Re-running under an injected 503 storm vs a naive client...")
        bench = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "bench_replay.py"),
             "--requests", "24"],
            cwd=str(ROOT), capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        for line in bench.stdout.splitlines():
            if line.startswith(("profile", "clean", "flaky", "storm",
                                "outage", "hostile", "-")):
                print("   ", line)

        rule("6  The receipt")
        stats = client.get(f"{BASE}/stats").json()
        totals = stats.get("totals") or {}
        if totals:
            show("requests", totals.get("requests"))
            show("tokens never sent", totals.get("tokens_saved"))
            show("upstream calls avoided", totals.get("upstream_calls_avoided"))
            show("cache hit rate", f"{totals.get('cache_hit_pct')}%")
            show("cache-hostile prompts", totals.get("cache_hostile_requests"))
            show("retries", totals.get("retries"))

    print(f"\n\033[1mLive dashboard:\033[0m  {BASE}/dashboard")
    print("Every number there comes from the local SQLite ledger.\n")


if __name__ == "__main__":
    main()
