# Kosh

> **The cheapest token is the one you never send — and on a shared GPU, every token you don't send is latency somebody else gets back.**

A drop-in, OpenAI-compatible gateway for the [BharatCode](https://bharatcode.ai) model API that cuts wasted GPU work, survives queue saturation, and shows you exactly what you spent.

Built for **BharatCode Build League Round 1** · Track: Coding & Developer Tools

---

## The one-line integration

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="bc-...")
```

That is the entire change. No other line of your code moves.

---

## Try it in 60 seconds — no API key, no account, no GPU

Kosh ships with a **replay mode**, so anyone can evaluate the whole system without touching BharatCode.

```bash
pip install -r requirements.txt
KOSH_REPLAY=1 uvicorn kosh.main:app
```

Then, in another terminal:

```bash
python scripts/demo.py
```

Open **http://localhost:8000/dashboard** to watch the numbers move.

Replay serves recorded responses and can inject real BharatCode failure modes — `429`, `503`, `timeout`, mid-stream `disconnect`, and `truncate`. Every one of those is a documented behaviour of the real API, not an invention.

---

## What we measured on the real API

The most useful thing in this repo is a number about BharatCode's own infrastructure.

BharatCode runs **vLLM**, which has **Automatic Prefix Caching** enabled by default: it hashes the prompt in blocks and skips prefill for any block it has already computed. But prefix matching runs **from the very first token**, so a single changed character near the front invalidates everything after it.

We measured this directly, using vLLM's own `prompt_tokens_details.cached_tokens` counter — not by timing, because TTFT on a shared GPU swings by 13 seconds and swamps any signal.

| Request | `prompt_tokens` | `cached_tokens` |
|---|---|---|
| repeated 2,093-token prompt | 2,093 | **800** |
| same prompt, **one character changed at the front** | 2,094 | **0** |
| completely different prompt | 2,109 | **0** |

**A one-character change is exactly as expensive as sending a completely different prompt.**

Which means any client that puts a timestamp, session id or request id at the top of a system prompt — the single most common pattern in production — gets **zero** prefix reuse on every call, and never finds out, because the requests succeed.

**This is what Kosh exists to fix.** It is also the one lever nobody else is pulling, for a reason that matters: every published approach to prefix caching (GKE Inference Gateway, llm-d, Alibaba ACK GIE) works by *routing requests to the replica that already holds the prefix*. Those report 57×–92% improvements, and all of them need multiple replicas and a cluster. BharatCode has **one GPU with three slots**. There is nowhere to route.

So Kosh fixes it from the only side it can: **the request**.

---

## What Kosh does

Nine stages. Seven are live.

| # | Stage | Status | What it does |
|---|---|---|---|
| 1 | auth | ✅ | Key passthrough — your allowance stays yours |
| 2 | **shape** | ✅ | Measures prefix stability; can move volatile markers out of the cacheable region |
| 3 | **cache** | ✅ | Exact-match cache keyed on the canonical request |
| 4 | route | ✅ | Any OpenAI-compatible provider |
| 5 | admit | ⏳ | Token bucket *(post-freeze)* |
| 6 | **call + retry** | ✅ | Full-jitter backoff + circuit breaker |
| 7 | **stream** | ✅ | Raw SSE passthrough, no buffering |
| 8 | **record** | ✅ | SQLite ledger + live dashboard |

### Streaming that actually streams

Raw byte passthrough. Kosh does not parse and re-frame SSE, because the provider already emits valid SSE and every re-frame step is a place to mishandle a keepalive or split a UTF-8 character.

`X-Accel-Buffering: no` is set, because nginx and most CDNs buffer responses by default and would silently convert your live token stream into one blob delivered at the end.

### Retry only the *open*, never the body

Once tokens are flowing, the only way to retry is to regenerate everything already produced — doubling cost for the user and load for the GPU. So both paths retry the **open** and never the body. That single rule explains most of `resilience.py`.

### The circuit breaker stops hammering a dead upstream

Under a simulated total outage (`python scripts/bench_replay.py`):

| profile | direct ok | Kosh ok | calls direct → Kosh | breaker |
|---|---|---|---|---|
| clean | 100% | 100% | 40 → 40 | 0 |
| flaky | 75% | 100% | 40 → 53 | 0 |
| storm | 25% | 100% | 40 → 159 | 0 |
| **outage** | 0% | 0% | **40 → 5** | 1 |
| hostile | 12.5% | 32.5% | 40 → 107 | 0 |

Read the `outage` row carefully: the upstream is 100% down, so **0% is the correct result for both**. Kosh's contribution there is **5 upstream calls instead of 40** — an 87% reduction in load on a service that is already dead. That is the thesis, measured.

Read the `storm` row honestly too: Kosh lifts completion from 25% to 100% and pays for it with **4× the upstream calls**. That tradeoff is real and it is printed by the script, not hidden.

### The cache refuses to be wrong

A false cache hit returns an answer to a question nobody asked, and it fails silently. So:

- **Truncated responses are never cached.** `finish_reason: "length"` arrives as **HTTP 200** with an incomplete answer. Caching it would serve that wrong answer to every future caller.
- Errors and dead streams are never cached.
- Hits return the **original payload byte-for-byte** — same `id`, same timestamps.

From `python scripts/bench_cache.py`:

| slice | requests | hit % | expected |
|---|---|---|---|
| exact duplicates | 16 | 100.0% | 100% |
| near duplicates (whitespace variants) | 16 | 100.0% | 100% shaped, 50% unshaped |
| **adversarial (one word changed)** | 3 | **0.0%** | 0% — a hit here is a bug |
| long-context (shared prefix) | 8 | 0.0% | 0% |
| unique | 16 | 0.0% | 0% |

**False hits: 0.** And shaping buys **+50 points** on near-duplicates: two prompts that differ only in whitespace are the same question, and only canonicalisation makes them hash the same.

---

## Prefix hygiene, in practice

Every response carries the measurement:

```
x-kosh-prefix-cacheable: 100.0      x-kosh-prefix-verdict: excellent
x-kosh-prefix-cacheable:  10.7      x-kosh-prefix-verdict: hostile
x-kosh-volatile-kinds: iso_timestamp
```

The second line is the same prompt with `"Current date: 2026-10-29T14:22:07Z\n"` prepended. Thirty characters, and they cost 21 of 24 cacheable prefix tokens.

**Shaping defaults to `report`.** It measures and tells you; it does not touch your prompt. Silently rewriting what someone asked is the fastest way to destroy trust in a proxy.

| `KOSH_SHAPE` | Behaviour |
|---|---|
| `report` *(default)* | Measure only. Body untouched. |
| `canonical` | Normalise whitespace so equivalent prompts are byte-identical |
| `reorder` | Additionally move volatile **lines** to the end — machine markers only, never prose |

---

## Admission, reasoning policy, and near-duplicate cache

Three optional stages. All three are off unless you turn them on, and each
one is measured below rather than asserted.

### Admission control — a cache hit must not be charged

`kosh/admission.py` is a per-key token bucket, but the interesting part is
that `check()` and `consume()` are deliberately separate calls. A request
that is served from cache never reaches the GPU, so it must never spend a
token. The gateway checks the bucket before doing any work, then consumes
only if the request actually went upstream:

    check()   -> admit?                      (every request)
    consume() -> spend                       (only if x-kosh-cache is absent)

Charging cached requests would mean a client that already saved you 100%
of the work gets rate-limited for it.

`KOSH_RATE_LIMIT=60` with 65 rapid requests: 60 admitted, 5 rejected with
429 and a `Retry-After`.

### Reasoning policy — only ever downgrades

On Day 1 one request took 141.6 s while its neighbours took 11-37 s. The
cause was not the GPU: `qwen-3.8-27b` emits `<think>` tokens and those
count against `max_tokens`, so the model spent its whole output budget
thinking and returned a truncated answer with a 200 status. Setting
`reasoning_effort="none"` dropped that run to 19 s.

`kosh/reasoning.py` turns that accident into a policy. The rule is
asymmetric on purpose:

- it will only ever **downgrade** to `reasoning_effort: "none"`, never upgrade
- it never overrides a value the caller set explicitly
- if the classifier is unsure, it leaves the request alone

Removing reasoning from a task that needed it produces a worse answer;
leaving it on a task that didn't only costs time. Of those two failures the
first is worse, so the classifier errs toward keeping it.

Measured over 4 prompts with `KOSH_REASONING=auto`: 3 downgraded, 1 kept,
0 caller-set values overridden.

### Near-duplicate cache — and what it is not

This is **lexical near-duplicate matching: Jaccard similarity over
normalised token sets, with an inverted index so only plausible candidates
are scored.** It is not embedding-based semantic similarity, and the
difference matters:

| it catches | it does not catch |
|---|---|
| case and punctuation changes | paraphrase |
| whitespace and filler words | synonym swaps |
| token reordering | different meaning, same shape |

Measured: "Please explain the difference between TCP and UDP in computer
networking" vs "Explain the difference between TCP and UDP in computer
networking" scores **1.000**. "How do I sort a list?" vs "What is the best
way to order these items?" scores **0.000**. "Explain why the sky looks
blue" vs "What causes the sky to appear blue in colour" scores **0.286**.
Calling it semantic would be overclaiming, so it isn't called that in the
code either.

**The benchmark** (`scripts/bench_semantic.py`) — 12 prompts x 5 variants,
exact cache on in both runs, so the gain is attributable to this feature
alone:

| variant | exact only | + near-dup | gain | false hits |
|---|---|---|---|---|
| exact repeat | 12/12 | 12/12 | 0 | – |
| punctuation | 0/12 | 12/12 | +12 | – |
| whitespace | 0/12 | 12/12 | +12 | – |
| filler word | 0/12 | 12/12 | +12 | – |
| **adversarial** (one noun swapped) | 0/12 | 0/12 | 0 | **0** |
| **TOTAL** | **12/60** | **48/60** | **+36** | **0** |

36 of 60 upstream calls avoided (60.0 pp), 0 false hits.

**Where it breaks** — the threshold is a dial between hit rate and
correctness. Turning it down buys hits and pays in false hits, so the cliff
is measured rather than guessed (`--sweep`):

| threshold | 0.50 | 0.60 | 0.70 | 0.75 | 0.80 | 0.92 | 0.95 |
|---|---|---|---|---|---|---|---|
| served | 59 | 57 | 52 | 50 | 48 | 48 | 48 |
| false hits | 11 | 9 | 4 | 2 | **0** | **0** | **0** |

The cliff sits between 0.75 and 0.80. The default of 0.92 is chosen to sit
clear of it, not to maximise hit rate — at 0.50 the cache answers 11 of 12
questions nobody asked, which is a silent correctness failure and the one
outcome this feature must never produce.

Two further limits, both deliberate: prompts with fewer than
`KOSH_SEMANTIC_MIN_TOKENS` content tokens skip fuzzy matching entirely (so
"capital of India?" vs "capital of Nepal?" never reaches the scorer), and
entries are only stored when `finish_reason` is a clean stop — never a
truncation or an error, the same rule the exact cache follows.

## Running it for real

```bash
cp .env.example .env      # add BHARATCODE_API_KEY
uvicorn kosh.main:app
```

Kosh is provider-agnostic. Point it anywhere OpenAI-compatible:

```bash
KOSH_UPSTREAM_BASE=http://localhost:11434/v1 KOSH_MODEL=qwen2.5 uvicorn kosh.main:app
```

| Variable | Default | Meaning |
|---|---|---|
| `KOSH_UPSTREAM_BASE` | `https://bharatcode.ai/api/model/v1` | Any OpenAI-compatible endpoint |
| `KOSH_MODEL` | `qwen-3.8-27b` | Default model |
| `KOSH_UPSTREAM_KEY` | *(empty)* | Empty = caller's key passes through |
| `KOSH_REPLAY` | `0` | `1` = serve recorded fixtures, no upstream |
| `KOSH_REPLAY_PROFILE` | `clean` | `clean` `flaky` `storm` `outage` `hostile` |
| `KOSH_SHAPE` | `report` | `off` `report` `canonical` `reorder` |
| `KOSH_CACHE` | `1` | Set `0` to disable caching |
| `KOSH_CACHE_TTL` | `3600` | Entry lifetime, seconds |
| `KOSH_BREAKER_THRESHOLD` | `5` | Failures before the circuit opens |
| `KOSH_LEDGER` | `1` | SQLite request log |
| `KOSH_MAX_ATTEMPTS` | `4` | Retry attempts |
| `KOSH_BACKOFF_BASE` / `_CAP` | `2.0` / `16.0` | Backoff ceiling, seconds |

---

## Tests

```bash
pytest -q
```

**164 tests in about 30 seconds. No network. No API key.**

| File | Tests | Covers |
|---|---|---|
| `test_resilience.py` | 22 | Backoff bounds, jitter desynchronisation, breaker state machine, retry counts, response closing |
| `test_replay.py` | 18 | Determinism, byte-for-byte contract, fault injection driving the real retry path |
| `test_shaping.py` | 37 | Volatile detection, cacheable scoring, safe reordering |
| `test_cache.py` | 28 | Keying, truncation rejection, LRU, TTL, end-to-end hits |
| `test_stretch.py` | 59 | Token bucket refill/burst, reasoning classifier, Jaccard scoring, adversarial pairs, and the wiring regressions below |

The replay tests are the ones worth reading. `ReplayTransport` replaces the **transport, not the logic** — it implements only `build_request` and `send`, so injected 503s travel through the *same* retry loop, jitter and circuit breaker as real ones. Mocking the handler instead would bypass everything the tests claim to cover.

`test_stretch.py` exists because of four bugs that **all unit tests passed
straight through**, and which only appeared by running the system:

1. `ExactCache.store(key, lines, text, ...)` and `SemanticCache.store(key,
   text, lines, ...)` take their arguments in **opposite order**. Both call
   sites passed the *completion* into the prompt slot, so every entry was
   indexed by the model's answer and no near-duplicate could ever hit.
2. The near-duplicate `store()` was nested inside the exact cache's
   enabled-check, so `KOSH_CACHE=0` silently disabled both.
3. The streaming tee that buffers bytes for caching was gated on the exact
   cache too — with it off, the buffer stayed empty and every streaming
   store was rejected for an empty `finish_reason`.
4. Making the cache key available to the semantic store also made it
   available to the exact lookup, so `KOSH_CACHE=0` stopped disabling
   anything. The key and the lookup are two different things.
5. The streaming cache store lived in an `else:` clause. Closing an async
   generator throws `GeneratorExit` at its last `yield`, which is not an
   `httpx.HTTPError`, so it propagates and Python **skips `else:` entirely**
   — the store never ran and no streamed response was ever cached. Whether
   the ASGI layer closes the generator before it resumes past the final
   yield is platform- and timing-dependent, so identical code cached every
   streamed response on Linux and none at all on Windows. The store is now
   in a `finally:`, and completion is detected either from normal iteration
   or from the provider's terminal `data: [DONE]` frame.

Bug 5 is the one worth remembering: it shipped green through 163 tests and
only appeared on another machine. `test_stream_store_survives_generator_closed_before_it_resumes`
reproduces the Windows behaviour directly by consuming up to the terminal
frame and then closing the generator instead of letting it resume.

Each of the four now has a regression test that was **verified to fail when
the bug is reintroduced** — and every one is parameterised over both the
streamed and non-streamed path, because the wiring exists twice in
`upstream.py` and a test covering only one lets the bug live in the other.

---

## What this does not prove

*(See also the near-duplicate cache's own limits in the section above: it is lexical, and paraphrase scores 0.000-0.286.)*

Published alongside the results, because a benchmark that lists no weaknesses is marketing.

1. **The prefix-cache numbers are direct counters, not timings.** They are immune to queue noise. Timing-based results on this GPU are not: within a single phase, TTFT ranged 2.4s–16.9s, while the difference between experimental conditions was 2.7s. **We tried to measure prefix reuse by timing and the shared queue made it unmeasurable.** That negative result is why we switched to `cached_tokens`.
2. **Under `storm`, Kosh amplifies upstream load 4×.** Retries buy completion with calls. The circuit breaker is what stops this under a genuine outage.
3. **Exact caching gets 0% on shared-prefix, different-question traffic.** That is correct behaviour and it is the baseline a semantic cache would have to beat.
4. **The prefix cache saturating at 800 tokens is unexplained.** It is repeatable (800, 800, 800) but we could not determine whether it is a configuration cap or eviction under load — BharatCode's configuration is proprietary. We publish the measurement and the open question.
5. **Latency figures in the dashboard exclude cached requests.** A cached response returns in well under a millisecond; including those would flatter the numbers and tell the operator nothing.

---

## Project layout

```
kosh/
├── kosh/
│   ├── main.py        FastAPI app, 9-stage pipeline, /dashboard, /stats
│   ├── shaping.py     prefix hygiene: detect, score, safely reorder
│   ├── cache.py       exact cache; refuses truncated and errored responses
│   ├── semantic.py    near-duplicate cache (lexical, not embeddings)
│   ├── admission.py   per-key token bucket; check and consume are separate
│   ├── reasoning.py   reasoning policy; only ever downgrades
│   ├── resilience.py  full-jitter backoff, Retry-After, circuit breaker
│   ├── replay.py      deterministic replay + fault injection
│   ├── ledger.py      SQLite WAL request log
│   ├── dashboard.py   single-file HTML dashboard, no CDN
│   └── upstream.py    SSE passthrough and non-streaming paths
├── tests/             164 tests, no network
├── scripts/
│   ├── demo.py                 one-command walkthrough (works offline)
│   ├── bench_cache.py          cache hit rate + false-hit detection
│   ├── bench_replay.py         failure behaviour vs a naive client
│   ├── bench_semantic.py       near-duplicate gain + false-hit cliff sweep
│   ├── bench_prefix.py         live prefix experiment (uses real tokens)
│   ├── probe_cached_tokens.py  15-second direct prefix-cache probe
│   └── smoke.py                A/B harness against the live API
└── fixtures/          recorded responses
```

---

## Roadmap

The three stages listed here previously as future work are now built and measured above: near-duplicate cache, token-bucket admission, and reasoning policy.

Deliberately **not** planned: React, Docker, Redis, an ORM, or a vector database — each is infrastructure theatre at this scale, and every one costs days.

The one change that would move the numbers most is real embedding-based similarity, which would catch paraphrase and synonym swaps (measured at 0.000-0.286 Jaccard today, i.e. complete misses). That needs a model to embed with, which on a shared GPU means spending the very tokens this project is trying not to spend. It is the right next step, and it is not free.

---

## License

MIT
