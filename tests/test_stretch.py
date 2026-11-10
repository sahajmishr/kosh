"""
Day 9 stretch tests: admission control, reasoning policy, near-duplicate cache.

The near-duplicate tests matter most. A false hit answers a question nobody
asked, silently, so the adversarial pairs are the whole point of the file.
"""
from __future__ import annotations

import pytest

from kosh.admission import TokenBucket, bucket_from_env
from kosh.reasoning import (
    NONE,
    PolicyStats,
    apply_policy,
    classify,
    contains_code,
)
from kosh.semantic import SemanticCache, jaccard, normalise


# ======================================================= admission =========

def test_disabled_by_default():
    assert TokenBucket(0, 0).enabled is False
    assert TokenBucket(0, 0).check("k") == (True, 0.0)


def test_allows_up_to_burst():
    bucket = TokenBucket(rate_per_minute=60, burst=3)
    for _ in range(3):
        assert bucket.check("k")[0] is True
        bucket.consume("k")
    assert bucket.check("k")[0] is False


def test_rejection_carries_a_retry_after():
    bucket = TokenBucket(rate_per_minute=60, burst=1)
    bucket.consume("k")
    allowed, wait = bucket.check("k")
    assert allowed is False
    assert wait > 0


def test_check_does_not_consume():
    """The two-phase design: peeking must be free, or every cache hit would
    silently burn budget."""
    bucket = TokenBucket(rate_per_minute=60, burst=2)
    for _ in range(10):
        assert bucket.check("k")[0] is True
    assert bucket.remaining("k") == 2.0


def test_refill_restores_capacity():
    bucket = TokenBucket(rate_per_minute=60, burst=2)
    bucket.consume("k")
    bucket.consume("k")
    assert bucket.check("k")[0] is False
    # Backdate to simulate a minute passing; deterministic, no sleeping.
    bucket._buckets["k"].updated -= 60
    assert bucket.check("k")[0] is True


def test_refill_is_capped_at_burst():
    bucket = TokenBucket(rate_per_minute=60, burst=5)
    bucket.check("k")                         # create the bucket first
    bucket._buckets["k"].updated -= 3600      # an hour passes
    assert bucket.remaining("k") == 5.0, "must not exceed burst"


def test_buckets_are_per_key():
    bucket = TokenBucket(rate_per_minute=60, burst=1)
    bucket.consume("alice")
    assert bucket.check("alice")[0] is False
    assert bucket.check("bob")[0] is True


def test_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv("KOSH_RATE_LIMIT", raising=False)
    assert bucket_from_env().enabled is False


def test_env_reads_rate(monkeypatch):
    monkeypatch.setenv("KOSH_RATE_LIMIT", "30")
    monkeypatch.setenv("KOSH_RATE_BURST", "10")
    bucket = bucket_from_env()
    assert bucket.rate_per_minute == 30
    assert bucket.burst == 10


# ======================================================= reasoning =========

def test_code_is_left_alone():
    assert contains_code("def f(x):\n    return x") is True
    assert contains_code("What is the capital of India?") is False


@pytest.mark.parametrize("prompt", [
    "What is the capital of India?",
    "Who was the first Prime Minister of India?",
    "Translate hello to Hindi.",
])
def test_trivial_questions_drop_reasoning(prompt):
    assert classify(prompt) == "skip"


@pytest.mark.parametrize("prompt", [
    "Debug this deadlock in the worker pool",
    "What is the time complexity of quicksort?",
    "Implement a binary search in Python",
    "Why does this SQL query return duplicates?",
    "def broken(x):\n    return x +",
])
def test_hard_tasks_keep_reasoning(prompt):
    assert classify(prompt) == "reason"


def test_unknown_prompt_defaults_to_keeping_reasoning():
    """Conservative: removing reasoning from a task that needed it is worse
    than leaving it on a task that didn't. Short prompts are treated as
    lookups, so the ambiguous case must be long enough to not match the
    trivial-signal pattern."""
    assert classify("") == "reason"
    assert classify(
        "Consider the various implications of this approach for the wider system"
    ) == "reason"


def test_long_prompts_keep_reasoning():
    assert classify("word " * 200) == "reason"


def test_apply_sets_none_for_trivial():
    body, action = apply_policy({"model": "m"}, "What is the capital of India?")
    assert action == NONE
    assert body["reasoning_effort"] == "none"


def test_apply_leaves_hard_tasks_untouched():
    body, action = apply_policy({"model": "m"}, "Debug this deadlock")
    assert action == "kept"
    assert "reasoning_effort" not in body


def test_apply_never_overrides_the_caller():
    body, action = apply_policy(
        {"model": "m", "reasoning_effort": "high"},
        "What is the capital of India?",
    )
    assert action == "caller_set"
    assert body["reasoning_effort"] == "high"


def test_apply_does_not_mutate_the_input():
    original = {"model": "m"}
    apply_policy(original, "What is the capital of India?")
    assert original == {"model": "m"}


# ======================================================== semantic =========

def test_disabled_by_default():
    assert SemanticCache(threshold=0.0).enabled is False


def test_normalisation_strips_case_punctuation_and_stopwords():
    assert normalise("What is the Capital of India?!") == \
           normalise("what is the capital of india")


def test_jaccard_basics():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_identical_text_is_a_hit():
    cache = SemanticCache(threshold=0.9)
    cache.store("k", "Explain jitter in one sentence.", ["data: [DONE]"], "stop")
    assert cache.find("Explain jitter in one sentence.") is not None


def test_case_and_punctuation_variants_hit():
    cache = SemanticCache(threshold=0.9)
    cache.store("k", "Explain jitter in one sentence.", ["data: [DONE]"], "stop")
    assert cache.find("explain jitter in one sentence") is not None
    assert cache.find("EXPLAIN JITTER IN ONE SENTENCE!!!") is not None


def test_word_reordering_hits():
    cache = SemanticCache(threshold=0.9)
    cache.store("k", "alpha beta gamma delta", ["data: [DONE]"], "stop")
    assert cache.find("delta gamma beta alpha") is not None


@pytest.mark.parametrize("stored,asked", [
    ("Who wrote the Ramayana?", "Who wrote the Mahabharata?"),
    ("Sort this list ascending.", "Sort this list descending."),
    ("Which port does HTTPS use?", "Which port does HTTP use?"),
    ("What is the capital of India?", "What is the capital of Nepal?"),
    ("Explain the difference between TCP and UDP protocols",
     "Explain the difference between TCP and IP protocols"),
    ("Describe how garbage collection works in the Java runtime",
     "Describe how garbage collection works in the Python runtime"),
])
def test_adversarial_one_word_differences_never_hit(stored, asked):
    """THE critical test. A hit here means the cache answered a question
    nobody asked."""
    cache = SemanticCache(threshold=0.92)
    cache.store("k", stored, ["data: [DONE]"], "stop")
    assert cache.find(asked) is None


def test_short_prompts_never_reach_fuzzy_matching():
    """Documented safety property: below min_tokens the lookup returns
    immediately, so very short prompts are never fuzzy-matched. They rely on
    the exact cache instead. This is why two-token adversarial pairs score
    zero candidates rather than a low Jaccard."""
    cache = SemanticCache(threshold=0.5, min_tokens=3)
    cache.store("k", "Who wrote Ramayana", ["data: [DONE]"], "stop")
    cache.find("Who wrote Mahabharata")
    assert cache.stats.candidates == 0
    assert cache.stats.hits == 0


def test_long_adversarial_pairs_are_scored_and_rejected():
    """Same idea, long enough to actually exercise the Jaccard comparison."""
    cache = SemanticCache(threshold=0.92)
    cache.store("k", "Explain the difference between TCP and UDP protocols",
                ["data: [DONE]"], "stop")
    assert cache.find("Explain the difference between TCP and IP protocols") is None
    assert cache.stats.candidates >= 1, "must reach the scoring stage"
    assert cache.stats.rejected_low_score == 1


def test_adversarial_scores_are_far_below_threshold():
    cache = SemanticCache(threshold=0.92)
    a = normalise("Who wrote the Ramayana?")
    b = normalise("Who wrote the Mahabharata?")
    assert jaccard(a, b) < 0.6, "one-word pairs must score well below 0.92"


def test_paraphrases_are_rejected_and_that_is_documented():
    """Honest limitation: this is lexical, not embedding-based. It must not
    pretend to catch paraphrases."""
    cache = SemanticCache(threshold=0.92)
    cache.store("k", "What is India's capital city?", ["data: [DONE]"], "stop")
    assert cache.find("Which city serves as the seat of India's government?") is None


def test_truncated_and_errored_are_never_stored():
    cache = SemanticCache(threshold=0.9)
    assert cache.store("a", "some text here now", ["x"], "length") is False
    assert cache.store("b", "some text here now", ["x"], "error") is False
    assert cache.store("c", "some text here now", ["x"], "") is False
    assert len(cache) == 0


def test_too_short_text_is_not_stored():
    cache = SemanticCache(threshold=0.9, min_tokens=3)
    assert cache.store("k", "hi there", ["x"], "stop") is False


def test_exclude_key_prevents_self_match():
    cache = SemanticCache(threshold=0.9)
    cache.store("k", "exactly the same words appear here", ["data: [DONE]"], "stop")
    assert cache.find("exactly the same words appear here", exclude_key="k") is None


def test_eviction_at_max_entries():
    cache = SemanticCache(threshold=0.9, max_entries=2)
    for index in range(3):
        cache.store(f"k{index}", f"distinct sentence number {index} here",
                    ["data: [DONE]"], "stop")
    assert len(cache) == 2


def test_inverted_index_is_cleaned_on_eviction():
    cache = SemanticCache(threshold=0.9, max_entries=1)
    cache.store("a", "unique alpha tokens here", ["data: [DONE]"], "stop")
    cache.store("b", "different beta wording now", ["data: [DONE]"], "stop")
    assert "alpha" not in cache._index


def test_stats_track_lookups_and_hits():
    cache = SemanticCache(threshold=0.9)
    cache.store("k", "repeated phrase for stats", ["data: [DONE]"], "stop")
    cache.find("repeated phrase for stats")
    cache.find("something else entirely different")
    assert cache.stats.lookups == 2
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == 50.0


# --- observability counters -----------------------------------------------
# Rule from earlier this session: when a feature is env-gated off by default,
# its counters must be observed on the ENABLED path before it counts as done.

def test_bucket_counts_admitted_and_rejected():
    bucket = TokenBucket(rate_per_minute=60, burst=2)
    # check() does not consume by design, so drain explicitly to reach the
    # rejection path. Two requests fit in a burst of two; the third must not.
    for _ in range(2):
        assert bucket.check("k")[0] is True
        bucket.consume("k")
    assert bucket.check("k")[0] is False
    st = bucket.stats()
    assert st["checked"] == 3
    assert st["admitted"] == 2
    assert st["rejected"] == 1
    assert st["enabled"] is True
    assert st["keys_tracked"] == 1


def test_bucket_stats_when_disabled():
    bucket = TokenBucket(rate_per_minute=0, burst=0)
    bucket.check("k")
    st = bucket.stats()
    assert st["enabled"] is False
    # A disabled bucket must not silently accrue counters.
    assert st["checked"] == 0 and st["admitted"] == 0


def test_policy_stats_tally_each_outcome():
    stats = PolicyStats()
    stats.record(NONE)
    stats.record("kept")
    stats.record("caller_set")
    assert stats.applied == 3
    assert stats.as_dict() == {
        "applied": 3, "downgraded": 1, "kept": 1, "caller_set": 1,
    }


# --- integration: the wiring, not just the class --------------------------
# Regression guard for a real bug: ExactCache.store(key, lines, text, ...) and
# SemanticCache.store(key, text, lines, ...) take their arguments in OPPOSITE
# order. Both upstream call sites passed the completion text into the prompt
# slot, so every entry was indexed by the model's ANSWER and no near-duplicate
# could ever hit. Unit tests on SemanticCache passed happily the whole time —
# only a test that goes through the real request path catches this.

import httpx
import pytest

from kosh.resilience import CircuitBreaker
from kosh.upstream import collect_upstream


def _fake_upstream():
    """A minimal OpenAI-shaped response that completes cleanly."""
    payload = {
        "id": "cmpl-fake",
        "object": "chat.completion",
        "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "A canned answer."},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload)
    )


async def _roundtrip(first: str, second: str):
    sem = SemanticCache(threshold=0.92)
    breaker = CircuitBreaker(threshold=99, reset_after=30.0)
    async with httpx.AsyncClient(transport=_fake_upstream()) as client:
        for prompt in (first, second):
            await collect_upstream(
                client=client,
                url="http://upstream.invalid/v1/chat/completions",
                body={"model": "m",
                      "messages": [{"role": "user", "content": prompt}]},
                headers={"Authorization": "Bearer t"},
                request_id="rid",
                breaker=breaker,
                cache=None,
                cache_key="key-" + prompt[:8],
                semantic=sem,
                prompt_text=prompt,
            )
    return sem


@pytest.mark.asyncio
async def test_semantic_cache_hits_through_the_real_request_path():
    """The bug this guards: near-duplicate must HIT, not merely be stored."""
    sem = await _roundtrip(
        "Explain the difference between TCP and UDP in computer networking",
        "Please explain the difference between TCP and UDP in computer networking",
    )
    assert sem.stats.lookups == 2, "second request must consult the cache"
    assert sem.stats.hits == 1, (
        "near-duplicate missed end-to-end — entries are indexed by the wrong "
        "string (completion text instead of the prompt)"
    )


@pytest.mark.asyncio
async def test_adversarial_pair_still_misses_through_the_real_request_path():
    """The safety property must also hold on the wired path, not just in the
    class's own unit tests."""
    sem = await _roundtrip(
        "Explain the difference between TCP and UDP in computer networking",
        "Explain the difference between TCP and IP in computer networking",
    )
    assert sem.stats.lookups == 2
    assert sem.stats.hits == 0


# --- regressions found by running the thing, not by reading it -------------
# Every test here is parameterised over BOTH request paths (streamed and
# non-streamed). The near-duplicate wiring exists in two places in
# upstream.py; a test that only covers one of them lets a bug live in the
# other. These were verified to FAIL when each bug is reintroduced.

import httpx as _httpx
import os as _os

_PROMPT = "Explain the difference between TCP and UDP in computer networking"
_NEAR = "Please explain the difference between TCP and UDP in computer networking"
_ADVERSARIAL = "Explain the difference between TCP and IP in computer networking"


async def _serve(prompts, stream=False):
    """Drive prompts through the real app; return the x-kosh-cache headers.

    Three things here are deliberate, and all three were bugs at some point:

    1. The body is CONSUMED EXPLICITLY, by iterating it ourselves via
       client.stream(). Kosh only caches a stream once the generator has run
       to completion - that is the rule that stops it caching a fragment.
       So a test that leaves the body unconsumed observes "nothing was ever
       cached" and reports a miss. client.post() happens to read the body in
       some httpx versions and not others; relying on that made these tests
       pass on one machine and fail on another. Iterate it ourselves.

    2. The body is asserted NON-EMPTY. An empty stream is the failure mode
       that looks exactly like "caching is broken" - zero chunks means the
       buffer never fills, so nothing is stored and every request misses.
       Without this check that presents as a bare assertion on 'near'.

    3. A FRESH app is built per call via create_app(). The module-level app
       reads its env when kosh.main is first imported, so if another test
       file imports it first these tests would inherit that environment.
    """
    from kosh.main import create_app

    app = create_app()
    out = []
    async with app.router.lifespan_context(app):
        if not getattr(app.state, "replay", None) or not app.state.replay.enabled:
            raise AssertionError(
                "replay transport is not active - these tests would hit the "
                "real network. KOSH_REPLAY must be 1."
            )
        transport = _httpx.ASGITransport(app=app)
        async with _httpx.AsyncClient(transport=transport,
                                      base_url="http://t") as client:
            for p in prompts:
                body = {"model": "m",
                        "messages": [{"role": "user", "content": p}]}
                if stream:
                    body["stream"] = True
                chunks = []
                async with client.stream(
                    "POST", "/v1/chat/completions", json=body,
                    headers={"Authorization": "Bearer t",
                             "Content-Type": "application/json"},
                ) as r:
                    if r.status_code != 200:
                        await r.aread()
                        raise AssertionError(
                            f"expected 200, got {r.status_code}: {r.text[:300]}"
                        )
                    async for chunk in r.aiter_bytes():
                        chunks.append(chunk)
                total = sum(len(c) for c in chunks)
                if total == 0:
                    raise AssertionError(
                        "streamed body was empty (0 bytes). The upstream "
                        "response produced no chunks, so the buffer never "
                        "filled and nothing can be cached. This is the real "
                        "fault - the cache assertion below is only a symptom."
                    )
                out.append(r.headers.get("x-kosh-cache"))
    return out


def _env(monkeypatch, **over):
    """Set EVERY variable Kosh reads, so the test does not depend on the
    developer's shell, a stray .env file, or which test file imported
    kosh.main first. Anything Kosh reads and we forget to pin can leak in
    from .env and change the result on someone else's machine.
    """
    base = {
        # replay: offline and deterministic
        "KOSH_REPLAY": "1",
        "KOSH_REPLAY_PROFILE": "clean",
        # upstream: unreachable on purpose - replay intercepts every request,
        # so if replay is ever bypassed this fails loudly instead of silently
        # calling the real API.
        "KOSH_UPSTREAM_BASE": "http://127.0.0.1:9/v1",
        "KOSH_MODEL": "m",
        "KOSH_UPSTREAM_KEY": "",
        "BHARATCODE_API_KEY": "",
        # shaping: never rewrite the prompt under test
        "KOSH_SHAPE": "report",
        # caches: pinned, and TTL/entry limits are explicit
        "KOSH_CACHE": "1",
        "KOSH_CACHE_TTL": "3600",
        "KOSH_CACHE_MAX_ENTRIES": "1000",
        "KOSH_CACHE_MAX_TEMPERATURE": "2.0",
        "KOSH_SEMANTIC_THRESHOLD": "0",
        "KOSH_SEMANTIC_MAX_ENTRIES": "500",
        "KOSH_SEMANTIC_MIN_TOKENS": "3",
        # admission + reasoning
        "KOSH_RATE_LIMIT": "0",
        "KOSH_RATE_BURST": "",
        "KOSH_REASONING": "off",
        # resilience: keep failures fast and retry-free, profile is "clean"
        "KOSH_MAX_ATTEMPTS": "1",
        "KOSH_BACKOFF_BASE": "0.001",
        "KOSH_BACKOFF_CAP": "0.01",
        "KOSH_BREAKER_THRESHOLD": "5",
        "KOSH_BREAKER_RESET": "30",
        # ledger off: these tests do not test persistence
        "KOSH_LEDGER": "0",
        "KOSH_LEDGER_PATH": ".kosh/ledger.db",
    }
    base.update(over)
    for k, v in base.items():
        monkeypatch.setenv(k, str(v))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["collect", "stream"])
async def test_kosh_cache_zero_does_not_serve_exact_hits(monkeypatch, stream):
    """Regression: making cache_key available to the semantic store also made
    it available to the EXACT lookup, so KOSH_CACHE=0 stopped disabling
    anything. Semantic is ON here on purpose - that is the configuration that
    produces a cache_key while the exact cache is meant to be off."""
    _env(monkeypatch, KOSH_CACHE="0", KOSH_SEMANTIC_THRESHOLD="0.92")
    got = await _serve([_PROMPT, _PROMPT], stream=stream)
    assert got == ["miss", "near"], (
        "KOSH_CACHE=0 must disable exact caching; with semantic on, an "
        "identical repeat may only be served as a near-duplicate"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["collect", "stream"])
async def test_semantic_cache_works_with_exact_cache_off(monkeypatch, stream):
    """Regression: the near-duplicate store was nested inside the exact
    cache's enabled-check, so KOSH_CACHE=0 silently disabled both."""
    _env(monkeypatch, KOSH_CACHE="0", KOSH_SEMANTIC_THRESHOLD="0.92")
    got = await _serve([_PROMPT, _NEAR, _ADVERSARIAL], stream=stream)
    assert got[0] == "miss"
    assert got[1] == "near", "near-duplicate must hit even with KOSH_CACHE=0"
    assert got[2] == "miss", "adversarial pair must never hit"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["collect", "stream"])
async def test_entry_is_indexed_by_prompt_not_completion(monkeypatch, stream):
    """Regression: ExactCache.store(key, lines, text, ...) and
    SemanticCache.store(key, text, lines, ...) take their arguments in
    OPPOSITE order. Both call sites passed the completion text into the
    prompt slot, indexing every entry by the model's ANSWER. No near-duplicate
    could ever hit, and every unit test on the class still passed."""
    _env(monkeypatch, KOSH_CACHE="0", KOSH_SEMANTIC_THRESHOLD="0.92")
    got = await _serve([_PROMPT, _NEAR], stream=stream)
    assert got[1] == "near", (
        "entry was stored under the completion text instead of the prompt"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["collect", "stream"])
async def test_exact_cache_wins_when_both_are_on(monkeypatch, stream):
    """An identical repeat goes to the exact cache: cheaper and exact."""
    _env(monkeypatch, KOSH_CACHE="1", KOSH_SEMANTIC_THRESHOLD="0.92")
    got = await _serve([_PROMPT, _PROMPT, _NEAR], stream=stream)
    assert got == ["miss", "hit", "near"]


@pytest.mark.asyncio
async def test_stream_store_survives_generator_closed_before_it_resumes():
    """Reproduces the Windows behaviour directly, without needing Windows.

    Closing an async generator throws GeneratorExit at its last yield. That
    is not an httpx.HTTPError, so it propagates and Python skips an `else:`
    clause entirely - the cache store never ran, and NO streamed response was
    ever cached. Whether the ASGI layer closes the generator before it
    resumes past the final yield is platform- and timing-dependent, which is
    why identical code cached everything on one machine and nothing on
    another.

    So: consume up to and including the terminal [DONE] frame, then close the
    generator instead of letting it resume. The store must still happen.
    """
    from kosh.replay import ReplayConfig, ReplayTransport
    from kosh.resilience import CircuitBreaker
    from kosh.upstream import stream_upstream

    sem = SemanticCache(threshold=0.92)
    breaker = CircuitBreaker(threshold=99, reset_after=30.0)
    # ReplayTransport stands in for httpx.AsyncClient itself (it implements
    # build_request and send), so it is passed as `client` directly rather
    # than being wrapped as a transport.
    client = ReplayTransport(ReplayConfig(enabled=True, profile="clean"))
    if True:
        resp = await stream_upstream(
            client=client,
            url="http://replay.invalid/v1/chat/completions",
            body={"model": "m", "stream": True,
                  "messages": [{"role": "user", "content": _PROMPT}]},
            headers={"Authorization": "Bearer t"},
            request_id="rid",
            breaker=breaker,
            cache=None,
            cache_key="k1",
            semantic=sem,
            prompt_text=_PROMPT,
        )
        iterator = resp.body_iterator
        chunks = []
        async for chunk in iterator:
            chunks.append(chunk)
            if b"data: [DONE]" in chunk:
                break              # terminal frame delivered; do NOT resume
        await iterator.aclose()    # GeneratorExit is thrown at the yield

    assert chunks, "expected a streamed body"
    assert len(sem) == 1, (
        "the store did not survive the generator being closed - it is either "
        "still in an `else:` clause (skipped on close) or it only trusts "
        "normal iteration instead of the terminal [DONE] frame"
    )
