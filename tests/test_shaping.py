"""
Day 5 prefix-hygiene tests — no network, no API key.

The claim being tested is narrow and specific: a volatile marker early in a
prompt destroys vLLM's prefix cache for everything after it, and Kosh can
detect that, quantify it, and safely move the marker without losing it.
"""
from __future__ import annotations

import pytest

from kosh.shaping import (
    MODES,
    PrefixReport,
    analyse,
    canonicalize_messages,
    find_volatile,
    flatten,
    normalise_whitespace,
    reorder_messages,
    shape_request,
)

CLEAN_SYSTEM = (
    "You are a careful teaching assistant for an undergraduate algorithms "
    "course. Answer concisely, show your reasoning in at most three steps, "
    "and never give the final answer without explaining how you reached it."
)


def body(system: str, question: str = "What is a hash table?") -> dict:
    return {
        "model": "qwen-3.8-27b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    }


# ------------------------------------------------------- canonicalisation ---

def test_whitespace_normalisation_collapses_trailing_space():
    assert normalise_whitespace("hello   \n\nworld  ") == "hello\n\nworld"


def test_whitespace_normalisation_handles_windows_newlines():
    assert normalise_whitespace("a\r\nb\r\nc") == "a\nb\nc"


def test_canonicalisation_makes_equivalent_prompts_identical():
    """Two prompts differing only in whitespace must become byte-identical,
    or they hash differently and miss every cache downstream."""
    a = canonicalize_messages([{"role": "system", "content": "hello world   "}])
    b = canonicalize_messages([{"role": "system", "content": "hello world"}])
    assert a == b


def test_canonicalisation_preserves_message_order():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert [m["role"] for m in canonicalize_messages(messages)] == ["system", "user"]


def test_canonicalisation_handles_multimodal_content_blocks():
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": "describe   "}, {"type": "image_url", "image_url": {}}],
    }]
    out = canonicalize_messages(messages)
    assert out[0]["content"][0]["text"] == "describe"
    assert out[0]["content"][1]["type"] == "image_url"


# ------------------------------------------------------------- detection ---

@pytest.mark.parametrize("snippet,kind_expected", [
    ("Current date: 2026-10-29T14:22:07Z", True),
    ("session id: 7f3a2b91c4", True),
    ("Request-ID: abc-123-xyz", True),
    ("id 3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c", True),
    ("timestamp 1730000000", True),
    ("The capital of India is New Delhi.", False),
])
def test_volatile_detection(snippet, kind_expected):
    assert bool(find_volatile(snippet)) is kind_expected


def test_detection_does_not_flag_ordinary_prose():
    prose = (
        "We measured 1200 requests across 30 minutes and found that the "
        "median latency improved from 4 seconds to less than 1 second."
    )
    assert find_volatile(prose) == []


def test_volatiles_are_sorted_and_non_overlapping():
    text = "2026-10-29T10:00:00Z then 2026-10-29T11:00:00Z"
    spans = find_volatile(text)
    assert spans == sorted(spans)
    for earlier, later in zip(spans, spans[1:]):
        assert earlier[1] <= later[0]


# --------------------------------------------------------------- scoring ---

def test_clean_prompt_scores_100_percent_cacheable():
    report = analyse(body(CLEAN_SYSTEM)["messages"])
    assert report.cacheable_pct == 100.0
    assert report.verdict == "excellent"
    assert report.volatile_count == 0


def test_timestamp_at_the_front_is_hostile():
    poisoned = f"Current date: 2026-10-29T14:22:07Z\n{CLEAN_SYSTEM}"
    report = analyse(body(poisoned)["messages"])
    assert report.first_volatile_at is not None
    assert report.first_volatile_at < 30, "the marker is right at the front"
    assert report.verdict == "hostile"
    assert report.cacheable_pct < 10.0


def test_the_same_timestamp_at_the_end_is_fine():
    """Position is everything: vLLM matches from the start, so late
    volatility costs almost nothing — the blocks before it still hit."""
    safe = f"{CLEAN_SYSTEM}\n\nCurrent date: 2026-10-29T14:22:07Z"
    report = analyse(body(safe)["messages"])
    assert report.cacheable_pct > 80.0
    assert report.verdict in ("good", "poor")


def test_cacheable_pct_is_a_percentage_of_prompt_length():
    report = analyse(body(CLEAN_SYSTEM)["messages"])
    assert 0.0 <= report.cacheable_pct <= 100.0


def test_empty_prompt_reports_cleanly():
    assert analyse([]).verdict == "excellent"


def test_report_exposes_headers():
    headers = analyse(body(CLEAN_SYSTEM)["messages"]).as_headers()
    assert "x-kosh-prefix-cacheable" in headers
    assert "x-kosh-prefix-verdict" in headers
    assert "x-kosh-prefix-tokens" in headers


# ------------------------------------------------------------- reordering ---

def test_reorder_moves_the_marker_to_the_end():
    poisoned = f"Current date: 2026-10-29T14:22:07Z\n{CLEAN_SYSTEM}"
    out = reorder_messages(body(poisoned)["messages"])
    text = flatten(out)
    assert text.startswith("You are a careful teaching assistant")
    assert "2026-10-29T14:22:07Z" in text


def test_reorder_loses_no_information():
    poisoned = f"Current date: 2026-10-29T14:22:07Z\n{CLEAN_SYSTEM}"
    before = flatten(body(poisoned)["messages"])
    after = flatten(reorder_messages(body(poisoned)["messages"]))
    assert "2026-10-29T14:22:07Z" in after
    assert CLEAN_SYSTEM in after
    assert len(after) >= len(before) - 10


def test_reorder_improves_the_score():
    poisoned = f"Current date: 2026-10-29T14:22:07Z\n{CLEAN_SYSTEM}"
    before = analyse(body(poisoned)["messages"])
    after = analyse(reorder_messages(body(poisoned)["messages"]))
    assert after.cacheable_pct > before.cacheable_pct
    assert after.verdict != "hostile"


def test_reorder_leaves_clean_prompts_untouched():
    messages = body(CLEAN_SYSTEM)["messages"]
    assert flatten(reorder_messages(messages)) == flatten(messages)


def test_reorder_is_a_noop_when_volatility_is_already_late():
    """Late volatility is cheap — moving it would only risk changing the
    prompt for no cache benefit, so we leave it alone."""
    safe = f"{CLEAN_SYSTEM}\n\nCurrent date: 2026-10-29T14:22:07Z"
    messages = body(safe)["messages"]
    assert flatten(reorder_messages(messages)) == flatten(messages)


def test_reorder_does_not_touch_prose_that_merely_looks_odd():
    prose = "In 1920 the population was 1060 and the area 1483 square miles."
    messages = [{"role": "user", "content": prose}]
    assert flatten(reorder_messages(messages)) == prose


def test_reorder_survives_a_prompt_that_is_only_a_marker():
    """If there is no stable text left, do nothing rather than emit an
    empty prompt."""
    messages = [{"role": "system", "content": "2026-10-29T14:22:07Z"}]
    assert flatten(reorder_messages(messages)) == "2026-10-29T14:22:07Z"


# ------------------------------------------------------------ shape modes ---

@pytest.mark.parametrize("mode", MODES)
def test_modes_are_all_accepted(mode):
    shaped, report = shape_request(body(CLEAN_SYSTEM), mode)
    assert isinstance(report, PrefixReport)
    assert "messages" in shaped


def test_off_mode_changes_nothing():
    original = body("2026-10-29T10:00:00Z\n" + CLEAN_SYSTEM)
    shaped, _ = shape_request(original, "off")
    assert shaped == original


def test_off_mode_does_not_mutate_the_caller():
    original = body(CLEAN_SYSTEM)
    snapshot = {"messages": [dict(m) for m in original["messages"]]}
    shape_request(original, "reorder")
    assert original["messages"] == snapshot["messages"]


def test_canonical_mode_merges_whitespace_variants():
    a, _ = shape_request(body(CLEAN_SYSTEM + "   "), "canonical")
    b, _ = shape_request(body(CLEAN_SYSTEM), "canonical")
    assert a["messages"] == b["messages"]


def test_reorder_mode_flags_that_it_rewrote():
    poisoned = body("2026-10-29T10:00:00Z\n" + CLEAN_SYSTEM)
    _, report = shape_request(poisoned, "reorder")
    assert report.rewritten is True


def test_report_mode_never_rewrites():
    poisoned = body("2026-10-29T10:00:00Z\n" + CLEAN_SYSTEM)
    shaped, report = shape_request(poisoned, "report")
    assert report.rewritten is False
    assert shaped["messages"] == poisoned["messages"]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        shape_request(body(CLEAN_SYSTEM), "aggressive")


def test_shape_preserves_non_message_fields():
    original = {**body(CLEAN_SYSTEM), "max_tokens": 64, "temperature": 0.2}
    shaped, _ = shape_request(original, "canonical")
    assert shaped["max_tokens"] == 64
    assert shaped["temperature"] == 0.2
    assert shaped["model"] == "qwen-3.8-27b"
