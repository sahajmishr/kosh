"""
Kosh — ledger (Day 7). The receipt.

A gateway that claims to save you tokens has to be able to prove it. This is
the proof: every request, written to SQLite in WAL mode, with what it cost
and what it would have cost.

WHY SQLITE AND NOT POSTGRES
---------------------------
This is append-only writes at a few per second plus dashboard reads. SQLite
in WAL mode handles that trivially, needs no server, no connection pool and
no migrations, and the whole database is one file you can email. Postgres
would be infrastructure theatre at this scale.

WHY NOT AN ORM
--------------
Two tables. SQL is clearer than ORM ceremony here, and it demonstrates that
the SQL is understood rather than hidden.

WAL MATTERS
-----------
Write-Ahead Logging lets reads proceed while a write is in flight. Without
it, every dashboard view would block on a request completing — and a
monitoring dashboard that stalls the thing it monitors is worse than none.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(".kosh/ledger.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL    NOT NULL,
    request_id        TEXT    NOT NULL,
    model             TEXT,
    cache             TEXT,
    attempts          INTEGER DEFAULT 1,
    status            INTEGER DEFAULT 200,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    saved_tokens      INTEGER DEFAULT 0,
    latency_ms        REAL    DEFAULT 0,
    prefix_verdict    TEXT,
    prefix_cacheable  REAL,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts    ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_cache ON requests(cache);
"""


@dataclass
class RequestRecord:
    request_id: str
    model: str = ""
    cache: str = "miss"
    attempts: int = 1
    status: int = 200
    prompt_tokens: int = 0
    completion_tokens: int = 0
    saved_tokens: int = 0
    latency_ms: float = 0.0
    prefix_verdict: str | None = None
    prefix_cacheable: float | None = None
    error: str | None = None
    ts: float = 0.0

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = time.time()


class Ledger:
    """Append-only request log with aggregate queries for the dashboard."""

    def __init__(self, path: Path | str = DEFAULT_PATH, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.conn: sqlite3.Connection | None = None
        if not enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because FastAPI handlers run on a thread
        # pool; writes are serialised by SQLite's own locking.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def record(self, row: RequestRecord) -> None:
        if not self.conn:
            return
        self.conn.execute(
            """INSERT INTO requests (ts, request_id, model, cache, attempts,
                   status, prompt_tokens, completion_tokens, saved_tokens,
                   latency_ms, prefix_verdict, prefix_cacheable, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row.ts, row.request_id, row.model, row.cache, row.attempts,
             row.status, row.prompt_tokens, row.completion_tokens,
             row.saved_tokens, row.latency_ms, row.prefix_verdict,
             row.prefix_cacheable, row.error),
        )
        self.conn.commit()

    # ------------------------------------------------------------ queries --

    def _scalar(self, sql: str, args: tuple = ()) -> float:
        if not self.conn:
            return 0.0
        cur = self.conn.execute(sql, args)
        value = cur.fetchone()[0]
        return float(value or 0.0)

    def totals(self) -> dict:
        if not self.conn:
            return {}
        requests = int(self._scalar("SELECT COUNT(*) FROM requests"))
        if requests == 0:
            return {"requests": 0}

        prompt = int(self._scalar("SELECT COALESCE(SUM(prompt_tokens),0) FROM requests"))
        completion = int(self._scalar(
            "SELECT COALESCE(SUM(completion_tokens),0) FROM requests"))
        saved = int(self._scalar("SELECT COALESCE(SUM(saved_tokens),0) FROM requests"))
        hits = int(self._scalar(
            "SELECT COUNT(*) FROM requests WHERE cache='hit'"))
        errors = int(self._scalar(
            "SELECT COUNT(*) FROM requests WHERE error IS NOT NULL"))
        hostile = int(self._scalar(
            "SELECT COUNT(*) FROM requests WHERE prefix_verdict IN ('poor','hostile')"))
        retried = int(self._scalar(
            "SELECT COUNT(*) FROM requests WHERE attempts > 1"))
        avoided = int(self._scalar(
            "SELECT COUNT(*) FROM requests WHERE cache='hit'"))

        return {
            "requests": requests,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "tokens_saved": saved,
            "tokens_not_sent_pct": round(100.0 * saved / max(1, saved + prompt + completion), 1),
            "cache_hits": hits,
            "cache_hit_pct": round(100.0 * hits / requests, 1),
            "upstream_calls_avoided": avoided,
            "upstream_calls_avoided_pct": round(100.0 * avoided / requests, 1),
            "errors": errors,
            "error_pct": round(100.0 * errors / requests, 1),
            "cache_hostile_requests": hostile,
            "cache_hostile_pct": round(100.0 * hostile / requests, 1),
            "retries": retried,
        }

    def latency_percentiles(self) -> dict:
        """p50/p95 over non-cached requests only.

        Cached requests return in ~0ms; including them would flatter the
        numbers and hide the real upstream latency from the operator.
        """
        if not self.conn:
            return {}
        rows = [
            r[0] for r in self.conn.execute(
                "SELECT latency_ms FROM requests WHERE cache != 'hit' "
                "AND latency_ms > 0 ORDER BY latency_ms"
            ).fetchall()
        ]
        if not rows:
            return {}

        def pct(q: float) -> float:
            index = min(len(rows) - 1, int(round((q / 100.0) * (len(rows) - 1))))
            return round(rows[index], 1)

        return {
            "n": len(rows),
            "p50_ms": pct(50),
            "p95_ms": pct(95),
            "p99_ms": pct(99),
            "max_ms": round(rows[-1], 1),
        }

    def recent(self, limit: int = 25) -> list[dict]:
        if not self.conn:
            return []
        cur = self.conn.execute(
            """SELECT ts, request_id, model, cache, attempts, status,
                      prompt_tokens, completion_tokens, saved_tokens,
                      latency_ms, prefix_verdict, error
               FROM requests ORDER BY id DESC LIMIT ?""", (limit,))
        keys = ["ts", "request_id", "model", "cache", "attempts", "status",
                "prompt_tokens", "completion_tokens", "saved_tokens",
                "latency_ms", "prefix_verdict", "error"]
        return [dict(zip(keys, row)) for row in cur.fetchall()]

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None


def ledger_from_env() -> Ledger:
    import os

    enabled = os.getenv("KOSH_LEDGER", "1").lower() in {"1", "true", "yes"}
    return Ledger(path=os.getenv("KOSH_LEDGER_PATH", str(DEFAULT_PATH)),
                  enabled=enabled)
