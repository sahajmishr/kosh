"""
Kosh — dashboard (Day 7). One HTML file, no build step, no CDN.

Everything is inlined so the page renders offline, inside a sandboxed
iframe, or behind a firewall. A dashboard is evidence for the product, not
the product — it should never have a toolchain.
"""
from __future__ import annotations

import html
import time

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0f1115;color:#e6e6e6;
 font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:34px 26px 70px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:#8b93a1;font-size:14px;margin-bottom:26px}
.tag{display:inline-block;background:#1d2430;border:1px solid #2b3542;
 color:#7dd3fc;font:600 11px/1 ui-monospace,monospace;letter-spacing:.07em;
 padding:6px 10px;border-radius:4px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:14px}
.card{background:#161b23;border:1px solid #232b36;border-radius:9px;padding:16px 17px}
.card .v{font:700 26px/1.1 ui-monospace,monospace;color:#7dd3fc}
.card .l{font-size:12px;color:#8b93a1;margin-top:6px;line-height:1.4}
h2{font-size:17px;margin:34px 0 12px;padding-top:20px;border-top:1px solid #232b36}
table{width:100%;border-collapse:collapse;font:
 12.5px/1.5 ui-monospace,"SF Mono",Menlo,monospace}
th,td{border:1px solid #232b36;padding:7px 9px;text-align:left}
th{background:#1a2029;color:#8b93a1;font-size:11px;text-transform:uppercase;
 letter-spacing:.05em}
tr:nth-child(even) td{background:#141920}
.hit{color:#4ade80}.miss{color:#fbbf24}.err{color:#f87171}
.note{background:#161b23;border-left:3px solid #7dd3fc;padding:12px 16px;
 border-radius:0 6px 6px 0;margin:18px 0;font-size:13.5px;color:#b9c1cd}
.empty{color:#8b93a1;font-size:14px;padding:26px;text-align:center;
 border:1px dashed #2b3542;border-radius:8px}
code{background:#1d2430;padding:2px 6px;border-radius:3px;color:#7dd3fc}
footer{margin-top:46px;padding-top:18px;border-top:1px solid #232b36;
 color:#6b7280;font-size:12px}
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _stat(value, label) -> str:
    return f'<div class="card"><div class="v">{_esc(value)}</div><div class="l">{label}</div></div>'


def render(totals: dict, latency: dict, recent: list[dict],
           cache_stats: dict, uptime_s: float) -> str:
    if not totals or totals.get("requests", 0) == 0:
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Kosh</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Kosh</h1><div class="sub">No requests recorded yet.</div>
<div class="empty">Send a request through Kosh and reload this page.<br><br>
<code>curl -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
-d '{{"model":"qwen-3.8-27b","messages":[{{"role":"user","content":"hi"}}]}}'</code>
</div></div></body></html>"""

    def row(r: dict) -> str:
        cache = r.get("cache") or "miss"
        cls = "hit" if cache == "hit" else "miss"
        error = r.get("error")
        if error:
            cls = "err"
        return (
            "<tr>"
            f"<td>{time.strftime('%H:%M:%S', time.localtime(r['ts']))}</td>"
            f"<td>{_esc((r.get('request_id') or '')[:14])}</td>"
            f"<td class='{cls}'>{_esc(cache)}</td>"
            f"<td>{r.get('attempts', 1)}</td>"
            f"<td>{r.get('status', '')}</td>"
            f"<td>{r.get('prompt_tokens', 0)}</td>"
            f"<td>{r.get('completion_tokens', 0)}</td>"
            f"<td>{r.get('saved_tokens', 0)}</td>"
            f"<td>{round(r.get('latency_ms') or 0)}</td>"
            f"<td>{_esc(r.get('prefix_verdict') or '-')}</td>"
            "</tr>"
        )

    lat_line = ""
    if latency:
        lat_line = (f" · p50 {latency['p50_ms']:.0f}ms"
                    f" · p95 {latency['p95_ms']:.0f}ms"
                    f" · p99 {latency['p99_ms']:.0f}ms")

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kosh — dashboard</title><style>{CSS}</style>
<meta http-equiv="refresh" content="10"></head>
<body><div class="wrap">
<h1>Kosh</h1>
<div class="sub">Drop-in efficiency gateway for the BharatCode model API
 · auto-refreshes every 10s</div>
<div class="tag">THE CHEAPEST TOKEN IS THE ONE YOU NEVER SEND</div>

<div class="grid">
{_stat(totals['requests'], 'requests through Kosh')}
{_stat(f"{totals['tokens_saved']:,}", 'tokens never sent')}
{_stat(f"{totals['tokens_not_sent_pct']}%", 'of all tokens avoided')}
{_stat(f"{totals['cache_hit_pct']}%", 'cache hit rate')}
{_stat(totals['upstream_calls_avoided'], 'upstream calls avoided')}
{_stat(f"{totals['error_pct']}%", 'error rate')}
{_stat(f"{totals['cache_hostile_pct']}%", 'requests with a cache-hostile prefix')}
{_stat(totals['retries'], 'requests that needed a retry')}
</div>

<div class="note">
<b>{totals['cache_hostile_requests']}</b> of {totals['requests']} requests
arrived with a volatile marker near the front of the prompt — a timestamp,
session id or request id. On vLLM those requests get <b>zero</b> prefix reuse,
because prefix matching runs from the very first token. Kosh reports this per
request in <code>x-kosh-prefix-verdict</code>.
</div>

<h2>Latency (uncached requests only{lat_line})</h2>
<div class="note">
Cached responses return in well under a millisecond. Including them would
make these numbers look wonderful and tell the operator nothing, so they are
excluded. This is what the upstream actually costs.
</div>

<h2>Live cache counters</h2>
<div class="grid">
{_stat(cache_stats.get('hits', 0), 'cache hits (process lifetime)')}
{_stat(cache_stats.get('misses', 0), 'cache misses')}
{_stat(cache_stats.get('rejects', 0), 'rejected: truncated or errored')}
{_stat(cache_stats.get('stores', 0), 'entries stored')}
</div>
<div class="note">
<b>rejected</b> is the honesty counter. A response that came back truncated —
HTTP 200, <code>finish_reason: "length"</code> — is never cached, because
serving an incomplete answer to a later caller is worse than a miss.
</div>

<h2>Most recent requests</h2>
<table>
<tr><th>time</th><th>request id</th><th>cache</th><th>tries</th><th>status</th>
<th>prompt</th><th>completion</th><th>saved</th><th>ms</th><th>prefix</th></tr>
{''.join(row(r) for r in recent)}
</table>

<footer>
Kosh · every number on this page comes from the local SQLite ledger
(<code>.kosh/ledger.db</code>) · {uptime_s:.0f}s of recorded history
</footer>
</div></body></html>"""
