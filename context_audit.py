#!/usr/bin/env python3
"""
context_audit.py - what is in a Claude Code session BEFORE the first message, measured from the
session logs Claude Code already writes to disk.

    python context_audit.py            # reads ~/.claude/projects/*/*.jsonl, writes context-audit.json
    python context_audit.py --selftest # synthetic logs with a known answer, plus mutation tests

Only numbers leave this script. It never writes prompt text, file contents, paths or e-mail
addresses to its output. Per session it keeps: the start date, the API-reported size of the first
request, and the character/token size of each attachment type recorded before that request.

WHAT IS MEASURED vs ESTIMATED
  measured  : the first request's size, taken from the `usage` block the API returned
              (input_tokens + cache_creation_input_tokens + cache_read_input_tokens).
  measured  : the characters of every attachment recorded before that request (skill list,
              agent list, deferred-tool list, MCP instructions, hook context, ...).
  estimated : tokens per component. tiktoken cl100k is NOT Claude's tokenizer. It is used as a
              ruler, then CALIBRATED against the API total with a regression across sessions
              (first_ctx = H + k * T). H is the fixed part (system prompt + built-in tools),
              k says how far cl100k under-counts. Both are reported with a bootstrap interval.
Components present in the log but possibly not sent to the model (hook_success is the record of
a hook running, not its context) are reported separately and excluded from the sum.
"""
import collections
import glob
import hashlib
import json
import os
import random
import statistics
import sys

import numpy as np
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "context-audit.json")
LOGS = os.path.join(os.path.expanduser("~"), ".claude", "projects", "*", "*.jsonl")

# attachment type -> the field(s) that carry the text the model is shown. Other fields
# (`names`, `addedTypes`, `addedNames`, ids, counters) are bookkeeping that repeats the same
# words; summing every string in an attachment counted skills and tools twice (caught 14 Oct).
SENT_FIELDS = {
    "skill_listing": ("content",),
    "agent_listing_delta": ("addedLines",),
    "deferred_tools_delta": ("addedLines",),
    "mcp_instructions_delta": ("addedBlocks",),
    "hook_additional_context": ("content",),
    "instructions": ("files",),
    "output_style_instructions": ("style",),
    "session_context": ("context",),
    "environment": ("snapshot",),
    "model": ("text",),
    "date": ("date",),
}
SENT = tuple(SENT_FIELDS)
# recorded, but not necessarily sent: reported, never summed
RECORD_FIELDS = {"hook_success": ("content",), "hook_system_message": ("content",)}
RECORD_ONLY = tuple(RECORD_FIELDS)

_cache = {}


def ntok(text):
    h = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
    if h not in _cache:
        _cache[h] = len(ENC.encode(text, disallowed_special=()))
    return _cache[h]


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


def ctx_of(usage):
    return ((usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0))


def read_session(lines):
    """-> dict(date, first_ctx, comps={type: text}, requests=[ctx...]) or None"""
    comps = collections.defaultdict(list)
    first = None
    date = None
    requests = []
    for line in lines:
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if date is None and o.get("timestamp"):
            date = o["timestamp"][:10]
        m = o.get("message")
        if o.get("type") == "assistant" and isinstance(m, dict) and m.get("usage"):
            c = ctx_of(m["usage"])
            if c > 0:
                requests.append(c)
                if first is None:
                    first = c
            continue
        a = o.get("attachment")
        if first is None and o.get("type") == "attachment" and isinstance(a, dict):
            fields = SENT_FIELDS.get(a.get("type")) or RECORD_FIELDS.get(a.get("type"))
            if fields:
                body = {k: v for k, v in a.items() if k in fields}
                comps[a.get("type")].append("\n".join(strings(body)))
    if first is None:
        return None
    return {"date": date, "first_ctx": first, "comps": {k: "".join(v) for k, v in comps.items()},
            "requests": requests}


def fit(rows):
    """first_ctx = H + k * T  -> (H, k, r2)"""
    x = np.array([r["T"] for r in rows], float)
    y = np.array([r["first_ctx"] for r in rows], float)
    A = np.vstack([np.ones_like(x), x]).T
    (h, k), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = h + k * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    return float(h), float(k), 1 - ss_res / ss_tot


def bootstrap(rows, n=2000, seed=6):
    rnd = random.Random(seed)
    hs, ks = [], []
    for _ in range(n):
        s = [rows[rnd.randrange(len(rows))] for _ in rows]
        if len({r["T"] for r in s}) < 3:
            continue
        h, k, _ = fit(s)
        hs.append(h)
        ks.append(k)
    if not hs:                       # every session identical: no spread to bootstrap
        return {"H_lo": None, "H_hi": None, "k_lo": None, "k_hi": None}
    q = lambda v, p: float(np.percentile(v, p))
    return {"H_lo": q(hs, 2.5), "H_hi": q(hs, 97.5), "k_lo": q(ks, 2.5), "k_hi": q(ks, 97.5)}


def analyse(sessions):
    rows = []
    for s in sessions:
        tok = {t: ntok(txt) for t, txt in s["comps"].items()}
        chars = {t: len(txt) for t, txt in s["comps"].items()}
        rows.append({"date": s["date"], "first_ctx": s["first_ctx"], "tok": tok, "chars": chars,
                     "T": sum(tok.get(t, 0) for t in SENT), "requests": s["requests"],
                     "comps": s["comps"]})
    rows.sort(key=lambda r: r["date"])
    h, k, r2 = fit(rows)
    ci = bootstrap(rows)
    latest = rows[-1]
    firsts = [r["first_ctx"] for r in rows]

    def count_lines(text):
        return len([l for l in text.split("\n") if l.startswith("- ")])

    comp_latest = {}
    for t in SENT + RECORD_ONLY:
        if t in latest["tok"]:
            comp_latest[t] = {"chars": latest["chars"][t], "cl100k_tokens": latest["tok"][t],
                              "est_claude_tokens": round(latest["tok"][t] * k)}
    big = sorted(((v["est_claude_tokens"], t) for t, v in comp_latest.items() if t in SENT), reverse=True)
    peaks = [max(r["requests"]) for r in rows if r["requests"]]
    return {
        "sessions": len(rows),
        "dates": [rows[0]["date"], rows[-1]["date"]],
        "first_request": {"median": int(statistics.median(firsts)), "min": min(firsts), "max": max(firsts),
                          "latest": latest["first_ctx"], "latest_date": latest["date"],
                          "last_10_median": int(statistics.median(firsts[-10:]))},
        "fit": {"H_fixed_tokens": round(h), "k_scale": round(k, 3), "r2": round(r2, 3), **{a: (None if b is None else round(b, 3)) for a, b in ci.items()}},
        "latest_components": comp_latest,
        "latest_ranked": [{"type": t, "est_claude_tokens": n, "share_of_first_request": round(n / latest["first_ctx"], 3)}
                          for n, t in big],
        "latest_counts": {"skill_entries": count_lines(latest["comps"].get("skill_listing", "")),
                          "agent_entries": count_lines(latest["comps"].get("agent_listing_delta", "")),
                          "tool_name_lines": len([l for l in latest["comps"].get("deferred_tools_delta", "").splitlines() if l.strip()])},
        "record_only_note": "hook_success / prompt_snapshot are logged but not summed",
        "peak_request": {"max": max(peaks), "median": int(statistics.median(peaks))},
        "by_session": [{"date": r["date"], "first_ctx": r["first_ctx"], "T_cl100k": r["T"],
                        "skill_chars": r["chars"].get("skill_listing", 0)} for r in rows],
    }


def load_real():
    out = []
    for f in sorted(glob.glob(LOGS)):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            s = read_session(fh)
        if s:
            out.append(s)
    return out


# ------------------------------------------------------------------ self test
def _fake(date, skill_chars, first_ctx):
    att = {"type": "attachment", "attachment": {"type": "skill_listing", "content": "- skill: " + "x" * skill_chars}}
    line1 = json.dumps({"timestamp": date + "T00:00:00Z", **att})
    line2 = json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 10, "cache_creation_input_tokens": first_ctx - 10,
                                                                      "cache_read_input_tokens": 0}}})
    return [line1, line2]


def selftest():
    rnd = random.Random(1)
    sessions = []
    for i in range(30):
        chars = rnd.randrange(2_000, 60_000)
        t = ntok("- skill: " + "x" * chars)
        sessions.append(read_session(_fake(f"2026-09-{1 + i % 28:02d}", chars, int(12_000 + 1.3 * t))))
    res = analyse(sessions)
    ok = abs(res["fit"]["H_fixed_tokens"] - 12_000) < 600 and abs(res["fit"]["k_scale"] - 1.3) < 0.03
    print("known H=12000 k=1.3 ->", res["fit"]["H_fixed_tokens"], res["fit"]["k_scale"], "PASS" if ok else "FAIL")
    # mutation 1: corrupt the API totals; the fit must NOT still say 1.3
    bad = [dict(s, first_ctx=rnd.randrange(20_000, 150_000)) for s in sessions]
    res2 = analyse(bad)
    ok2 = abs(res2["fit"]["k_scale"] - 1.3) > 0.1 or res2["fit"]["r2"] < 0.5
    print("mutation: random totals -> r2", res2["fit"]["r2"], "PASS (detected)" if ok2 else "FAIL (not detected)")
    # mutation 2: text never leaks into the output
    secret = "SECRET-PROMPT-TEXT"
    leak = read_session([json.dumps({"timestamp": "2026-09-01T00:00:00Z", "type": "attachment",
                                      "attachment": {"type": "skill_listing", "content": "- a " + secret}}),
                         json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 1000}}})])
    res3 = analyse([leak, dict(leak, date="2026-09-02"), dict(leak, date="2026-09-03")])
    ok3 = secret not in json.dumps(res3)
    print("no text in output ->", "PASS" if ok3 else "FAIL")
    return 0 if ok and ok2 and ok3 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    res = analyse(load_real())
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("sessions", "dates", "first_request", "fit", "latest_counts", "latest_ranked", "peak_request")}, indent=1))
    print("wrote", OUT)
