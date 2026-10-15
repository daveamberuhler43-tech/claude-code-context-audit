#!/usr/bin/env python3
"""
video_facts.py - every number video 006 says, derived from this machine's Claude Code logs.

    python video_facts.py            # writes video-facts.json (numbers only, no prompt text)
    python video_facts.py --selftest # synthetic log with known answers + mutation checks

Builds on context_audit.py (first-request sizes, list sizes) and adds:
  * how many skills were actually invoked (Skill tool calls + slash commands), de-duplicated
    by tool_use id, because Claude Code writes one log line per content block of a message;
  * skill-list size by source plugin, taken from the logged `names` field;
  * the before/after pair around the day the big skills pack was installed.
Nothing here is a benchmark of Claude Code. It is an audit of one corpus (one user, one machine).
"""
import collections
import glob
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ca", os.path.join(HERE, "context_audit.py"))
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

HOME = os.path.expanduser("~")
TOP = os.path.join(HOME, ".claude", "projects", "*", "*.jsonl")
SUB = os.path.join(HOME, ".claude", "projects", "*", "*", "subagents", "*.jsonl")
OUT = os.path.join(HERE, "video-facts.json")


def usage_from_lines(lines):
    """-> (skill_tool_ids{id: name}, slash{(sessionline): name}, assistant_msg_ids set)"""
    tools, slashes, msgs = {}, [], set()
    for line in lines:
        try:
            o = json.loads(line)
        except ValueError:
            continue
        m = o.get("message")
        if o.get("type") == "assistant" and isinstance(m, dict):
            if m.get("id"):
                msgs.add(m["id"])
            if isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Skill":
                        inp = b.get("input") or {}
                        tools[b.get("id")] = inp.get("skill") or inp.get("name") or "?"
        if o.get("type") == "user" and isinstance(m, dict):
            c = m.get("content")
            txt = c if isinstance(c, str) else " ".join(ca.strings(c)) if c else ""
            for cm in re.findall(r"<command-name>/?([^<]+)</command-name>", txt):
                slashes.append(cm.strip())
    return tools, slashes, msgs


def split_entries(content, names):
    """Split the logged skill list into entries. A line starts a new entry only if it begins
    '- <name>' for a name the log itself lists; anything else is a continuation line."""
    nameset = set(names)
    entries, cur = [], None
    for line in content.split("\n"):
        key = line[2:].split(": ", 1)[0].split(" - ", 1)[0] if line.startswith("- ") else None
        if key in nameset:
            cur = [key, [line]]
            entries.append(cur)
        elif cur is not None:
            cur[1].append(line)
    return [(k, "\n".join(v)) for k, v in entries]


def plugin_of(name):
    return name.split(":", 1)[0] if ":" in name else "(unprefixed)"


def latest_skill_attachment(paths):
    best, bts = None, ""
    for f in paths:
        ts0 = None
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if ts0 is None and o.get("timestamp"):
                    ts0 = o["timestamp"]
                if o.get("type") == "assistant":
                    break
                a = o.get("attachment")
                if o.get("type") == "attachment" and isinstance(a, dict) and a.get("type") == "skill_listing" and ts0 and ts0 >= bts:
                    best, bts = a, ts0
    return best, bts


def facts():
    top = sorted(glob.glob(TOP))
    allp = top + sorted(glob.glob(SUB))
    audit = ca.analyse([s for s in (ca.read_session(open(f, encoding="utf-8", errors="ignore")) for f in top) if s])
    tools, slashes, msgs = {}, [], set()
    for f in allp:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            t, s, m = usage_from_lines(fh)
        tools.update(t)
        slashes += s
        msgs |= m
    att, ts = latest_skill_attachment(top)
    entries = split_entries(att["content"], att["names"])
    by_plugin = collections.defaultdict(lambda: [0, 0])
    for name, text in entries:
        p = plugin_of(name)
        by_plugin[p][0] += 1
        by_plugin[p][1] += ca.ntok(text)
    ranked = sorted(by_plugin.items(), key=lambda kv: -kv[1][1])
    skill_tok_listing = ca.ntok(att["content"])
    used_tool = set(tools.values())
    used_slash = {s for s in slashes if s not in ("model", "compact", "clear", "auto-mode-setup", "help", "context", "config")}
    first = audit["by_session"]
    before = first[0]
    # MACHINE-SPECIFIC: 21 Aug 2026 is the day after I installed the skills pack on my machine. Set it to the
    # first date your own first-request size jumped, or ignore the before_after block.
    AFTER_DATE = "2026-08-21"
    after = [r for r in first if r["date"] == AFTER_DATE]
    return {
        "scope": {"top_level_sessions": audit["sessions"], "dates": audit["dates"],
                  "assistant_messages_all_logs_incl_subagents": len(msgs)},
        "first_request": audit["first_request"],
        "latest_lists": {"skills_listed": att["skillCount"], "skill_list_chars": len(att["content"]),
                         "skill_list_cl100k_tokens": skill_tok_listing,
                         "agent_types": audit["latest_counts"]["agent_entries"],
                         "tool_name_lines": audit["latest_counts"]["tool_name_lines"],
                         "components": audit["latest_components"]},
        "components_cl100k_total": sum(v["cl100k_tokens"] for t, v in audit["latest_components"].items() if t in ca.SENT),
        # plugin names are replaced by their rank so the output holds numbers only
        "skills_by_plugin": [{"plugin": f"plugin-{i + 1}", "entries": n, "cl100k_tokens": t} for i, (p, (n, t)) in enumerate(ranked[:8])],
        "skills_used": {"skill_tool_calls": len(tools), "distinct_skill_tool": len(used_tool),
                        "distinct_slash_skills": len(used_slash), "distinct_either": len(used_tool | used_slash),
                        "share_of_listed": round(len(used_tool | used_slash) / att["skillCount"], 4)},
        "before_after": {"before": before, "after": after[0] if after else None,
                         "after_all": [r["first_ctx"] for r in after]},
        "peak_request": audit["peak_request"],
        "tokenizer_note": "tiktoken cl100k, a ruler not Claude's tokenizer; API totals are measured, component tokens are lower-bound estimates",
    }


# ------------------------------------------------------------------ self test
def selftest():
    def msg(i, tool_id=None):
        content = [{"type": "tool_use", "id": tool_id, "name": "Skill", "input": {"skill": "alpha"}}] if tool_id else [{"type": "text", "text": "x"}]
        return json.dumps({"type": "assistant", "message": {"id": i, "content": content}})
    lines = [msg("m1", "t1"), msg("m1", "t1"), msg("m2", "t2"), msg("m3"), json.dumps(
        {"type": "user", "message": {"content": "<command-name>/beta</command-name>"}})]
    tools, slashes, msgs = usage_from_lines(lines)
    ok1 = len(tools) == 2 and len(msgs) == 3 and slashes == ["beta"]
    print("dedupe by tool id / message id:", "PASS" if ok1 else f"FAIL {tools} {msgs} {slashes}")
    names = ["p:a", "b"]
    content = "- p:a: first\ncontinues here\n- b: second\n- not-listed: third"
    e = split_entries(content, names)
    ok2 = [k for k, _ in e] == ["p:a", "b"] and "continues here" in e[0][1] and "not-listed" in e[1][1]
    print("entry split keeps continuation lines:", "PASS" if ok2 else f"FAIL {e}")
    # mutation: if dedupe were removed the count would double; confirm test would notice
    naive = sum(1 for ln in lines[:3] if '"Skill"' in ln)
    ok3 = naive == 3 and len(tools) == 2
    print("mutation (no dedupe would say 3, we say 2):", "PASS" if ok3 else "FAIL")
    return 0 if ok1 and ok2 and ok3 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    f = facts()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(f, fh, indent=1)
    print(json.dumps({k: f[k] for k in ("scope", "first_request", "skills_used", "skills_by_plugin", "components_cl100k_total")}, indent=1))
    print("latest lists:", {k: v for k, v in f["latest_lists"].items() if k != "components"})
    print("before/after:", f["before_after"]["before"], f["before_after"]["after"], f["before_after"]["after_all"])
    print("wrote", OUT)
