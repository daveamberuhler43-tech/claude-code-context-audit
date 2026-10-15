# claude-code-context-audit

How big is Claude Code's context window **before you type anything**, and what is in it? These scripts read the
session logs Claude Code already writes to disk (`~/.claude/projects/*/*.jsonl`) and print **numbers only**.
No prompt text, file contents, paths or e-mail addresses are written to the output (the self-test checks this).

Companion to the Blast Radius video *Claude Code Loads 2,260 Skills Before I Type. I've Used 21.*

## Run it

```bash
pip install tiktoken numpy
python context_audit.py            # writes context-audit.json next to the script and prints a summary
python context_audit.py --selftest # synthetic logs with a known answer, plus mutation tests
python video_facts.py              # how many skills were actually used, the skill list by plugin (ranked, not named)
python video_facts.py --selftest
```

Post the `first_request` numbers (median, latest) in the video's comments. Or just type `/context` in a fresh session.

## What is measured and what is estimated

| Item | Status |
|---|---|
| Size of each session's first request | **measured**: `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` from the API's own `usage` block |
| Characters of every list loaded before it (skills, agent types, tool names, ...) | **measured** |
| Tokens per list | **estimate, a floor**: tiktoken `cl100k` is not Claude's tokenizer |
| The share of the first request that is not in the log | **not visible**: Claude Code's own instructions and built-in tools |
| Skills "used" | tool calls to the Skill tool plus slash commands, de-duplicated by tool-use id |

`context_audit.py` also prints a calibration regression (`fit`). It is weak (r² about 0.55 on my data) and **the video
does not rely on it**. Ignore it unless you have many sessions.

## Limits

- One corpus, one user, one machine. It audits a setup; it is not a benchmark of Claude Code.
- Logs only go back as far as your install keeps them (mine start 18 Aug 2026).
- `video_facts.py` has one machine-specific constant, `AFTER_DATE` (the day after I installed a skills pack).
- Log field names can change between Claude Code versions. If a script fails on `skillCount` or `names`, that is why.

MIT licence.
