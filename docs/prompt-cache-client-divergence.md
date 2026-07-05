# Client Prefix Divergence

**What it is:** a deep prefix-cache loss caused by the *client* changing the
early tokens of its own prompt mid-session — not by the server evicting or
losing anything. The radix tree matches exact token sequences; one changed
token at offset _N_ makes every cached token past _N_ unusable for that
request, by design.

This page documents the investigated behavior so the next person staring at a
"cache hit rate fell off a cliff" dashboard doesn't re-run the same two-hour
forensics. It is **not** a bug writeup: in every investigated occurrence the
server behaved correctly.

## The 2026-07-05 incident (the canonical example)

An 80k-token OpenCode session hit three "cache loss at almost full depth"
events in 25 minutes (65 s / 116 s TTFTs), immediately after PR #157 merged —
so it read as a regression from the SSD-tier work. It wasn't. The trail:

| Time | Prompt | Matched | Cause (from raw request recordings) |
|------|-------:|--------:|--------------------------------------|
| 13:14 | 58,214 | 8,597 | System prompt **gained** an `Instructions from: …/tesseract-web/AGENTS.md` block between two requests, nine seconds apart, same session |
| 13:31 | 84,422 | 8,735 | The injected block's **content changed** (the session's agent edited AGENTS.md — added a `## Commit policy` section) |
| 13:38 | 86,834 | 58,059 | The block **disappeared**: the agent ran `git reset --hard` in tesseract-web, reverting AGENTS.md; the prompt flipped back to the *old* variant and matched the old branch — which the SSD tier restored in 262 ms instead of re-prefilling 58k tokens |

Mechanism: OpenCode re-reads AGENTS.md/rules files from projects the session
touches and splices their **live file content** into the system prompt on
every request. That session's task was *editing* the very AGENTS.md being
injected, so every save and git operation rewrote token ~8.6k of the prompt.
Any client that injects mutable state into the prompt prefix (rules files,
timestamps, directory listings, environment blocks) produces the same
signature.

## The signature, and how the server now reports it

- `sharedPrefixLength` collapses between consecutive requests of the same
  session (e.g. 83,707 → 8,735) while `partitionDigest` is unchanged.
- Lookup still *hits* — at a shallow `branchPoint` — and prefill creates a new
  branch at the divergence offset. The old branch stays restorable (SSD), which
  is what makes a later flip-back cheap.
- Since issue #158, lookups probe for this directly: when the prompt
  *contradicts* cached content, the `lookup` diagnostics event carries
  `divergenceOffset`, `abandonedCachedTokens`, and a `divergence`
  classification —
  - `clientPrefixChange`: deep loss (mismatch well below the abandoned depth,
    ≥4,096 abandoned tokens). Rendered as an orange `divergence` console line
    and a cache-panel notable event.
  - `tailRewind`: the routine per-turn divergence near the leaf (Think-Strip
    Rewind, retried sampling). Logged, never alarmed on.
- The converse also holds: a shallow match with **no** divergence probe means
  the deep branch is simply gone — that one *is* server-side
  (eviction/GC/budget), and the eviction events say why.

## Triage checklist

1. Console/panel says `client prefix change` → diff the raw request recordings
   (`…/tmp/tesseract-debug/http-completions/`, first line is `// session=…`)
   for the same session across the loss; the changed bytes are the answer.
2. No divergence marker → check `eviction` / `budgetChange` /
   `ssdPartitionInvalidated` events in
   `Application Support/CacheDiagnostics/<day>.jsonl`.
3. Per-request ground truth lives in
   `Application Support/PrefixCacheTraces/trace-<day>.jsonl`.

## What not to "fix"

The re-prefill is unavoidable given the request: the tokens genuinely differ.
Do not add fuzzy matching, prompt normalization, or client-specific heuristics
to the cache path — exact-token matching is the correctness contract.
The server's obligations are (a) branch at the divergence and keep the old
branch restorable (it does), and (b) attribute the loss honestly (issue #158).
Users can avoid the cost by not having a session mutate the rules files it is
itself running under, or by accepting one full re-prefill per such mutation.
