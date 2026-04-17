# Prefix Cache Log Analysis — 2026-04-17

## Scope

- Source logs: `14` captures matching `/tmp/tesseract-bench-*.log`
- Request starts observed: `130`
- Requests that completed inside the captured log windows: `125`
- Saved request bodies used for correlation: `.../tmp/tesseract-debug/http-completions/*-request.json`

Important note on methodology:

- The logs can contain queued requests, so I did **not** correlate `HTTP request logged` entries to completion starts by FIFO order.
- Instead, I rematched each start to its saved request JSON by `(message count, tool definition count, last role)`.
- For “latest cached identity” analysis, I only counted continuations where the later request **explicitly included the previous model output** in its message list. This avoids false positives on branched conversations.

## Executive Summary

- Overall prefix-cache performance in these logs is strong.
- On completed requests, `122 / 125` reused cached tokens: `97.6%`.
- Token-weighted reuse was `5,734,928 / 6,267,523 = 91.5%`.
- The suspected pattern:
  `tool call -> assistant responds after tool use -> next user turn`
  did **not** fail in the observed data.
- I found `8` clear instances of that pattern, and `8 / 8` of the following user turns hit the latest canonical leaf.
- I found exactly **one** proven “latest cached identity” miss, and it was **not** on the next-user-after-tool-response path.
- The one real miss happened on an **immediate tool-result continuation** after a tool-calling turn, and it looks consistent with a narrow multi-tool continuation gap.

## Aggregate Metrics

### Completion-Level Metrics

| Metric | Value |
| --- | ---: |
| Completed requests | 125 |
| Completed with non-zero `cachedTokens` | 122 |
| Completed cold misses (`cachedTokens = 0`) | 3 |
| Completed cache-hit rate | 97.6% |
| Prompt tokens across completed requests | 6,267,523 |
| Cached tokens across completed requests | 5,734,928 |
| Token-weighted cache reuse | 91.5% |

### Checkpoint Type on Completed Requests

| Restored checkpoint type | Count |
| --- | ---: |
| `leaf` | 112 |
| `system` | 5 |
| `branchPoint` | 5 |
| none / cold miss | 3 |

Interpretation:

- `leaf` dominates, which is what you want if the cache is following the latest reusable path.
- A `system` or `branchPoint` hit is **not automatically a bug**.
- In these logs, only `1` of the `10` non-leaf hits was a proven latest-leaf miss.

## “Latest Cached Identity” Math

I defined “latest cached identity” as:

- the deepest previously captured leaf
- from the same session / same prompt shape
- where the current request demonstrably includes that previous output

This is stricter than “same session” and avoids overcounting branched conversations as misses.

### Results

| Metric | Value |
| --- | ---: |
| Continuations with an inferable latest leaf | 111 |
| Continuations that hit that latest leaf | 110 |
| Continuations that missed that latest leaf | 1 |
| Latest-leaf request hit rate | 99.10% |
| Expected latest-leaf tokens | 5,683,012 |
| Actual reused tokens on those continuations | 5,676,547 |
| Latest-leaf token coverage | 99.886% |
| Total lost tokens from latest-leaf misses | 3,650 |

### By Continuation Kind

| Continuation kind | Candidates | Misses | Hit rate |
| --- | ---: | ---: | ---: |
| Tool-result continuations | 79 | 1 | 98.73% |
| User continuations after a completed assistant response | 32 | 0 | 100.0% |

This is the key answer to your question:

- the logs do **not** show a broad problem with “we do not catch the latest cached identity”
- and they especially do **not** show that problem on the “assistant responded after tool use, then user spoke” path

## Your Suspected Pattern: Tool Call -> Tool Result Response -> Next User

I looked specifically for the motif:

1. request `A` ends with `finishReason=tool_calls`
2. request `B` includes `A`’s assistant tool-call output and ends with `finishReason=stop`
3. request `C` includes `B`’s assistant response and ends with `lastRole=user`

Observed motifs: `8`

| Metric | Value |
| --- | ---: |
| Motifs found | 8 |
| Next-user turn hit latest canonical leaf | 8 |
| Next-user turn missed latest canonical leaf | 0 |
| Median gap between response turn and next user turn | 107 s |
| Max observed gap | 752 s |

So the specific hypothesis is **not supported** by the logs.

### What *does* happen on that path

The next user turn usually hits a `canonicalUserLeaf`, not the earlier `directToolLeaf`.

That is expected from the code:

- `selectHTTPLeafStoreMode` chooses `directToolLeaf` when tool calls were emitted, and `canonicalUserLeaf` otherwise for thinking templates.
  See `tesseract/Features/Agent/LLMActor.swift:1734-1744`.
- The server explicitly documents why old assistant turns cannot always be reused byte-for-byte on the next turn:
  older assistant messages may be re-rendered without `<think>...</think>` blocks, so the stable reusable path is the last-message boundary / canonical form instead.
  See `tesseract/Features/Agent/LLMActor.swift:1432-1445`.

### Practical consequence

If you were reading `cachedTokens` and seeing a **smaller** value on the first user turn after a tool loop resolved, that is usually:

- a **shallower but still correct canonical leaf**
- not a miss

Across the `8` resolved tool-loop motifs:

- median `canonicalUserLeaf - previous directToolLeaf` delta was `-280` tokens

So there is a real depth drop there, but in these logs it was still a **hit to the latest valid leaf** every time.

## The One Real Latest-Leaf Miss

There was exactly one proven miss:

| Field | Value |
| --- | --- |
| Session | `ses_265240aa4ffehFi9KXPA3IHYAu` |
| Parent request | `12-53-35-0010` |
| Current request | `12-53-38-0011` |
| Parent finish reason | `tool_calls` |
| Expected leaf type | `directToolLeaf` |
| Expected leaf offset | `7,856` |
| Actual restored checkpoint | `system` |
| Actual restored offset | `4,206` |
| Lost tokens | `3,650` |
| Gap between parent and current | `3 s` |

### Why this one matters

This was not a branch several minutes later.

- It was an **immediate** tool-result continuation.
- The next request clearly included the previous assistant’s tool-call output.
- The cache should have reused the latest direct-tool leaf and did not.

### Why I think this is a multi-tool continuation hole

In the failing sequence:

- the parent request had already built up several assistant-tool/tool-result pairs
- the failing continuation appended an assistant message with **4 tool calls** and then **4 tool result messages**

The direct-tool probe logic currently uses **one** synthetic tool result to determine the reusable path:

- `computeToolContinuationStoredTokens(...)`
- `LLMActor.swift:1804-1832`

Specifically:

- it renders the tool-call assistant turn
- then re-renders it with exactly one synthetic tool message:
  `Aqkz_tool_probe`
- then stores the shared prefix as the reusable direct-tool leaf

That looks like a plausible mismatch for real continuations that append **multiple** tool results.

I cannot prove that root cause from logs alone, but it is the strongest code-level hypothesis that matches the lone observed miss.

### Why this likely escaped existing coverage

The current E2E tool-loop benchmark only exercises a **single-tool** loop:

- `PrefixCacheE2ERunner.swift:543-547` explicitly says:
  “Call exactly one tool now... Emit only the tool call.”
- The continuation path then appends the generated tool calls and synthetic tool results for that single-tool scenario.
  See `tesseract/Features/Agent/Benchmark/PrefixCacheE2ERunner.swift:566-606`.

That means the benchmark covers:

- one tool call
- one tool-result continuation

It does **not** cover:

- assistant turns that emit multiple tool calls
- continuations that append multiple tool result messages

## Other Patterns Worth Noting

### 1. `no-canonical-restore-boundary`

Count: `2`

Both occurrences were small prompt cases that already had a `548`-token leaf hit and then skipped canonical leaf capture because there was no deeper canonical restore boundary.

They did **not** produce a later proven latest-leaf miss in the observed logs.

### 2. `droppedTooLargeForBudget` on SSD admission

Count: `17`

This is not the same as an immediate cache miss.

What it means:

- the leaf was captured
- but the SSD tier refused to persist that body because it was too large for budget

Observed impact in these logs:

- it is a persistence / restart / eviction risk
- it did **not** break the resolved-tool-loop -> next-user pattern
- one resolved-tool-loop motif still hit the latest canonical leaf `752` seconds later despite SSD body drop

So:

- this is a real operational pressure point
- but it is not the primary explanation for your suspected user-turn cache loss

### 3. Raw tool-marker warnings

Count: `3`

These requests logged:

- “Raw output contains tool call markers but no `.toolCall` events were emitted by library”

This is worth keeping an eye on, but in these logs:

- none of the warnings coincided with a proven latest-leaf miss

## What I Would Prioritize

### 1. Fix the direct-tool probe for multi-tool continuations

Most likely high-value change:

- make `computeToolContinuationStoredTokens(...)` probe with the **same number of synthetic tool results as emitted tool calls**
- not just one synthetic tool result

Why:

- the only real miss lines up with a continuation that appended multiple tool results
- the current probe shape only models a single tool result

### 2. Add a regression test for multi-tool direct-tool reuse

Specifically:

- assistant emits `N > 1` tool calls
- next request appends `N` tool results
- assert that the continuation hits the latest `directToolLeaf`

The existing E2E harness already covers the single-tool version, so this is a clean extension rather than a new benchmark class.

### 3. Add one more diagnostic field

On `directToolLeaf` capture and on lookup, log:

- emitted tool-call count
- appended tool-result count on the continuation request

That would make future miss triage much faster.

## Bottom Line

The logs say:

- overall cache reuse is good
- latest-leaf reuse is extremely good
- the path you suspected is **not** where the observed misses are

What the logs actually point to is narrower:

- a likely `directToolLeaf` reuse hole on an immediate **multi-tool** continuation
- not a general “assistant responded after tool use, then the next user turn skipped the latest cache” problem
