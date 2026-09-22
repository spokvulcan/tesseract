# Compaction against the next turn's growth: preregistration

Refs [#554](https://github.com/spokvulcan/tesseract/issues/554) (item 6),
[#534](https://github.com/spokvulcan/tesseract/issues/534) and
[#533](https://github.com/spokvulcan/tesseract/issues/533). The
[2026-09-21 profile](../2026-09-21/README.md) set today's compaction
threshold. This folder registers how the retune is decided, before any
number is read.

## The question

Since #533 a restore reserves the prompt's rows, and any growth past a
cache's capacity reallocates the whole attention body once. Compaction (#534)
rebuilds a leaf whose retained capacity exceeds the threshold (the smaller of
a quarter of the body and 64 MiB) at its offset plus one 256-row step. A next
turn whose prompt adds more than 256 rows then copies the whole body again to
grow, although the capacity compaction freed could have absorbed it. Retained
capacity also holds RAM for as long as the leaf stays resident.

Only the threshold, the spare rows compaction leaves, and which exits compact
may change. The vendor's growth policy stays as it is.

## The measurement

No model run. The data is the owner's own prefix-cache diagnostics on this
host (`~/Library/Application Support/CacheDiagnostics/*.jsonl` and the rotated
`*.jsonl.old`, 2026-09-13 to 2026-09-22), read by
[`compaction_retune.py`](compaction_retune.py). The raw files stay private;
the summary records their SHA-256 and carries scalars only (no prompt text,
token ids, request ids or timestamps). Requests whose ids appear anywhere
under `benchmarks/` are benchmark campaigns and are excluded.

An **observation** is a leaf a request left behind:

- a **check-in**: the last `capturingLeaf` sample of a request that stored a
  leaf (`leafStore` with `path` live, boundary or direct), at the stored
  `leafOffset`;
- a **rewind**: a `leafRewind` event, at its lease offset.

Both samples are taken after compaction, so the **retained capacity before
compaction** is the unused full-attention bytes plus the bytes compaction
freed; in rows, that divided by the row size (logical bytes over the offset).
An observation is **compactable** when that retained capacity exceeds today's
threshold.

Its **next turn** is the first later request of the same model whose `lookup`
hit at exactly that offset. From it:

- the **suffix** it prefilled past the leaf (`newTokensToPrefill`), the rows
  its prompt reservation adds;
- the tokens it **generated** (its live leaf's offset minus its prompt), when
  it stored one;
- the leaf's **residency**: seconds from the observation to that lookup.

Reported per exit kind (check-in, rewind): counts; the distribution of
retained rows and MiB, of their share of the tree budget, of the next suffix,
of generation and of residency; retained MiB-seconds; and how often the next
suffix fits the retained rows and how often it fits today's 256 spare rows.

## The decision rule, fixed now

Let E be the compactable observations that have a next turn.

1. If E has fewer than 5 members, the owner's sessions do not exercise
   compaction enough to retune it: **keep today's rule**.
2. Otherwise let W be the share of E whose next suffix is over 256 rows but
   within the retained capacity: the growth copies compaction itself caused.
   - W at most 0.20: **keep today's rule**.
   - W above 0.20, and every such case a check-in: **compact only after a
     rewind**.
   - Otherwise: compaction keeps spare rows equal to the 90th percentile of
     E's next suffix, rounded up to a multiple of 256 and capped at 4096, the
     growth step's cap (at the cap this is "the growth cap as the compaction
     floor").

Whatever the rule selects, a toy decode through the Server Completion module
checks it: a next turn whose suffix fits the capacity a leaf keeps does not
reallocate, and the next turn after a compaction pays at most one whole-body
copy. The chosen policy and the numbers behind it are recorded below and in
ADR-0069's as-built notes.

## Results (taken 2026-09-22, after the rule above was committed)

[`summary.json`](summary.json) holds the scalars: the thirteen diagnostics
files' SHA-256, 1,742 benchmark request ids excluded, 387 observations, 222
of them paired with a next turn.

The script needed two fixes before its numbers could be read, and neither
changes a definition. Its first run failed on the sink's ISO 8601
timestamps. Those are whole seconds, so sorting by them put a leaf's store
ahead of its own capture sample whenever both fell in one second, and most
check-ins went uncounted. The script now keeps the sink's append order (a
day's rotated `.jsonl.old` first), and "later" means later in that order. A
run between the two fixes counted 71 observations and reached the same
decision.

| | Check-ins | Rewinds |
| --- | ---: | ---: |
| Observations | 384 | 3 |
| Compactable under today's threshold | 144 | 3 |
| Paired with a next turn | 219 | 3 |
| Retained capacity, median / max | 13.4 / 225.4 MiB | 272 / 272 MiB |
| Retained rows, median / max | 215 / 3,607 | 4,352 / 4,352 |
| Retained share of the tree budget, median / p90 | 0.13% / 0.47% | 53.7% (one reading) |
| Next suffix, median / max | 30 / 11,512 rows | 259 / 474 rows |
| Next suffix fits the retained rows | 163 of 219 | 3 of 3 |
| Next suffix fits 256 spare rows | 176 of 219 | 1 of 3 |
| Residency before the next turn, median / max | 1 / 467 s | 9 / 14 s |

Ordinary check-ins keep little: mostly what is left of the last 256-row
growth step, a fraction of a percent of the tree budget, for about a second
before the next turn. The share passes 100% only at 137K-token contexts on
the 27B, where the tree budget itself had fallen to 8.4 MiB. Of the 144
check-ins that count as compactable, 140 are short contexts (74 to 790
tokens) whose quarter-body threshold is a few MiB. They keep at most one
step, which compaction leaves alone, and their next turns add 17 rows at the
median and 92 at most. The other four are agent turns on the 27B at 43K to
46K tokens, where one decode growth step is thousands of rows: they kept 95
to 225 MiB, compaction freed 79 to 209 MiB of it, and their next turns added
29, 247, 378 and 400 rows. 56 next turns grew past everything their leaf
kept and would have grown whatever compaction did.

Rewinds keep as much as the longest check-ins: 105 to 272 MiB after
cancelled generations on the 27B, held for 5 to 14 seconds, in one reading
more than half the tree budget. Their next turns added 253, 259 and 474
rows.

### Decision

E has 45 members (42 check-ins, 3 rewinds). W is 4 of 45, 0.09: the two
long check-ins whose next turns added 378 and 400 rows, and the two rewinds
whose next turns added 259 and 474, all of which the freed capacity would
have held and 256 spare rows did not. That is under 0.20, so by the rule
registered above **today's rule stays**: the threshold (the smaller of a
quarter of the body and 64 MiB), one 256-row step of spare rows, and both
exits compacting. No code changes for item 6.

The four cases are worth the record. Compaction there cost the next turn a
whole-body growth copy it would not otherwise have made, and in exchange
freed 88 to 256 MiB until that turn. For the two check-ins
the next turn came within the same second, so the freed capacity bought
nothing; for the rewinds it was held 5 to 14 seconds. The sample has four
such cases, split between the exits; a later measurement with more long
agent turns can revisit spare rows at long contexts.

The toy decode checks the kept rule through the Server Completion module
(`ServerCompletionKeyedSequencingTests`): a next turn that fits the capacity
its leaf kept prefills and decodes without reallocating
(`aNextTurnThatFitsTheKeptCapacityDoesNotReallocate`), and the resend after
a compacted rewind grows the body exactly once
(`theNextTurnAfterACompactionGrowsTheBodyAtMostOnce`).
