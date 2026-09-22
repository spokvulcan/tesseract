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

## Results

Not taken yet.
