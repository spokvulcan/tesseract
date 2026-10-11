# ADR-0093: One Decode Start starts decode for every generation path; the Decode Handover says where the KV Scheme converts

- Status: Accepted (built; loaded-model gates in As built)
- Date: 2026-10-11
- Builds: ADR-0087 decision 8's "planned Decode Start (one module that
  converts the KV cache once and starts decode for every generation path)",
  ahead of the rest of ADR-0087, which stays Proposed
- Amends: ADR-0083 (decision 2 and its amendment: where each path converts
  is now one rule in one module), ADR-0053's 2026-07-22 amendment (the
  Generation Logit Processor seam loses its `pathQuantizesKVUpFront` flag)
- Relates to: ADR-0016 (no new Model Session verb), ADR-0079 (engagement
  stays in `Speculation.plan`), ADR-0044 (the Prefill Strategy route keeps
  its own conversion), ADR-0006 and ADR-0007 (no iterator past the app's
  prefill applies `kvBits`), ADR-0078 decision 8 and ADR-0069 decision 8 (the
  behaviour bar)

## Context

Six production paths start decode: the keyed turn's standard iterator, the
keyed turn's DFlash2 iterator at a split, the keyed whole-prompt arm (MTP),
the Unkeyed Completion, and the Raw Generation Start's speculative and
standard arms. Each decided by itself where the KV Scheme converts, by four
mechanisms: an explicit `session.quantizeKVCache` after the app's prefill
(the keyed standard path), `maybeQuantizeKVCache` inside the app iterator
after its `prepare` (Unkeyed), the vendor iterator built from parameters
(the Prefill Strategy's single-shot arm), and the DFlash2 iterator with a
read-back (the keyed split, the raw speculative arm). 27e49510 was a decode
start that forgot to convert. Around the conversion, each path also handed
the decoded array to its owner by hand (`handoff?.cache.adopt(liveCache)`,
`FinalGenerationCache(...)` at four sites), badged a speculative arm (three
sites), and started the token loop through one of two parallel iterator
enums (`KeyedDecodeIterator` in `ServerCompletion`,
`SpeculativeDecodeIterator.startGeneration` in `Speculation`). The scheme
string was parsed twice (`RequestKeyingPhase`, `RawGenerationStart`), and
`RequestFacts.decodeParameters` carried a `kvBits` reset only the decode
starts read. A `pathQuantizesKVUpFront:` flag rode the Generation Logit
Processor seam to say which paths had converted before their iterator.
Seven commits had to touch both `ServerCompletion` and `RawGenerationStart`.

## Decision

1. **One Decode Start.** `DecodeStart` (`Features/Agent/DecodeStart.swift`)
   is the one module that starts decode. Its interface is four members:
   `make(_:cache:session:parameters:)` builds the iterator, `cacheOwner(lease:)`
   hands the decoded array to its owner, `reportPrefillFinished(_:to:)` emits
   the badge and the prefill's numbers, and `startLoop(...)` starts the token
   loop once. Every path calls it, through the Model Session's existing verbs:
   no verb was added (ADR-0016), and engagement stays in `Speculation.plan`
   (ADR-0079).
2. **The Decode Handover names who prefills what is left.** Four cases:
   `standard` (the app prefilled all but the last prompt token), `speculative`
   (the Speculation Plan's iterator prefills from a split, or from zero),
   `preparing` (the app's state-threaded iterator prefills the whole prompt)
   and `prefillStrategy` (the Prefill Strategy route prefills into a cache of
   its own). The case fixes the verb and the conversion point, in one switch:

   | Handover | Verb | The scheme converts | The prompt's last token runs over |
   |---|---|---|---|
   | standard | `quantizeKVCache`, then `makeDecodeIterator` | in the Decode Start, before the iterator | the converted cache |
   | speculative | `makeSpeculativeDecodeIterator` | in the DFlash2 iterator, before its final position; read back | the converted cache |
   | preparing | `makePreparingDecodeIterator` | in the iterator, after its prefill; read back | full precision |
   | prefillStrategy | `makeRawDecodeIterator` | in the route (ADR-0044) | full precision, or converted when the chunked arm has penalties |

   Each verb gets the request's parameters minus what is no longer its to
   apply: the standard iterator gets no KV plan at all (the cache is
   converted), and no iterator past the app's prefill gets `kvBits`
   (ADR-0006, ADR-0007). `RequestFacts.decodeParameters` is gone.
3. **The decoded array has one way to its owner.** `make` returns with the
   caller's `cache` set to the array decode advances, so no caller reference
   keeps the layers a conversion replaced. `cacheOwner(lease:)` gives it to a
   leased leaf's `FinalGenerationCache`, which adopts it, or to a new owner.
4. **One loop start, one badge.** The Decode Start's iterator enum (the
   app's state-threaded iterator, a Speculation Plan's iterator, the vendor
   iterator) absorbs `KeyedDecodeIterator` and
   `SpeculativeDecodeIterator.startGeneration`; the vendor iterator keeps
   upstream's loop. The speculation badge is emitted in one place, immediately
   before `prefillFinished`, which is where every speculative path emitted
   it, so the order of every path's progress events is unchanged.
5. **The penalty seed rides the handover.** The standard handover carries
   `fullText`, the text the app prefilled, to seed the iterator's penalties;
   every other kind seeds them with the whole prompt it prefills. The
   Generation Logit Processor seam keeps two shapes, an app processor
   (`processor(for:)`) for an iterator that holds it and a component
   (`components(for:)`) for an iterator built from parameters. Its
   `pathQuantizesKVUpFront` flag is gone: the one carve-out it encoded — the
   single-shot arm's explicit-processor init runs no KV plan, so with
   `kvBits` it keeps the vendor's own penalties (theoretical, #252) — now
   sits in `PrefillStrategy`, the only place it applies.
6. **The scheme is parsed in one place.** `KVScheme(_ parameters:)` reads
   the vendor's raw string back; Request Keying and the Raw Generation Start
   both call it.
7. **A strict pure refactor.** The bar of ADR-0078 decision 8: diagnostics,
   the order of steps, MainActor hops, Model Session entries and memory peaks
   are unchanged. Each path keeps its conversion point, its error scopes and
   checks, and the order of its progress events and loop start (the Raw
   Generation Start still starts its loop before the badge). The one oddity
   the inventory found stays as a named row: the chunked Prefill Strategy
   arm converts before its prime forward only when penalties are configured
   (`PrefillExecutor.makeIterator`'s explicit-processor branch), and after it
   otherwise. Making it convert like the standard handover is a one-row
   follow-up with its own check.

## Considered options

- **A Model Session verb that starts decode.** Rejected by ADR-0016: the
  verbs mirror model operations, and the Decode Start's rule is above them.
- **Move the conversion out of the iterators that prefill.** The preparing
  iterator and DFlash2 convert at the end of a prefill that runs inside
  their init; pulling the conversion out needs a split `prepare` verb and a
  second iterator form (ADR-0083's amendment rejected it). The Decode Start
  owns the decision instead: which verb runs, with which part of the KV plan,
  and the read-back.
- **Unify the badge after the loop start, or before `prefillFinished`
  everywhere by moving the raw loop start.** Either reorders one path's steps.
  The badge belongs to the prefill report, which every path already emits.
- **Fold the Prefill Strategy route into the standard and preparing
  handovers.** The raw chunked arm decodes through the vendor iterator, which
  differs from the app's state-threaded one in its loop and memory hygiene;
  swapping it is a behaviour change.

## Consequences

- `ServerCompletion` loses `KeyedDecodeIterator`, three iterator builds, the
  read-back, both adoption sites and both badges; `RawGenerationStart` builds
  through one `make` for both arms. Adding a decode path is one
  `DecodeHandover` case, one verb call in `make`, one table row.
- `DecodeStartTests` is the test surface: one table of handover × KV Scheme ×
  penalties over the toy Model Session, reading the verbs called, the text,
  the cache form the prompt's last token and the first decode step ran over,
  the badge and the kept array; plus the `kvBits` carve-out row and the
  lease's adoption. It replaces the per-path wiring tests in
  `GenerationLogitProcessorTests`, the Unkeyed Completion's scheme test and
  `RequestFactsTests`' decode-parameters checks. The toy's DFlash2 verify
  moves only full-precision rows, so the speculative rows run without a
  scheme; the loaded-model gates and the vendor's
  `testDFlash2IteratorOverTurboQuantCache` cover DFlash2 over a scheme.

## As built (2026-10-11)

Measured on a MacBook Pro, Apple M3 Max, 48 GB, macOS 27.0.1.
`ServerCompletion.swift` went from 2,755 to 2,669 lines and
`RawGenerationStart.swift` from 156 to 139; `DecodeStart.swift` is 179.

Unit tests: the whole target passes (3,646 tests, 7 skipped), and
`DecodeStartTests` holds 30 table rows. Dropping the standard handover's
conversion fails the table and the lease row.

Loaded-model gates, each run once on `main` (fa438b35) and once on this
change, on the same Debug configuration:

- `scripts/dev.sh prefix-cache-e2e` on `qwen3.5-4b-paro`: 38 of 38 checks
  pass on both; the check lines match with timings stripped.
- The same e2e on `qwen3.8-27b` with its DFlash2 draft resident: 38 of 38
  on both. Every turn decodes through DFlash2, and the 32 acceptance lines
  (`decode=dflash2 accepted=N/M`, keyed splits, the agent's raw whole-prompt
  turns and the image turns) are identical.
- The same with `TESSERACT_E2E_KV_SCHEME=turbo8v4`: 37 of 38 on both, the
  same 32 acceptance lines. The one failure is `main`'s own and reads the
  same on both sides (`requestP3_line_restores_the_thread_up_to_the_picture:
  cachedTokens=177 expected ≥ 280`, Step P under a scheme); it is outside
  this change.
- `scripts/dev.sh hybrid-cache-correctness`: 12 of 12 on both, every restore
  bitwise (`maxAbsDiff=0.0`); the summary lines are identical.
- The diagnostics of each e2e run, before against after, with ADR-0078's
  masking: ids, timings, process memory, the SSD writer's pending counters,
  eviction recency and segment file sizes masked; the periodic memory
  samples, budget re-measures and pressure changes dropped; lines written
  from other tasks (system events, the one-second-later `afterRelease`
  sample, the Speculative Canonical Prefill's lines and admission probes)
  compared per request as multisets, everything else per request in order.
  Two runs of `main` compare clean under this method. Before against after:
  0 of 1,959 (4B), 2,020 (27B) and 2,018 (27B, turbo8v4) in-order lines
  differ, and the requests appear in the same order. Of the 172–174
  other-task lines, the only differences are the Speculative Canonical
  Prefill's scheduling race — a seed that finds the next request already
  started logs `not-idle`, one that wins logs its probe and
  `below-residual-threshold` — which went both ways between the two runs
  (4–8 lines per comparison) and follows the machine's load: the `main`
  runs shared it with other builds.
