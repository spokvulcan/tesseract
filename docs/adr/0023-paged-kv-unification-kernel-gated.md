---
status: accepted
---

# Paged KV unification for batch lanes, gated on a kernel microbenchmark

PRD #151 said the snapshot-vs-paged storage verdict must be decided "only
against a real batch workload; deciding from priors is forbidden." The
2026-07-05 grilling deliberately amended that: with the batch engine
(ADR-0022) committed as a feature, building deep-copy lanes first and paged
lanes second means building the storage layer twice, and subagent fan-out —
the primary workload — is the worst case for deep-copy redundancy (N lanes
holding byte-identical copies of the same long system-prompt prefix). The
deviation is recorded here, and falsifiability is restored as a pre-registered
gate instead of a workload measurement.

## Decision

The radix tree's RAM tier becomes refcounted **KV Pages** (the SGLang shape,
full unification — not a page pool bolted beside the snapshot tree): restore
is a refcount bump, capture of an already-resident prefix is a no-op, and a
page with a live lane reference is structurally unevictable. The SSD tier
survives as-is — segments, chains, and ledger store bytes; hydration writes
into pages.

**Kernel gate (spike phase 0):** the enabling kernel — attention gathering
*quantized* KV (the production cache is `QuantizedKVCache`; pages must align
to quantization groups) from non-contiguous pages — is microbenchmarked first,
on the ParoQuant custom-Metal-kernel precedent. Pre-registered threshold:
decode overhead vs contiguous attention ≤15% at N ∈ {1, 4} on one small model
and ornith-35b, with no stability regressions. Fail → v1 falls back to
deep-copied per-lane caches plus a duplicate-prefix-bytes meter, and paged
waits for a better kernel story; the batch engine is unaffected either way.

## Considered options

- *Deep copies + redundancy meter, paged only if a pre-registered trigger
  fires on real fan-out data* — the PRD's original posture. Rejected to avoid
  building the lane storage layer twice; the kernel gate keeps the exit.
- *Copy-on-write restore* (alias shared prefix, copy on write): reintroduces
  the exact live-buffer aliasing the deep-copy fix banned after the
  InvalidResource SIGABRT, under the workload most likely to retrigger it,
  while the MLX-core attribution remains contested. Not acceptable.
- *Paged lanes beside the snapshot tree*: pays for kernels and refcounting
  while keeping the boundary copies they were meant to kill.

## Consequences

- `ActiveInferenceReserve` loses its `2×` capture-copy factor: a lane's
  marginal RAM is its new decode pages, since shared prefix pages are
  refcounted tree residents — admission arithmetic admits more lanes than
  today's math suggests.
- The Restore Pin dissolves into refcounts: a lane holding page references
  *is* the pin, so Budget Floor membership for in-flight restore paths becomes
  structural rather than bookkept.
- Parts of the just-shipped RAM-tier body machinery (deep-copy capture/
  restore, `hasResidentBody` semantics) are reopened; ADR-0018/0019's
  byte-denominated budget, floor, and Leaf Home Guarantee carry over
  unchanged in meaning.
- The decode-shape spike (ADR-0022) runs on the paged layout if the gate
  passes — contiguous-KV measurements would not transfer.
