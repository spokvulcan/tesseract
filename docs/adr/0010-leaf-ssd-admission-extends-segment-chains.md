---
status: accepted
---

# Leaf SSD admission extends segment chains via ownership transfer

This records the fix direction for issue #78 (SSD churn: ~1 GB leaf
admission per agent turn via supersession), decided in the 2026-06-12
grilling. See `CONTEXT.md` → **Snapshot Segment**, **Segment Chain**,
**Leaf Extension Admission**.

The observed pattern: every agent turn captured a full leaf snapshot,
serialized every layer's whole KV state to SSD, then superseded —
deleted — the previous turn's leaf. A live OpenCode session admitted
928 MB → 1,432 MB files within minutes, each one rewriting a prefix
that was already on disk, byte-for-byte, in the file it was about to
delete. Write amplification scaled with conversation length, not new
content, and the writer competed for I/O during generation.

The decision: when a new leaf strictly extends an SSD-backed ancestor
leaf on the same radix path (the leaf it is about to supersede), the
admission persists **only the suffix** as one new segment file, and
supersession **transfers ownership** of the old leaf's segments to the
new leaf instead of deleting them. A committed leaf is now a **segment
chain**: an ordered list of files tiling `[0..offset]`, owned by exactly
one manifest entry. Per-turn SSD writes drop from O(conversation) to
O(new tokens).

Key shape choices, and why:

- **Ownership transfer, not parent references.** A delta could have
  pointed at the base as a separate live entry (a parent/child DAG).
  Rejected: it needs refcounting or cascade-delete in the LRU cut,
  supersession could no longer delete files with children, and the
  corrupt-manifest rebuild would have to reconstruct a link graph.
  Single-owner chains keep every existing invariant unchanged in shape:
  one `SnapshotRef` ↔ one manifest entry; eviction, deletion, budget,
  and hydration all operate on whole chains. Supersession transfers
  atomically inside the ledger-locked admit, so no GC ever runs.

- **Heterogeneous per-layer encoding.** `KVCacheSimple` and
  `QuantizedKVCache` state slices cleanly along the token axis
  (dim −2; quantization groups pack along the head dim, so a token
  slice never splits a group — verified against the vendored
  `mlx-swift-lm`). Mamba/`ArraysCache` recurrent state, rotating
  windows, and chunked caches do not slice; those layers ride whole in
  every segment, last-segment-wins at composition. Earlier segments'
  copies become dead bytes — bounded (recurrent state is O(1) in
  sequence length) and accepted over rewriting files.

- **Worth-it gate.** If the estimated suffix payload is ≥ 90% of the
  full payload (mostly non-sliceable layers, or a near-root base), the
  admission writes full. A "delta" larger than what it replaces is
  never written.

- **Enqueue-validate, commit-fold.** The base is chosen tree-side (the
  deepest ref-bearing ancestor leaf), validated under the ledger lock at
  enqueue (pending, in-flight, or resident — FIFO guarantees the base
  is settled before the extension is processed), shielded from the LRU
  cut while the extension is pending, and folded into the new entry
  atomically at commit. If the base vanishes mid-flight (hydration
  failure, dropped pending write), the extension self-vetoes: the leaf
  stays RAM-only and the next turn writes full. Crash story: the base
  entry stays authoritative in the manifest until the extension
  commits, so a crash during the window warm-starts at the base offset
  — strictly better than losing everything.

- **RAM-only admissions now preserve ancestor SSD backings.** Before:
  any leaf admission deleted the superseded leaf's file, so ADR-0009's
  RAM-only partial settle destroyed the canonical leaf's SSD copy. Now
  a RAM-only leaf keeps the ancestor's backing alive (body dropped, ref
  kept) — it remains the warm-start fallback and the extension base for
  the next SSD-backed leaf. This closes the residual churn ADR-0009
  documented.

- **Schema v7 → v8.** Descriptors gain the segment list and the own-file
  base offset; the per-file embedded header carries the full chain known
  at write time, so the directory-walk rebuild reconstructs chains by
  picking head descriptors (files nobody lists as inherited) and
  dropping entries whose chain is broken. v7 manifests are wiped on
  first boot under v8 — a cache, not data.

Considered and rejected:

- *Write-behind debounce* (defer the SSD persist until the leaf
  survives a quiet period). Simpler, and kills the per-turn pattern —
  but every persist that does happen is still a full rewrite, the
  first-class fix the issue named was extension-aware admission, and
  debounce trades durability (a crash loses the whole leaf, not just
  the last turn's suffix).
- *Delta against checkpoints only* (no chains). Degenerates: in a
  linear agent session the only committed checkpoints are the shallow
  `.system` one and sparse `.branchPoint`s, so the "delta" is
  near-full. Fixing that would mean redesigning checkpoint placement —
  a second project.
- *Per-partition admission rate caps.* Blunt: still writes full
  snapshots, and a badly timed cap skips exactly the leaf that needed
  durability.

Consequences: hydration of a long session's leaf reads N segment files
and composes per layer (sequential reads; peak RAM ≈ one snapshot plus
one layer); the SSD budget counts chain totals, so one conversation's
disk footprint is unchanged — only the write traffic collapses; dead
non-sliceable-layer bytes accumulate along a chain (revisit if hybrid
recurrent models make them material); and a dropped extension can leave
an unreachable-but-warm-start-recoverable base entry behind until the
LRU cut reclaims it.
