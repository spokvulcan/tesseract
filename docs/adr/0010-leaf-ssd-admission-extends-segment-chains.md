---
status: accepted
---

# Leaf SSD admission extends segment chains via ownership transfer

Issue #78: every agent turn captured a full leaf snapshot, serialized
every layer's whole KV state to SSD, then deleted the previous turn's
leaf. A live OpenCode session admitted 928 MB → 1,432 MB files within
minutes — each rewriting a prefix that was already on disk, byte-for-byte,
in the file it was about to delete. Write amplification scaled with
conversation length, not new content.

**Decision.** When a new leaf strictly extends an SSD-backed ancestor
leaf on the same radix path, the admission persists **only the suffix**
as one new segment file, and supersession **transfers ownership** of the
old leaf's segments to the new leaf instead of deleting them. A committed
leaf is now a **segment chain**: an ordered list of files tiling
`[0..offset]`, owned by exactly one manifest entry. Per-turn writes drop
from O(conversation) to O(new tokens). See `CONTEXT.md` → **Snapshot
Segment**, **Segment Chain**, **Leaf Extension Admission**.

## Why these shapes

- **Ownership transfer, not parent references.** A delta pointing at the
  base as a separate live entry (a parent/child DAG) needs refcounting or
  cascade-delete in the LRU cut, blocks supersession from deleting files
  with children, and forces the corrupt-manifest rebuild to reconstruct a
  link graph. Single-owner chains keep every invariant unchanged in shape
  — one `SnapshotRef` ↔ one manifest entry; eviction, deletion, budget,
  and hydration all operate on whole chains. The transfer happens
  atomically inside the ledger-locked commit, so no GC ever runs.

- **Heterogeneous per-layer encoding.** `KVCacheSimple` /
  `QuantizedKVCache` slice cleanly along the token axis (quantization
  groups pack along the head dim, so a token slice never splits a group).
  Recurrent (`ArraysCache`/Mamba), rotating, and chunked layers do not
  slice; they ride whole in every segment, last-segment-wins at
  composition. Their duplicated bytes are dead but bounded (recurrent
  state is O(1) in sequence length).

- **Worth-it gate.** If the estimated suffix payload is ≥ 90% of the full
  payload (mostly non-sliceable layers, or a near-root base), the leaf
  admits full — a "delta" rivaling the full write buys chain complexity
  for nothing.

- **Enqueue-validate, commit-fold.** The base (deepest ref-bearing
  ancestor leaf) is chosen tree-side, validated and shielded from the LRU
  cut under the ledger lock at enqueue, and folded into the new entry
  atomically at commit. The base entry stays authoritative in the
  manifest until commit, so a crash in the window warm-starts at the base
  offset rather than losing the conversation. If the base vanishes
  mid-flight, the extension self-vetoes: the leaf stays RAM-only and the
  next turn writes full.

- **RAM-only admissions preserve ancestor backings.** Previously any leaf
  admission deleted the superseded leaf's file, so ADR-0009's RAM-only
  partial settle destroyed the canonical leaf's SSD copy. Now a RAM-only
  leaf keeps the ancestor's backing alive (body dropped, ref kept) as the
  warm-start fallback and next-turn extension base — closing the residual
  churn ADR-0009 documented.

- **Schema v7 → v8.** Descriptors gain the segment list and own-file base
  offset; each per-file header carries the full chain known at write time,
  so the directory-walk rebuild reconstructs chains by picking head
  descriptors (files nobody lists as inherited). v7 manifests are wiped on
  first boot — a cache, not data.

## Rejected

- *Write-behind debounce.* Kills the per-turn pattern, but every persist
  that does happen is still a full rewrite (the fix #78 named was
  extension-aware admission), and it trades durability — a crash loses the
  whole leaf, not just the last turn's suffix.
- *Delta against checkpoints only (no chains).* Degenerates: a linear
  session's only checkpoints are the shallow `.system` one and sparse
  `.branchPoint`s, so the "delta" is near-full. Fixing that means
  redesigning checkpoint placement — a separate project.

## Consequences

Hydrating a long session's leaf reads N segment files and composes per
layer (sequential reads; peak RAM ≈ one snapshot plus one layer). The SSD
budget counts chain totals, so a conversation's disk footprint is
unchanged — only write traffic collapses. A dropped extension costs only
the suffix; the base keeps its manifest entry and tree ref, stays
hittable, and the next turn extends it again.
