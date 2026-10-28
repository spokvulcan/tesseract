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
  for nothing. (Amended 2026-10-28: the gate counts the attention's
  tokens, see the amendment at the end.)

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

## Amendment 2026-10-28: the worth-it gate counts the attention's tokens

The gate weighed bytes. It counted the sliceable attention past the base, and
every whole-state layer whole, against the full payload. So a base extended a
turn of J tokens only if it held J/9 + W/(9a) tokens, with W the body's
whole-state bytes and a its attention's bytes per token. That second term
belongs to the model, not to the turn. On Qwen3.8-27B it is 261 tokens
(153,944,064 bytes of float32 GatedDeltaNet state, 65,536 bytes a token of bf16
attention), on Qwen3.5-4B PARO 175 (51,511,296 and 32,768), and no 27B base of
261 tokens or fewer could extend anything.

The gate's premise, that a near-full suffix buys chain complexity for nothing,
predates ADR-0012. A full write deletes its base at commit (`deferredDelete`),
so the base boundary stops being a Chain-Prefix Restore point, and ADR-0090
restores the request after every Shown Picture from exactly that boundary. In
the prefix-cache e2e, Step P2's picture turn extended at 89.94% of the full
payload on the 27B (a 434-token leaf over 280) and at 89.40% on the 4B (353
over 204). A 27B leaf past 451 tokens would have written full, and Step P3
would have restored the 177-token system checkpoint instead of the thread
before the picture. Under `turbo8v4` the term was 662 tokens, and the
`turbo8v4` e2e made no extension at all until ADR-0083's amendment of the same
day.

An extension always writes less than the full leaf, by the base's attention.
What it costs is one more copy of the whole-state layers on SSD, the one in
the base's segment: 154 MB on the 27B, 52 MB on the 4B. That copy is the
restore point's recurrent state (ADR-0012). Hydration maps the segments and
materializes only the ones a layer composes from, so the copy costs disk
space, not reads.

**Decision.** The gate compares the sliceable attention alone. A leaf extends
when its base holds at least a tenth of its tokens
(`SnapshotPayload.extensionMaxSuffixFraction` stays 0.9, now the most of the
leaf's tokens the suffix may hold). A body with no sliceable-attention layer
still admits full, since its segment would repeat the whole body, and so does
a near-root base under a tenth of the leaf's tokens. Tokens carry no
precision, so ADR-0083's amendment of the same day, which weighed a KV
Scheme's attention at full precision, has nothing left to do, and its
`storedForm` plumbing is removed.

The two rules differ only for a base less than W/(9a) tokens above a ninth of
the turn. A long thread extends each ordinary turn under both and already
keeps a whole-state copy per boundary. What moves is a short thread's first
turns and a big early jump, such as a 1,300-token picture over a 27B base
under 406 tokens (now 145). Each such turn now keeps its boundary, for one
more whole-state copy on SSD. A thread whose first leaf holds 500 tokens or
more is past the band for every turn under 2,000 tokens, so short-prompt
clients and the e2e are what change.

**Rejected.**

- Pricing each boundary, extending when the prefill it saves down to the next
  restore point (ADR-0011's recovery cost) outweighs its whole-state copy. It
  needs those restore points and an exchange rate between bytes and FLOPs,
  for a band of 261 tokens.
- Dropping the tenth too, so every leaf with a base extends. A turn more than
  nine times its base would keep a boundary next to the root, where the
  system checkpoint usually sits, for a whole-state copy.
- Keeping the rule and lengthening Step P's thread before the picture. The
  e2e would pass with room, a short-prompt client would still lose the
  boundary, and ADR-0083's scheme weighting would stay as a second rule.
- Keeping the base on a full write (`preserved` instead of `deferredDelete`),
  the duplicate-on-supersede ADR-0012 rejected.

**Consequences.** Step P2's base now extends any leaf up to ten times its
length, 2,800 tokens on the 27B (was 451) and 2,040 on the 4B (was 468). A
short-prompt conversation's chain keeps a whole-state copy per turn from its
second leaf on, as a long one already did. Each such extension deep-copies
the whole-state layers into its payload, the copy ADR-0069 budgets for at
check-in, and in exchange the body stays free for check-out while the suffix
writes, where a full payload holds the body's own arrays until its write
releases them (**Deferred Payload Extraction**). `LeafExtensionAdmissionTests`
pins the rule. A hybrid leaf whose whole-state outweighs its attention extends
from a base holding a tenth of its tokens and admits full under it, and a body
with nothing sliceable admits full. `TurboQuantSnapshotTests` still pins that
a scheme leaf extends where the full-precision leaf does.

**As built** (Qwen3.5-4B PARO and Qwen3.8-27B 4-bit, Debug, M3 Max 48 GB, the
gate set of `docs/prefix-cache-testing.md` on 6c8de286 and on the change,
compared by `scripts/gate-compare.py`). The 4B PARO e2e, the 27B e2e and the
27B `turbo8v4` e2e (38 checks each) and `--hybrid-cache-correctness` compare
identical to their base runs, apart from the Speculative Canonical Prefill's
skips and probe resolves, which follow the machine's load. No base run wrote a
leaf full over a base (four supersessions `transferred`, none
`deferredDelete`), so the token rule decides every leaf of these runs as the
byte rule did. Against the `turbo8v4` run on 7002c1fd, before either
amendment, the change differs exactly where ADR-0083's amendment did: the four
supersessions (`deferredDelete` to `transferred`), Step P3's restore (177 to
280 tokens, one hydration) and its DFlash2 acceptance (21/70 to 19/82). The
token rule alone keeps the scheme's turn boundaries. Rebased onto main with
the LFM3 models (#647) and the tool-prompt rendering (#650), the gate set and
the LFM3-1.7B e2e compare identical again (28efa123, main's gate code against
the change). LFM3's whole state is 2,940,928 bytes, five 256-token rings and
its short convolutions, against 12,288 bytes a token of global attention, so
its band is 27 tokens.
