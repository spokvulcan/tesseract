---
status: accepted
---

# Superseded leaf boundaries stay restorable as chain-prefix restore points

This records the restore-floor fix decided in the 2026-06-13 grilling of
the interrupt/Think-Strip-Rewind incident. See `CONTEXT.md` →
**Chain-Prefix Restore**, **Segment Chain**, **Tool Stretch**,
**Think-Strip Rewind**, **Stretch Abandonment**.

The observed incident (2026-06-12 OpenCode session, trace + request
corpus archived locally): a user interrupted a hung tool mid **Tool
Stretch** and sent a steering message. The new user message moved the
template's think-strip boundary, so the request's token path diverged
from the cached spine at offset 41,897 — but the deepest restorable
snapshot at-or-below the divergence was a `.branchPoint` at 21,125. The
turn re-prefilled 66,370 tokens in the foreground: 91.3 s TTFT. Every
intermediate per-turn leaf between those offsets had been superseded by
**Leaf Extension Admission** — their *identities* (tree refs) were
consumed at writer commit, while their *bytes* remained on SSD as
inherited segments of the live leaf's chain, tiled at exactly the
historical leaf boundaries. The data to restore from 41.5k was on disk;
nothing could address it.

The decision: a node whose leaf was consumed by an extension keeps a
**chain-prefix restore point** — a tree-side reference into the owning
entry's chain (`owner snapshot ID + boundary offset`), hydrated by
composing only the leading **Snapshot Segment**s covering
`[0..boundary]`. Chains keep exactly one owner and the manifest schema
is unchanged; the restore floor on any mid-history divergence tightens
from "nearest surviving durable checkpoint" to "nearest turn boundary".

Key shape choices, and why:

- **Tree-side reference, not a manifest entry.** ADR-0010's single-owner
  invariant ("no cross-entry references") holds: no refcounting, no
  cascade-delete, no link graph in the corrupt-manifest rebuild.
  Eviction, deletion, and budget still operate on whole chains. When a
  chain head is deleted or evicted, its dependent restore points are
  cleared through the same eager backing-loss plumbing that clears
  committed refs (the committed-ref index added for recovery-cost
  eviction); a cleared restore point degrades the next lookup to the
  next-shallower point, never corrupts it.

- **Restore points only at segment boundaries.** Segments are the
  slicing unit, and their boundaries are historical leaf offsets — turn
  boundaries, which is where divergent futures actually fork (a
  **Think-Strip Rewind** forks at the first assistant turn of the
  stretch; client-side history edits fork at message boundaries).
  Arbitrary-offset hydration would require re-slicing segment files and
  buys nothing the turn grid doesn't.

- **ADR-0010's "dead bytes" are load-bearing here.** Non-sliceable
  layers (recurrent, rotating, chunked) ride whole in every segment,
  last-segment-wins. For a prefix hydration up to segment *k*, segment
  *k*'s copy is the recurrent state exactly as of that boundary —
  written at that leaf's own admission. Prefix restore is therefore
  correct for hybrid models *because* ADR-0010 chose to carry those
  copies per segment rather than rewrite files.

- **Warm start derives restore points from chain descriptors.** The
  inherited-segment list (v8 schema) already carries each segment's
  token range, so the directory-walk rebuild can reconstruct every
  restore point without a schema bump or new persisted state.

- **Recovery-cost pricing follows the backing.** Under ADR-0011, a node
  backed by a chain prefix prices its recovery as hydration of the
  prefix bytes (seconds), not re-prefill FLOPs (minutes) — which is
  what makes keeping these nodes' refs nearly free and evicting their
  RAM bodies safe.

Considered and rejected:

- *Duplicate-on-supersede* (write the boundary prefix as its own SSD
  entry when superseded). Simple, but gigabytes of duplicate writes per
  stretch and double budget counting — re-introducing exactly the write
  amplification ADR-0010 removed.
- *Preserve-and-share* (keep the base entry alive while the extension
  inherits the same files). Cross-entry file sharing needs refcounting
  and breaks supersession's delete path — the parent-reference DAG
  ADR-0010 already rejected, through a side door.
- *Cross-entry chain extension* (letting a later leaf — e.g. a
  speculated canonical spine — admit only its suffix with its prefix
  resolving through another entry's chain). Byte-optimal but violates
  single-owner with real lifecycle coupling. Explicitly deferred, not
  adopted: the post-rewind turn self-heals with a full write instead.
- *No floor — rely on Stretch-Abandonment speculation alone.* When the
  user interrupts and types within seconds, no background window
  exists; the foreground restore then starts at a deep-frozen
  checkpoint. The floor is also what keeps the speculative pass's
  runway short enough to finish inside typical idle windows.

Consequences: every offset where a leaf was once extended remains a
restore point for divergent futures at zero additional disk bytes;
lookup must consider chain-prefix points alongside owned refs when
resolving the deepest usable snapshot; hydration gains a compose-prefix
path (read fewer segments than the chain holds — strictly less work
than a full-chain hydration); and the floor guarantee is only as good
as canonical-render fidelity — a render that drifts a few tokens past a
boundary (observed: 33 tokens at an answer tail in the incident corpus)
makes the boundary unusable, so the fidelity gate ships first.
