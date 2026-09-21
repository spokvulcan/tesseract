---
status: accepted
---

# Prefix-View Checkpoints and the Compressed Warm Tier

PRD #520 removes duplicated attention bytes at rest while preserving the
single-owner rule of ADR-0064 and the copy-on-write rejection of ADR-0023.
Interior checkpoints become Prefix-View Checkpoints; cold RAM bodies may
become Warm Bodies. Every restore still copies or moves, never aliases a
running generation's buffers.

## Prefix-View Checkpoints

Planned non-system checkpoints and transient Think-Strip Rewind and
Speculative Canonical Prefill boundaries own whole-state layers,
metadata and an absolute token offset. Capture still synchronizes prefill
before copying whole-state layers. System checkpoints keep owned full bodies:
they are small, reused by every conversation, and must survive a leased leaf.

Snapshot Resolution chooses a resident, unleased, full-body descendant in the
same partition as the Backing Leaf, nearest by offset, preferring an owned
uncompressed body over a Warm Body at equal offset, then most recent.
The pure resolution ladder chooses; the manager performs effects. Without a
backer, resolution falls through to the view's committed Snapshot Ref, then a
Chain-Prefix Restore point, then a shallower hit. No backer identity is stored
on the view. A temporary Leaf Lease makes a backer unavailable until check-in.

View Materialization runs inside the Model Session: slice attention rows to
the view offset, deep-copy and evaluate them, and deep-copy the view's own
whole-state layers. Quantization packs along the head dimension (ADR-0010),
so token-axis slicing stays exact; image key-path index equals KV offset
(ADR-0007). A Restore Pin protects both nodes. Leaf Checkout keeps its
`checkpoint` copy reason. Views count only whole-state bytes and are never
eviction candidates; Leaf Handoff, Leaf Lease and Leaf Rewind keep their meaning.

The tree never materializes a view when its last Backing Leaf departs. A view
holds no attention bytes, so losing every backer is not an eviction: its
durability is exactly its SSD admission (ADR-0019's #526 amendment).
Without a resident or temporarily leased descendant, Snapshot Ref or
chain-prefix point, the view becomes empty and the usual topology self-heal
applies. A leased descendant preserves the checkpoint for return, but cannot
serve a restore during the lease.

#526's end-of-turn SSD admission slices the checked-in leaf at the
extraction edge and detach every retained array on the Metal thread. The
writer receives a full snapshot at the view offset, in the unchanged segment
format. Such a payload never blocks the leaf's next checkout. Adaptive Write
Eagerness and the type-protected SSD cut keep their existing meanings.

## Compressed Warm Tier

#527 introduces compression behind an Eviction Configuration flag, default
off until #528 passes. On drain, compression precedes Snapshot Demotion in
the existing eviction-selection order. Leased leaves, the Budget Floor's
most-recently-extended leaf, system bodies and already-quantized partitions
are exempt. #529 adds the Hot Leaf Set to compression's exclusions, preserving
the most recently checked-in leaves of up to two paths by default. A successful
Leaf Admission or Lease check-in queues an opportunistic pass at the Model
Session's next quiescent point. It compresses cold leaves while RAM remains
above a configured fraction of the ceiling (initial opt-in default 0.75),
without demotion. The path limit and fraction belong to Eviction Configuration;
neither is a production tuning measurement. Advancing a path replaces its set
entry, and lookups and rewinds do not change check-in order. Leases remain
exempt independently of the path limit. The Budget Floor and ordinary demotion
eligibility are unchanged. A pressure drain takes precedence over opportunistic
work and keeps its existing compression-before-demotion guarantee.

Compression uses the vendor's quantize-to-cache conversion into fresh arrays,
8 bits, group size 64, affine by default. Whole-state layers stay unchanged;
fp16 arrays are dropped only after the warm arrays are evaluated. Restore
dequantizes into fresh fp16 arrays, reports `warmBody`, and never hands off.
Live fp16 KV stays fp16 (#252). A Warm Body can back a view (#530): slice in
quantized form, then dequantize. Nearest offset wins, with fp16 preferred at
equal offset before recency.

The pre-registered #528 gate requires the canonical-echo fidelity threshold
of the fp16 baseline and warm-hit TTFT no greater than fp16 restore-by-copy
TTFT plus measured dequantization cost. Only a passing 8-bit run can enable
the default. Four bits is an experiment under the same gate. Loaded-model
runs belong to the owner and require the resource plan and memory stop
threshold specified in the capture-handoff report.

Behind that gate, #531 makes Stored Form part of partition identity and SSD
chains homogeneous. Extension suffixes and view payloads are quantized at
extraction; hydration yields Warm Bodies; Warm-Start Plan drops stale forms.
The parity gate must pass again for SSD-hydrated warm bodies. ADR-0010's
amendment belongs to that slice. Compression-before-demotion amendments to
ADR-0011 and ADR-0018 belong to #527; the Pressure-Reactive Budget's band,
Budget Floor and eviction score retain their meaning.

## As built

#524 implements planned branch-point views in RAM, view restore and its
telemetry. #525 makes last-user and last-message boundary helpers request-local
views. For a structurally valid think-stripping stop turn, the quiescent fed
path checks in a full RAM leaf before canonical reconstruction. This uses the
existing handoff/copy eligibility and does not register a canonical Emitted Path
or write an extra SSD payload. Resolution chooses a current Backing Leaf by the
same pure ladder without inserting the transient view into the tree, then pins
that leaf. Stop, tool, and abort speculative seeds retain only the view, never
a backer identity.
A leased or departed backer falls through to the existing boundary re-prefill;
its diagnostic reports requested and restored offsets. The future key path must
still match the view's original prefix, including image pseudo-token runs.

Request memory telemetry reports the additional transient helpers' count and
whole-state array bytes, deduplicated by capture offset. A planned checkpoint
at the same offset is already accounted among planned checkpoints.
#526 retains planned views' SSD intent until the turn's quiescent point after
leaf check-in. The ordinary write-eagerness policy and type-protected cut run
before buffer allocation; deferred views retain intent and can earn a write
through later hits. Extraction claims the current backer while the Model
Session copies and evaluates all payload arrays. The claim ends before enqueue;
payload ownership is recorded separately from its full segment format. Async
completion revalidates the view identity, and unbacked views self-heal when their
last ref or chain-prefix point is lost. A committed view without a backer uses
the existing SSD hydration rung to become an owned full body.
Extraction reserves one view at a time and releases that reservation on every
non-enqueued exit; replacement views cannot inherit the old body's attempt.
Only hit-earned promotions receive deferred writer scheduling; pressure-triggered
checkpoint writes retain the ordinary write-through class.
#527 implements opt-in Warm Bodies, compression-first drain, copy restore and
telemetry. #529 implements the Hot Leaf Set and opportunistic check-in pass.
#530 lets stored and transient views resolve through Warm Bodies. The packed
attention rows are sliced to the view offset before dequantization into private
live buffers; the view's whole-state layers retain their own checkpoint state.
Lookup telemetry keeps `source=view` and `copyReason=checkpoint`, and adds
`backingLeafForm=warm|ownedBody`. Existing quantized-KV partitions continue to
restore their owned quantized form without the Warm Body conversion.

For a warm-backed view's SSD admission, metadata-only pricing uses the full
live dtype's byte count, and extraction moves the already private materialized
buffers into a full-form payload. It neither dequantizes the entire descendant
nor copies the resulting prefix a second time. SSD Stored Form (#531) and the
#528 loaded gate remain pending; Warm Bodies stay opt-in.

#528's [pre-registration](../../benchmarks/warm-body-parity/2026-09-19/README.md)
fixes the fidelity, paired TTFT and memory reporting rules before any loaded
run. Status: **run on 2026-09-20 and failed** — see the
[owner run](../../benchmarks/warm-body-parity/2026-09-20/README.md). The
three-arm loaded runner (`--warm-parity-bench`) drives the fp16 copy control,
warm-8 and warm-4 arms through the in-process Server Completion path on
Qwen3.8-27B (4-bit, speculation off, greedy) over four cases (direct 4k and
32k, planned view 16k, think-stripping boundary 8k), six pre-registered arm
orders each. Fidelity had zero mismatches and the intended restore path held
in every observation, but the greedy 32-token continuation restored from a
Warm Body differs from the fp16 control in every block of three of the four
cases for both bit widths, deterministically, at the same token position; the
paired TTFT excess also exceeded the measured dequantization allowance for
warm-8 in two cases. Warm-4 passed only the planned-view case. Memory stayed
inside the manifest's 48 GiB bounds throughout. **`warmCompressionEnabled`
stays off**; the production conversion remains 8-bit; the 4-bit setting is
measurement-only. #531 (SSD Stored Form) does not proceed on this evidence.
The tokenizer-only corpus gate is not warm-restore evidence and did not see
the divergence.

The opt-in drain queues one Model Session batch and rechecks body identity and
Budget Floor membership when committing each conversion on MainActor. The
existing Leaf Lease read exclusion protects a moved body's arrays during the
conversion. Admission never awaits its own session queue; model teardown awaits
the pending drain. Compression preserves recency and reconciles quantized byte
accounting through the existing tree body transition. Until #531, an unbacked
Warm Body selected for Snapshot Demotion is restored and captured in the Model
Session so the SSD writer still receives the normal full form. Adaptive Write
Eagerness does not promote Warm Bodies in this slice.

Loaded-model parity and performance evidence remain owner work. Validation
uses the toy Model Session and tiny snapshot fixtures. An app-host bootstrap
that unexpectedly prewarmed Whisper during early tests was discovered and
guarded with the existing test-host detector before further validation.

## Amendment 2026-09-19 — a view's reuse credits its Backing Leaf

A Prefix-View Checkpoint has no bytes to score and is never an eviction
victim, so before this amendment its reuse was invisible to eviction: the hit
bumped the view node, while the Backing Leaf that served the restore kept its
older recency and a terminal Recovery Cost bounded by the view's offset. Under
pressure the tree dropped that leaf like any cold leaf, the view emptied and
self-healed, and a fork that had already proven reuse prefilled cold. The
loaded-model e2e gate's branch-point survival check caught this.

Two rules now make the view's value count where it lives:

- **Hit credit.** A hit served through a view, stored or transient, credits the
  chosen Backing Leaf exactly as a direct hit would: `lastAccessTime` and
  `hitCount`. Eviction recency and Adaptive Write Eagerness see the reuse.
- **Recovery span.** A leaf's terminal Recovery Cost spans from the nearest
  ancestor holding restorable state. A body-less junction (a split left by a
  sibling path or a released body) restores nothing and is skipped, and so is
  a view the leaf alone keeps alive when that view holds neither a Snapshot
  Ref nor a chain-prefix point: dropping the leaf loses the view's prefix too.
  Anything that lets the view outlive the drop bounds the span at the view:
  another resident full body, a leased leaf whose return preserves the
  checkpoint, a committed ref or a chain-prefix point.

Self-heal is unchanged. A genuinely tight budget can still take a view's last
backer; the amendment only makes that leaf as expensive to lose as it is.

## Amendment 2026-09-19 — the boundary backing leaf is released

#525 consumes a think-stripping turn's transient boundary views by checking in
the live leaf (the fed path plus the raw generated tail) as their Backing Leaf
before the boundary path restores at the view and re-prefills the canonical
residual. That live leaf was then left resident beside the canonical leaf: two
bodies of leaf size per boundary turn, where the pre-#525 tree kept one and the
transient full copies it replaced lived only for the turn. The loaded-model e2e
gate showed the doubled eviction rate.

Once the canonical leaf is admitted it is a descendant of every boundary the
turn resolved, so it backs those views itself. The Leaf Store then releases the
live leaf: an exact-path, RAM-only, unleased leaf body that is not the canonical
leaf is dropped through the tree's ordinary body drop and reported as a
`leafSupersession` with the new mode `released` (no SSD backing existed). When
the canonical store fails, or the body is leased or already gone, the live leaf
stays and the Leaf Store logs a `boundaryBackingLeafRelease` skip. The Speculative Canonical
Prefill resolves its boundary against whatever backer is resident when it runs.

