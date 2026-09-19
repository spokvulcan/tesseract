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

Planned non-system checkpoints and, in a later slice, transient Think-Strip
Rewind and Speculative Canonical Prefill boundaries own whole-state layers,
metadata and an absolute token offset. Capture still synchronizes prefill
before copying whole-state layers. System checkpoints keep owned full bodies:
they are small, reused by every conversation, and must survive a leased leaf.

Snapshot Resolution chooses a resident, unleased, full-body descendant in the
same partition as the Backing Leaf, nearest by offset and then most recent.
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
durability is exactly its SSD admission (the ADR-0019 amendment accompanies
#526). Without a resident or temporarily leased descendant, Snapshot Ref or
chain-prefix point, the view becomes empty and the usual topology self-heal
applies. A leased descendant preserves the checkpoint for return, but cannot
serve a restore during the lease.

In #526, end-of-turn SSD admission will slice the checked-in leaf at the
extraction edge and detach every retained array on the Metal thread. The
writer receives a full snapshot at the view offset, in the unchanged segment
format. Such a payload never blocks the leaf's next checkout. Adaptive Write
Eagerness and the type-protected SSD cut keep their existing meanings.

## Compressed Warm Tier

#527 introduces compression behind an Eviction Configuration flag, default
off until #528 passes. On drain, compression precedes Snapshot Demotion in
the existing eviction-selection order. Leased leaves, the Budget Floor's
most-recently-extended leaf, system bodies and already-quantized partitions
are exempt. #529 adds opportunistic compression above a configured fraction
of the ceiling, preserving the most recently checked-in leaves of up to two
paths by default, at the Model Session's next quiescent point.

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

#524 implements planned branch-point views in RAM in PR #539. #527 implements
opt-in Warm Bodies, the compression-first drain, copy restore and telemetry.
These independent slices carry this shared decision; neither enables the
Compressed Warm Tier by default. Transient views (#525), view SSD admission
(#526), the Hot Leaf Set (#529), warm-backed views (#530) and SSD Stored Form
(#531) remain separate slices.

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
uses the toy Model Session and tiny snapshot fixtures. The app's unit-test host
skips service bootstrap so unit tests never start model prewarms.
