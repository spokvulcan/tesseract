---
status: accepted
---

# The server spends idle GPU after a finished stretch on speculative canonical prefill

CONTEXT.md → **Speculative Canonical Prefill**, **Think-Strip Rewind**,
**Stretch Abandonment**; issues #76, #100. GPU work no request asked for, run
after the response is delivered.

Under a thinking template the canonical user leaf only covers the render up to
the think-strip divergence; the **Tool Stretch** past it (assistant/tool turns
re-rendered without thinks) re-prefills *interactively* on the next user
message — 15,271 tokens / 17.1 s measured live, versus 0.2–0.5 s for a turn
that hits the leaf. That span is 99%+ precomputable: the future token path is
fixed by the stored conversation (a second divergent user probe pins the shared
prefix through the next user-turn header), and the only unknown — the user's
message — sits past the admitted leaf. So the server restores the just-admitted
leaf and extends it along that path in the background, admitting a deeper leaf
the next request hits directly. The GPU-vs-TTFT trade is decided for TTFT: the
GPU time is reclaimed from a window idle by construction, since the next event
after a finished stretch is human-paced.

## Triggers

`ServerCompletion.speculativeSeedPlan` maps a finished turn to a pass (nothing
seeds under the **Preserve-Thinking Render**, #98 — that render is
append-stable, so there is no rewind span):

- **Stop-finish** (canonical-user boundary): seed immediately, durable leaf
  by default. The original #76 trigger.
- **Tool-stretch finish** (**Stretch Abandonment**, #100): arm a timer; the
  pass starts only if no follow-up lands inside a 5 s idle window — short
  enough to keep most of a human pause as runway, long enough that an agent
  loop's next tool-result request (milliseconds-to-seconds) preempts the
  sleeping pass first. The spine admits **RAM-only**, so a false alarm (the
  tool result does arrive) costs zero SSD writes; the rewind landing later
  persists the branch via its own full write.
- **Client abort / disconnect** (#100): seed immediately from the request's
  *completed* messages — the half-generated turn never enters the path — also
  RAM-only, so a reconnect-and-continue wrote nothing to SSD.

## Guardrails

- **Interactive work always wins.** Every generation entry cancels-and-awaits
  the pass; the chunk loop checks cancellation between `container.perform` hops
  with a synchronous per-chunk `eval`, then settles (below), so a preempting
  request acquires the container actor within ~one chunk plus at most one
  RAM-only capture. Unload drains it the same way. A pass preempted while still
  awaiting its probe forwards the cancellation explicitly (`Task.value` is not a
  cancellation point for the waiter); the probe body is synchronous, so its
  cooperative checks end the wait within one render.
- **Preempted progress is kept, not recomputed.** A preempted pass admits its
  progress as a **RAM-only** leaf when it has prefilled ≥ 2,048 tokens (below
  that, capture rivals the re-prefill it saves). RAM-only fits: the preempting
  request supersedes it with its own SSD-backed leaf moments later. That request
  *awaits* the settle — fire-and-forget would let its lookup race past the
  admission and re-prefill the span just computed. Reverses the first revision's
  "strictly droppable" stance, after a live session preempted at 6,144 of
  15,896 tokens and re-prefilled all of it interactively.
- **Worth-it threshold.** Spans under 512 residual tokens are skipped — the
  interactive re-prefill is already sub-second.
- **Safety margin.** The admitted path stops 2 tokens short of the probe LCP:
  its final tokens sit at the template→user-content seam where BPE can merge
  across the boundary, and since the deeper leaf supersedes the canonical one
  (no shallower fallback), overshooting would orphan the leaf.
- **The window starts early.** The future-path probe runs on the CPU
  concurrently with the canonical leaf's GPU-side store, spawned before it, so
  the pass spends none of its human-paced window on its own stage 1.

## Experimental Asymmetric-State Restore

Issue #134 adds an opt-in, disabled-by-default experimental body for the
stop-finish trigger: **Asymmetric-State Restore**. Instead of restoring the
canonical leaf and re-prefilling the whole think-stripped stretch, the pass can
capture the think-bearing final cache, excise from sliceable attention layers
the token runs the future render drops (**Render-Diff Excision**: spans come
from aligning the bearing tokens against the pass's actual admit path, never
from scanning for literal `<think>` delimiters, which conversation content can
carry as data), re-rotate retained keys to their shifted positions, and leave
non-sliceable recurrent state at the bearing render. The result is a synthetic
stripped-path boundary; the ordinary speculative tail then only prefills the
small future-user-header residual. When alignment ends at a re-tokenized seam,
synthesis proceeds at the shallower aligned depth (partial synthesis) rather
than declining outright.

This is deliberately not the default path. The recurrent state is stale by
construction, so correctness is measured, not assumed: the unit suite pins the
array surgery and the loaded-model `HybridCacheCorrectnessRunner` reports
KL/top-k/greedy divergence against a gold full re-prefill. If preflight cannot
prove the render/cache offset contract, the pass falls back to ordinary
speculative prefill; if synthesis fails after surgery starts, it admits nothing
deeper than the canonical leaf. A debug **test mode** setting drops the pass's
worth-it floor to one token and logs first-divergence forensics on declines, so
the path is exercisable at any context or reasoning length.

Storage is decided after the pass knows its outcome: an ASR-derived admission
(a synthesized boundary, or anything extended on one) stays **RAM-only**, while
a declined-ASR fallback keeps the ordinary durable RAM+SSD admission. A
synthesized snapshot is built from non-contiguous token-axis pieces and does
not fit ADR-0010's contiguous segment-chain model; persisting it as an SSD
extension would chain stripped-path K/V onto a bearing-path base.

## Rejected alternatives

- *Capturing the think-stripped leaf directly from the turn's final cache* —
  the on-device KV state is the think-*bearing* render; a plain trim does not
  convert it. Asymmetric-State Restore is the later experimental exception: it
  performs explicit attention-layer excision plus delta-RoPE, leaves recurrent
  state stale, stays off by default, and reports distributional drift rather
  than claiming correctness.
- *Running the pass under the GPU lease* — would queue TTS behind background
  work and risk a reload from `ensureLoaded`; the registry drain covers the
  reload hazard and TTS lives in a different container, so kernel timesharing
  is the worst case.
- *Dropping all progress on preemption* (the first revision) — the bounded
  settle wait buys back multiples of itself, and the 2,048-token floor keeps
  the trade a guaranteed win.

## Consequences

- Post-answer energy rises on tool-heavy sessions (bounded by span length,
  observable via the `speculativePrefill` diagnostics events, which carry a
  `preempted` flag).
- When Asymmetric-State Restore is enabled, `asymmetricStateRestore`
  diagnostics report synthesized/unavailable/mid-synthesis outcomes plus
  bearing-capture and synthesis timings.
- Each completed pass supersedes the canonical leaf, shortening its SSD
  lifetime — the churn ADR-0010 later mostly closes.
- Preempting entries wait out the settle even when targeting a different
  conversation.
- The speculative leaf is admitted without an `AlphaTuner` request record —
  the tuner models requests, not background passes.
