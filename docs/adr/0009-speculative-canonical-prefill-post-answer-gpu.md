---
status: accepted
---

# The server spends idle GPU after stop-finish answers on speculative canonical prefill

This records a decision from the 2026-06-11/12 prompt-caching grilling (see
`CONTEXT.md` → **Speculative Canonical Prefill**, **Think-Strip Rewind**;
issue #76). It introduces something the server never did before: GPU work that
no request asked for, run after the response has already been delivered.

The trigger: with a thinking template, mid-loop caching is token-perfect, but
the canonical user leaf can only cover the render up to the think-strip
divergence — the probe LCP ends where the template will rewrite history once
the next real user message arrives. Everything past it (the whole tool loop's
assistant/tool turns, re-rendered without thinks) re-prefills *interactively*
on that next message: 15,271 tokens / 17.1 s measured live, versus 0.2–0.5 s
for turns that hit the canonical leaf. The spans are 99%+ precomputable — the
future token path is fully determined by the stored conversation (a second
divergent user probe pins the shared prefix through the next user-turn
header), and the only unknown, the user's actual message, sits past the
admitted leaf.

So: after a stop-finish turn that stored a canonical leaf, the server restores
that leaf and extends it along the future shared path in the background,
admitting a deeper leaf the next request hits directly. The GPU-vs-TTFT trade
is real and decided in TTFT's favor because the GPU time is reclaimed from a
window where it is otherwise idle by construction — the next event after a
final answer is human-paced.

Guardrails, in priority order:

- **Interactive work always wins.** Every generation entry (cache-aware start,
  standard raw path) cancels-and-awaits the pass; the chunk loop observes
  cancellation between `container.perform` hops with a synchronous per-chunk
  `eval`, then settles (below), so a preempting request acquires the container
  actor within ~one chunk plus at most one RAM-only capture. Unload drains it
  the same way.
- **Preempted progress is kept, not recomputed.** A preempted pass settles by
  admitting its partial progress as a leaf when it has prefilled at least
  2,048 tokens (below that, the capture would rival the re-prefill it saves —
  the progress is dropped and the cache is left exactly as the canonical leaf
  left it). The chunk loop keeps the warm cache chunk-aligned to the admit
  path, so the partial capture needs no trim. The partial leaf is **RAM-only**:
  its sole purpose is the imminent preempting request, which supersedes it
  with its own SSD-backed leaf moments later — skipping the SSD payload
  extraction keeps the settle short and the disk churn (#78) at zero. The
  preempting entry *awaits* the settle; fire-and-forget cancellation would let
  its lookup race past the partial admission and re-prefill the very span the
  pass just computed. This reverses the first revision's "strictly droppable"
  stance — the first live session preempted a pass at 6,144 of 15,896 tokens
  (39% of the span) and re-prefilled all of it interactively, exactly the
  deep-preemption telemetry the revisit clause asked for.
- **Worth-it threshold.** Spans under 512 residual tokens are skipped — the
  interactive re-prefill is already sub-second, not worth a GPU wake-up plus a
  leaf admission and its SSD write.
- **Safety margin.** The admitted path stops 2 tokens short of the probe LCP:
  its final tokens sit at the template→user-content seam where BPE can merge
  across the boundary, and since the deeper leaf supersedes the canonical one
  (no shallower fallback), overshooting would orphan the leaf entirely.
  Undershooting costs the next request a few header tokens.
- **The window starts early.** The future-path probe (two renders plus
  tokenizations) runs on the CPU concurrently with the canonical leaf's
  GPU-side store, spawned by the drive task before that store begins — the
  pass spends none of its human-paced window on its own stage 1. The window
  is short in the cases that hurt: the measured deep preemption had ~10
  seconds between answer and next question for an ~18-second span.

Considered and rejected: *capturing the full think-stripped leaf directly from
the turn's final cache* — the KV state on device corresponds to the
think-bearing render, not the stripped one; there is no trim that converts one
into the other (Mamba state cannot be rewound, and even attention-only trims
drift sampled decoding — see the normalization-trim skip). *Running the pass
under the GPU lease* — would make TTS queue behind background work and risk a
model reload from `ensureLoaded`; the registry drain already covers the
unload/reload hazard, and TTS lives in a different container, so kernel-level
timesharing is the worst case. *Dropping all progress on preemption* (the
first revision's choice, made to keep the preempting request's entry
zero-cost) — rejected on first-session telemetry; the bounded settle wait
buys back multiples of itself in skipped re-prefill, and the 2,048-token
floor keeps the trade a guaranteed win.

Consequences: post-answer energy use rises on tool-heavy sessions (bounded by
the span length, observable via the `speculativePrefill` diagnostics events,
which carry a `preempted` flag for partial admissions); each background pass
supersedes the canonical leaf, shortening its SSD lifetime (the churn pattern
tracked in #78); preempting entries wait out the settle (~one chunk plus at
most one RAM-only capture) even when they target a different conversation;
and the speculative leaf is admitted without an `AlphaTuner` request record —
the tuner models requests, not background passes.
