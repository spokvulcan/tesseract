# ADR-0062: Live Leaf Capture — the finished turn's leaf comes from the live cache when the fed path proves canonical

- Status: Accepted
- Date: 2026-09-06
- Relates to: ADR-0009 (speculative canonical prefill; its rejected
  alternative "capture from the final cache" is adopted here for the
  proven-equal case), ADR-0033 (`LeafStorePhase`), ADR-0056/0059
  (speculative decoding on the keyed path), ADR-0019 (leaf home guarantee)

## Context

After the model emits its stop token the server stores the turn's leaf so
the next request hits it. The two structured modes (`canonicalUserLeaf`,
`directToolLeaf`) did this by restoring a boundary snapshot taken
mid-prefill and re-prefilling the canonical re-render of the turn from
there: every generated token again, at prefill speed, about 7.3 ms per
token on Qwen3.8-27B (M3 Max). The client sees that as dead time between
the last delta and the finish chunk: 2.3 s on a 267-token answer, 47 s on
a 6.4k-token one, 120 s when the boundary was the last user message of a
cold 40k-token session. ADR-0009 had rejected capturing the leaf from the
final cache because, under a think-stripping template, the on-device KV
state is the think-bearing render and no trim converts it (GDN/Mamba
recurrent state cannot be rewound).

That objection does not hold on an append-stable render. Under the
**Preserve-Thinking Render** (Qwen3.8's default) and inside every tool
stretch of the Qwen3-family templates, the canonical re-render of the
finished turn is the prompt path plus the emitted ids plus a couple of glue
tokens, so the boundary re-prefill recomputes a KV state the live cache
already holds. The product rule that followed: with preserved thinking
there must be no re-prefill at all.

## Decision

The Leaf Store phase decides per turn, by comparison rather than by flag,
whether the leaf can be captured from the live decode cache
(`LiveLeafCapture.decide`, pure and GPU-free):

1. The decode loop records every id it feeds (`GeneratedTokenRecorder`),
   stop token included.
2. After generation the structured modes compute the canonical stored path
   as before (the **Leaf Admission Builder**'s probe). The live path is the
   request's **Cache Key Path** plus the recorded ids; the live cache's own
   reported offset bounds the capture (an unfed DFlash2 bonus token has no
   cache entry yet).
3. If the live path up to that offset equals the stored path's prefix, the
   phase captures the live final cache at that offset and admits it under
   the stored path. Zero prefill; the cost is the snapshot copy plus
   admission. The decision runs on the probe result before any restore
   boundary is looked up, so a live turn never pays **Snapshot Resolution**
   (which can hydrate from SSD) and needs no boundary at all: turns that
   used to skip for lack of one now store a live leaf.
4. Any mismatch falls back to the boundary plan and the unchanged
   restore-and-re-prefill. Eligibility misses (intervened turn,
   non-identity key space with image placeholders, no fed ids) log at info.
   A real disagreement (divergence, live longer than stored) logs a warning
   with the offset, both ids and four ids of context on each side, because
   on an append-stable render it means the render or the wire text is
   wrong; under the strip-by-default canonical render, which drops the
   emitted thinking by design, the same two log at info. A cache offset
   outside the live path always warns. That warning is how the vendored
   `ToolCallProcessor`'s dropped-prefix bug surfaced (mlx-swift-lm #609).
5. The speculative seed takes the canonical leaf offset from the live
   decision, so the ADR-0009 pass extends the same leaf.

Every post-generation stage is timed and reported in one `leafStore`
diagnostics event (render, plan, restore, prefill, capture, payload, admit,
the whole phase, and the span from generation end to the drive's finish —
the client's wait for its terminal chunk). The drive emits it at notice
level, so it survives in `log show`; the phase and tail seconds also reach
the trace corpus as optional fields.

## Consequences

- Client-visible tails on the same replays: 2.34 → 0.10 s (4.5k prompt,
  stop turn), 0.54 → 0.06 s (its follow-up), 0.39 s for a 60k-token stop
  turn. A 52k-prompt, 7.2k-token tool turn still takes 4.7 s: 3.0 s is the
  SSD full-payload extraction and 1.2 s the snapshot deep copy. Moving the
  payload write off the critical path is the follow-up.
- Correctness never rests on the render flag. A template that claims
  append-stability but re-renders differently falls back, and the warning
  names the first differing token.
- The comparison is one CPU pass over the fed ids, negligible next to the
  snapshot; the recorder adds one array append per generated token.
- Think-stripping templates still diverge at the strip point on a
  canonical-user boundary and keep paying the re-prefill; ADR-0009's
  background pass remains their answer. Their tool stretches are
  live-eligible.

## Rejected alternatives

- Trusting the Preserve-Thinking flag and skipping the comparison. It would
  have stored a mangled leaf for every turn hit by the `ToolCallProcessor`
  bug without anyone noticing; the comparison is what made it visible.
- Trimming the live cache to the stored length when the live path is
  longer. Recurrent state cannot be trimmed (ADR-0009), so a longer live
  path has no valid state at the stored end.
- Comparing rendered text instead of token ids. A BPE re-split of identical
  text passes a text check and misses the cache key; the leaf keys on ids.
