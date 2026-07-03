---
status: accepted
---

# The Model Session seam sits under Server Completion; the test peer is a toy model running the real verbs

This records the design decided in the 2026-07-03 architecture-review grilling
(PRD: #137; vocabulary in `CONTEXT.md` → **Model Session**). ADR-0015 rejected
"seal the module behind a port" with an explicit reopening condition — *"revisit
only when a genuine in-memory peer would exercise real behaviour at a specific
MLX edge"* — and this decision is that condition being met, not a reversal.

The **Server Completion**'s ~1,450 lines of resolve → restore → prefill → drive
→ capture → admit sequencing are testable only behind a loaded model; its tests
cover the extracted pure statics while the ordering they are sequenced in — where
bugs like #136 live — has no surface, and `LLMActor` carries 11 `ForTesting`
shims plus ~15 admin forwards that exist only to reach around the missing seam.
The decision: a **Model Session** port at the model verbs the module already
performs (prepare, cache creation, restore, chunked prefill, decode-iterator
construction, KV quantization, snapshot capture, the vision-continuation query).
The port's single entry is a scoped session mirroring `container.perform`; verbs
are synchronous inside it, so one session = one Metal-affine batch and the
"decide before entering" discipline survives verbatim. MLX value types
(`MLXArray`, `[any KVCache]`) stay in the port vocabulary. The production
adapter wraps the container and is the only place inside the module that touches
it directly; ADR-0015's actor confinement and off-actor drive are unchanged.

The second adapter — the one that makes the seam real — is a toy
`LanguageModel` in the test target: a `Module` subclass with scripted logits, a
cache-dimension provider, and deterministic sampling (the vendored MLXLMCommon
protocol is in-tree, so conformance is ours). The in-memory session executes the
**real** implementations behind the verbs — genuine `newCache`, the real
`PrefillExecutor`, a real `StateThreadedTokenIterator` whose init runs its
genuine prime forward — over microscopic tensors. Sequencing tests are therefore
integration-flavoured by design: real MLX ops, hermetic, no model download.

## Considered / rejected

- **Wrap `ModelContainer`/`perform` wholesale.** No — the interface would be as
  wide as MLX itself, and the stand-in would have to fake `context.model`, which
  is precisely the thing that cannot be faked shallowly.
- **Per-verb `async` port methods.** No — each verb becomes its own executor
  hop, silently rewriting the concurrency structure the module was carved
  around (per-token sink work off-actor; `MLXArray.asData()` on the Metal
  thread inside one perform block).
- **Phase-level verbs (`runPrefill(plan)`, `captureLeaf(…)`).** No — the
  iterator-construction and quantize wiring would migrate into the production
  adapter where it is untestable again, and the migration stops being a
  strangler move. Revisit *inside* the seam only if verbs cluster in practice.
- **A canned-verb test peer (no MLX model).** No — the port would have to
  abstract the decode iterator (its init runs a real forward pass), and the
  peer would be a second implementation of decode semantics that drifts from
  the real one. The toy keeps the second adapter thin: only the model is
  substituted, which is exactly what varies.
- **Dissolving the admin wall by handing callers the `PrefixCacheManager`.**
  No — the manager is rebuilt per model load, so a bare handle goes stale; a
  MainActor current-cache accessor owns "which manager is live" (deletion test:
  that knowledge otherwise reappears at every caller). Drain and
  speculative-prefill scheduling stay genuine `LLMActor` lifecycle members.
