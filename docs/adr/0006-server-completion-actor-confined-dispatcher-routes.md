---
status: accepted
---

# Server Completion is an actor-confined module; the dispatcher owns the route; the lease is the primary guard

This records the design decided in the 2026-06-09 architecture-review grilling for
the **Server Completion** deepening (see `CONTEXT.md` → **Server completion**).
The modules named below land with that refactor (PRD: #61); this ADR exists so
the shape is not re-litigated.

`LLMActor` (2,147 lines) wears two interfaces: the agent generation spine
(~400 lines) and the HTTP server's cache-aware completion execution (~1,450 lines —
`generateServerTextCompletion`, `makeHTTPPrefixCacheGeneration`, leaf capture, the
snapshot-payload extraction statics). The decision side of that path is already
deep and tested (**Prefill Planner**, **Leaf Admission Builder**); the execution
side they feed is carved into **Server Completion** — a non-`Sendable`,
actor-confined module stored in `LLMActor`, installed at model load (prefix cache,
SSD config snapshot, load-time identity facts) and drained-then-cleared at unload.
The 16-field `HTTPPrefixCacheGeneration` bundle becomes the module's private
value. Files live in `Features/Server/`; `LLMActor` lifts into
`DependencyContainer` and is injected into both `AgentEngine` (chat path,
lifecycle) and `ServerInferenceService` (cache-aware route).

`ServerInferenceService` becomes the **dispatcher**: it owns the **Completion
Route**, the pure request-shape decision (empty conversation /
last-message-assistant / unshaped prefix-cache conversation) that today lives as
`nil`-returns inside the actor plus fallback branching in `AgentEngine`. The
route gets model-free unit tests; the **Server Completion** module never sees a
request it cannot serve.

`AgentEngine` sheds the tunnel: its ~10 server pass-throughs (sole production
consumer: one `PromptCacheTelemetryStore` call), the `startManagedHTTPGeneration`
MainActor re-pump (a second stream+task around a handle the actor already
composes), and the HTTP flip of `AgentEngine.isGenerating`. The safety evidence,
traced before deciding: the GPU lease is held across the **whole** HTTP request
(`CompletionHandler` wraps acquisition-through-stream-end in
`withExclusiveGPU(.llm)`), so arbiter-mediated unload/reload cannot interleave;
`AppTerminationCoordinator` drains the HTTP server **before** any engine-level
cancel (ordering pinned by tests); engine-level `isGenerating` has zero external
readers — every UI read is the **Agent Run**'s flag. The registry duty moves into
the actor and tightens: `LLMActor.unloadModel` cancels-and-awaits the module's
active completion before releasing the container, replacing the engine's
fire-and-forget `cancelGeneration()`.

## Considered / rejected

An architecture review that re-suggests any of these should treat them as
already-decided:

- **"`ServerInferenceService` is a ~90-line pass-through — dissolve it."** True
  before this carve (the same review's report said exactly that); false after.
  The dispatcher composes two adapters at the seam and owns the **Completion
  Route**; deleting it scatters routing into the 1,086-line `CompletionHandler`.
  The deletion test flips with the second adapter.
- **"Make Server Completion its own actor."** No. It would push `@unchecked
  Sendable` values (`ModelContainer`, `HybridCacheSnapshot`) across a new actor
  hop, split unload sequencing across two actors, and buy no isolation — the GPU
  lease already serializes consumers. This is a module split, not an isolation
  split: module state stays confined to `LLMActor`, and the driving loop runs
  off-actor with its model-affine steps inside `container.perform`, exactly as
  the pre-carve driving task did — no new hop, no second unload sequence.
- **"Keep the `nil`-fallback in the actor."** No. The bypass checks are pure
  request-shape facts; inside the actor they are untestable without a model and
  smear the route across two modules.
- **"Keep the MainActor re-pump / a shared busy flag for unload safety."** No —
  rejected on traced evidence, not taste (lease spans the request; termination
  drains the server first; the flag has no readers). The actor-side registry is
  the backstop, not the primary guard.
- **"Seal the module behind a port so the sequencing is unit-testable."** Not
  now. One adapter today means a hypothetical seam (ADR-0001's rule); this carve
  buys locality first. Revisit only when a genuine in-memory peer would exercise
  real behaviour at a specific MLX edge.
