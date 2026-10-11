# ADR-0081: Speech never waits for the LLM — the GPU lease becomes the LLM gate

- Status: Accepted
- Date: 2026-10-01
- Amends: ADR-0034 (the Proofread Pass's skip reads "the LLM is generating"),
  ADR-0038 (the speech engine's `GPULeasing` port and per-burst lease are gone),
  ADR-0039 (load, warm-up and priming no longer wait for a lease), and the "GPU
  lease" wording of ADR-0015, ADR-0022, ADR-0064 and ADR-0069, which now means
  the LLM gate
- Relates to: ADR-0080 (the Companion's moments take the gate like any chat),
  ADR-0009 (speech and the speculative prefill already shared the GPU)
- Amended: 2026-10-11 (Model Residency: one module for what is loaded and for
  Offload Model, which now reaches every model that can be freed)

## Context

Every MLX consumer — LLM generations, the voice engine's per-segment bursts,
loads and warm-ups — took one FIFO **GPU lease**. On the owner's first full day
with the Companion (2026-10-01) the voice stalled whenever LLM work arrived:

- The voice took the lease once per segment and released it between segments,
  so any LLM work queued meanwhile ran between two sentences and the voice
  stopped until it finished. A Triage held the GPU 5–27 s, the Morning Plan
  163 s (34 s of prompt, 106 s of output, 19.5 s saving its cache), a chat turn
  with tool rounds up to 74 s; every call kept the GPU up to 20 s after its
  answer while it saved its cache.
- A spoken reply started only after the whole turn, and read-aloud during a
  generation waited for all of it.

The owner's view: memory, not GPU time, is the constraint. Two separate
questions follow — is concurrent MLX evaluation safe, and what else did the
lease guarantee?

- **Safety.** The pinned mlx-swift (0.31.4-28, core v0.31.1) wraps every
  `eval`/`asyncEval`, `MLXArray.eval()`, `Stream.synchronize`,
  `Memory.clearCache`, compile and IO in one process-wide `NSRecursiveLock`
  (`evalLock`), and the allocator has its own mutex. Evaluation from different
  tasks on the shared default stream is therefore safe as long as consumers
  share no arrays — and each model has its own container. MLX already ran
  beside the LLM before this change: the proofreader after its busy check, the
  embedder, the speculative prefill beside speech (ADR-0009), the Neural Engine
  codec build, a cancelled speech segment's tail frame. None of the recorded
  crashes came from two models running at once.
- **What else the lease did.** One LLM generation at a time (the prefix cache's
  Cache Claim, Leaf Handoff, Pending-Payload Wait and the server's single
  active-generation slot all assume it), the loaded model never changing under
  a running consumer, HTTP requests waiting FIFO with a 60 s timeout, and
  Offload Model waiting for the running generation. All of these are about the
  LLM, none about speech.

## Decision

1. **The GPU Lease Queue becomes the LLM Gate.** Only LLM work takes it: chat
   turns, the Companion's moments (their compaction included), HTTP requests,
   `/compact`, reloads and offload. `InferenceArbiter.withLLM` takes the gate,
   makes sure the selected model is loaded, and runs the body — so the model
   still cannot change under a generation, and generations still run one at a
   time, FIFO.
2. **Speech never waits.** The speech package loses its `GPULeasing` port: loads,
   warm-ups, voice priming and every segment run when asked. The engine actor
   and the model's own generation lock still order the engine's work.
3. **Dictation never waits either.** The Proofread Pass keeps skip-when-busy: it
   reads whether the LLM gate is held and commits the cleaned raw text rather
   than share the GPU with a long generation.
4. **Offload Model** releases the voice at once and the LLM after the running
   generation.
5. **Memory is what the models share, so the owner sees it.** The menu bar's
   Models section lists every model — language model, its speed-up draft, voice,
   dictation, proofreader, memory search — loaded or not, what it is doing right
   now ("Jarvis · Triage", "Speaking", "Listening"), its size, and the app's
   footprint with the system's swap. No submenu; it refreshes while the menu is
   open.
6. **The MLX buffer pool is capped at 512 MB** (it was 2 GB, and a long prefill
   filled it every turn). Decode reuses small buffers, which fit; large prefill
   buffers go back to the system.

## Consequences

- Speech interleaves with a generation instead of waiting for it. MLX's lock is
  held through each GPU wait, so consumers take turns one evaluation at a time:
  a speech frame can wait behind one prefill chunk (1,024 tokens, about 5 s on a
  27B model), while decode steps interleave finely. The voice keeps about 8 s
  of audio ahead (its lookahead), so playback rides through a chunk; the LLM
  decodes somewhat slower while the voice speaks (the voice needs about 15% of
  real time).
- Memory peaks can now coincide (a voice burst during an LLM prefill). The
  prefix cache's headroom does not count the voice; the Models section makes
  the footprint visible.
- `Memory.clearCache()` from either side flushes the shared buffer pool: a speed
  cost, not a correctness one.
- HTTP requests wait only behind other LLM work; the voice no longer counts
  against their 60 s.
- Not done: a stream per consumer for true GPU overlap. Measure first; MLX's
  thread-local streams arrive with 0.32, which is blocked (#513).

## As built

- `Features/Agent/LLMGate.swift` — the gate (was `GPULeaseQueue.swift`);
  `LLMGateTests`.
- `Features/Agent/InferenceArbiter.swift` — `withLLM`, `isLLMBusy`, offload;
  `InferenceArbitrating.swift` — the one-member seam (`withLLM`).
- `Vendor/tesseract-speech` — `SpeechEngine` without the `GPULeasing` port;
  `Features/Speech/ArbiterGPULease.swift` deleted.
- `Features/Dictation/Proofread/ProofreadPass.swift` — `isLLMBusy`.
- `Platform/MenuModelsSection.swift`, `App/DependencyContainer+ModelActivity.swift`
  — the Models section.
- `Features/Agent/LLMActor.swift` — `cacheLimitMB = 512`.

## Amendment (2026-10-11): Model Residency

Decisions 4 and 5 stand; where they are built moved, and Offload reaches
further. The menu's Models section and the Models page used to derive "which
models are loaded" twice, with different reads of the language model's state;
the arbiter mirrored residency for two models (`.llm`, `.tts`) while Offload
freed three; the recall embedder was never unloaded, and its loaded flag sat on
the composition root.

- **Model Residency** (`tesseract/App/ModelResidency.swift`) is the one module
  that knows which models are in memory. Each model's holder reports to it as
  one row (`tesseract/App/DependencyContainer+ModelResidency.swift`): the
  model, whether it is loaded and what it is doing, and how to free it. The
  Models section (decision 5), the Models page's in-memory mark and Offload
  Model all read it.
- **Offload Model** frees every model a holder lets go of, at once: the voice,
  the proofreader and the recall embedder right away, the language model after
  the running generation (decision 4: `InferenceArbiter.offloadLLM` waits its
  turn at the gate). Dictation's Whisper stays, as before (always-armed
  dictation, ADR-0025). Unloading the embedder is safe: an embed already
  running keeps its own reference to the model, and the next recall loads it
  again (0.4 s). Clear Disk Cache still frees the voice and the language
  model before it wipes.
- The arbiter no longer mirrors residency: `ModelSlot`, `loadedSlots` and the
  voice closures are gone; it is the language model's holder.
- The loaders that need a complete model find its files through one answer,
  `ModelDownloadManager.usableDirectory(for:)`: the proofreader, the recall
  embedder (now `RecallEmbedder`, which keeps its own loaded mirror), the
  arbiter's load and the vision probe. Whisper keeps its own completeness
  contract, and the voice engine reads the folder of the spec it shares with
  its catalog entry. It lives with the download manager rather than Model
  Residency: the download manager already decides where a model's files go and
  whether they are complete, and the iPhone app shares it.

Behaviour changes: Offload frees the recall embedder, and frees the
proofreader without waiting for the language model's generation; the Offload
item is enabled while any model it can free is loaded, the proofreader or the
embedder alone included. Tests: `ModelResidencyTests` (replacing
`LoadedModelsTests`).
