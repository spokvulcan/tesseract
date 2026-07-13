# ADR-0039: TTS residency — lazy load, keep warm, and no TTS cache limit at all

- Status: Accepted (under overnight delegation, 2026-07-13 — owner review pending; idle-warm number to be validated by measurement)
- Date: 2026-07-13
- Relates to: map #334, ticket #344; ADR-0037 (one checkpoint — tier switching vanished), ADR-0038 (boundary: `prepare(Readiness)`, deterministic `unload`, `MemoryPolicy` seam); #338 (budgets), #339 (benchmarks, envelope-definition flag), autopsy M1–M6/D2/F1/F2/F4

## Context

ADR-0037 collapsed residency to one question: when is the single quantized 1.7B-VoiceDesign checkpoint loaded, warm, and unloaded — and how does it coexist with the LLM stack. The v1 defects to design away: lazy load *inside* the first request's lease with no warmup (F1/F2), voice-prefix prefill on the hot path (F4), the 100 MB process-global cache-limit clobber never restored (M1), five scattered clear-cache sites tuned by env var (M2/D2), anchor/prefix retained forever (M3), and "unload = drop the reference" (M6).

Also to settle: #339's flagged definition question — the #338 envelope (≤ 3 GB peak / ≤ 2.5 GB idle-warm) never said *which* memory number it means. Measured on 8-bit: MLX active-memory peak 5.43 GB vs process peak RSS 3.24 GB.

The LLM stack's own conventions are decisive context (read this session): `LLMActor.Defaults.cacheLimitMB = 2048` re-asserted at three entry points precisely because v1 TTS clobbers it (`LLMActor.swift:206,320,362`); the codebase already documents "that knob belongs to the agent's `LLMActor`" (`MemoryEmbedder.swift:16-17`); the arbiter's FIFO lease serializes all GPU compute (`InferenceArbiter.swift:122-137`).

## Decision

**1. The envelope metric is process peak RSS, TTS-attributable, validated in the TTS-only bench harness.** Exactly how #339 measured — the scorecard stays valid, and RSS is what a 16 GB floor machine actually feels. MLX active-memory is *rejected as the metric* (it over-counts transient allocator hold by ~2 GB on 8-bit) but kept as a diagnostic the bench reports. The ADR-0037 precision gate reads this metric.

**2. Load lazily on first use; keep warm thereafter.** No launch-time preload: the measured cold path (load + first audio ≈ 1.8–1.9 s) already passes both cold budgets (≤ 2 s fast / ≤ 4 s quality), so first-press-of-the-day pays a budgeted cost instead of every launch paying ~2+ GB of resident weights for a feature possibly unused that day. Two refinements:
   - **Warmup runs immediately after load, inside the same lease** (kernel/JIT + fused-weight eval + tokenizer materialization + the configured voice's prefix prefill) — so the *second* request hits warm TTFA (≤ 300 ms measured). F1/F2/F4 die here.
   - **The companion arms eagerly by construction**: opening its long-lived `SpeechSession` at conversation start performs load + warmup + prefix prefill off the utterance hot path (ADR-0038's `session()` contract).
   - Voice-description settings changes re-prime the prefix off the hot path *if loaded*; they never trigger a load.

**3. Keep-warm policy: resident until told otherwise; one cache clear per utterance end.** No idle TTL — rejected: the ≤ 2.5 GB idle-warm budget exists precisely to permit keep-warm, reload costs ~2 s of user-visible latency, and the LLM stack doesn't TTL either. Idle-warm state = weights + codec + bounded voice-prefix pool (limit 4) + the pinned anchor (a few KB of ingredients per ADR-0038). Session/utterance KV dies with its owner (M3 fix is structural, from ADR-0038). The single `Memory.clearCache()` at utterance end returns the buffer pool to its floor; the vendor's internal 50-step cadence stays as shipped upstream.

**4. The TTS engine never touches the process-global cache limit.** v1's 100 MB clobber (M1) existed to survive the windowed decode's transients — a defect the upstream stateful decoder removed (D1). TTS transient working set (a ≤ 200-token segment's KV + streaming-decoder state) fits comfortably inside the LLM's 2 GB pool cap, which MLX enforces globally regardless of who allocates; the arbiter's lease already serializes GPU compute so there is no concurrent contention. **Ownership convention, now stated once**: `Memory.cacheLimit` belongs to `LLMActor`, full stop; the engine's `MemoryPolicy` carries `respectsProcessCacheLimit = true` and the pinning test asserts the limit is bit-identical before and after any TTS burst. This is *simpler* than scoped set/restore — there is nothing to restore because nothing is set.

**5. Unload is deterministic and app-triggered.** `prepare(.unloaded)` / `unload()` cancels active utterances, releases weights + KV + pools, clears MLX caches, and synchronizes the GPU stream before returning (M6 fix, engine-owned — the `f70a3ff9` termination convention moves inside). Triggers, owned by the app layer (the engine must not know the app): TTS features disabled in settings; macOS memory-pressure *critical* (the app's pressure source calls `unload()`); app termination. Sessions survive unload as ingredient values and rebuild KV transparently on next use (ADR-0038).

**6. Coexistence arithmetic (16 GB floor, restated from #338 with v2 numbers):** LLM working set ~5.5 GB + TTS ≤ 3 GB peak (generation) / ≤ 2.5 GB idle-warm, GPU compute serialized by the lease, LLM prefix cache auto-sized with its existing 20 GiB headroom formula unchanged. TTS generation bursts (~4–6 s GPU per ~16 s audio segment) interleave with LLM turns at segment boundaries per ADR-0038.

## Considered / rejected

- **MLX active-memory as the envelope metric**: over-counts transients ~2×; would fail every candidate including the ones that fit by RSS; kept as diagnostic only.
- **Launch-time preload**: pays resident weights on every launch; cold budgets are met without it. Revisit only if a measured first-press complaint appears (then it's a settings toggle, not a policy rewrite).
- **Idle-TTL unload**: complexity without a budget justification; the envelope permits keep-warm.
- **Scoped TTS cache-limit set/restore**: strictly dominated by not setting one — the shared-pool cap self-regulates and the LLM's re-assertions become belt-and-suspenders instead of load-bearing defenses.
- **Engine-subscribed memory-pressure handling**: violates "engine must not know the app"; the app owns the trigger, the engine owns the mechanism.

## Consequences

- `QWEN3TTS_CACHE_LIMIT_MB` and the other six env vars die with no replacement knobs beyond the typed `MemoryPolicy`/`EngineTuning` defaults.
- `LLMActor`'s three defensive cache-limit re-assertions become redundant (kept — they're cheap and guard third-party regressions) once the pinning test proves TTS never moves the limit.
- The bench harness gains an idle-warm measurement mode (post-utterance RSS after the utterance-end clear) to validate the ≤ 2.5 GB number and feed the ADR-0037 precision gate.
- The morning listen's precision gate reads: 8-bit long-form peak RSS ≤ 3 GB → ship 8-bit; else 6-bit (2.84–2.86 GB measured, fits today).

## Accepted costs

- First press of the day pays ~1.9 s cold start (budgeted, measured).
- Keep-warm holds ~2+ GB resident indefinitely after first use — exactly what the #338 idle-warm budget licenses.
- Relying on the LLM's 2 GB pool cap couples TTS transient behavior to an LLM-owned constant; acceptable because the lease serializes use and the pinning test would catch a future TTS path that inflates the pool beyond it.
