# VLM Prefill Perf — measurements & root-cause confirmation (2026-04-09)

**Companion to:** [`docs/mlx-swift-lm-prefill-memory-research.md`](./mlx-swift-lm-prefill-memory-research.md) (the prior research note from 2026-04-08).

**TL;DR.** The earlier research doc predicted a perf hit from the VLM `Qwen35.prepare` path which **bypasses chunked prefill entirely** (§2.6 of that doc). This note confirms the prediction with empirical numbers from 5 benchmark runs and a Time Profiler trace, after the `mlx-swift-lm v3` upgrade landed. **Prefill is 3.4× slower** than the pre-VLM-switch baseline. The root cause is a single-line change (`4de46681`, "feat(agent): add VLM support for ParoQuant models") that flipped `LLMModelFactory.shared.typeRegistry` → `VLMModelFactory.shared.typeRegistry`. This loaded `MLXVLM/Models/Qwen35.swift` instead of `MLXLLM/Models/Qwen35.swift`. The VLM class has its own `prepare` override that ignores `windowSize` and runs the entire prompt as one batch — even for text-only inputs.

**The fix is documented but not yet implemented.** This doc captures the measurements so a future session can verify the fix works.

---

## 1. Background — what changed and when

| date | commit | what |
|---|---|---|
| 2026-03-29 13:32 | (last 4B PARO bench before VLM switch) | baseline runs use **`LLMModelFactory.shared.typeRegistry`** → text-only Qwen35 path (chunks prefill, calls `eval(cache)` between chunks) |
| 2026-03-30 01:39 | `877d31d1 feat(agent): add image attachment support with VLM integration` | links `MLXVLM` framework, `ModelFactoryRegistry` auto-discovers it and starts loading Qwen3.5 as VLM |
| 2026-03-30 02:18 | `4de46681 feat(agent): add VLM support for ParoQuant models` | flips `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift` from `LLMModelFactory.shared.typeRegistry` → `VLMModelFactory.shared.typeRegistry`. From this commit on, `loadParoQuantModel` resolves the `qwen3_5` model_type via `MLXVLM/Models/Qwen35.swift` |
| 2026-04-08 | `docs/mlx-swift-lm-prefill-memory-research.md` written | identifies the issue in §2.6 ("The VLM `Qwen35.prepare` path does not chunk at all") and includes a fix in §7.1 patch ladder, but no fix lands |
| 2026-04-09 | `25917da8 build(mlx-swift-lm): upgrade fork to v3 main + cascade audio and app updates` | v3 upgrade. Cherry-picks ParoQuant + KV cache + TokenRing + FinalizedKVCacheHandle on top of upstream v3 main. **Inherits the VLM-path regression — does not fix it.** |

**The new context after the v3 upgrade:** all of the v3 work (Downloader/TokenizerLoader protocols, MLXHuggingFace macros, FinalizedKVCacheHandle, etc.) is functioning. Throughput is *not* regressed by anything v3 added — the regression was there since 2026-03-30 and was simply carried forward. Verified by the per-line diff: `MLXLLM/LLMModel.swift` (the chunking loop) and `MLXVLM/Models/Qwen35.swift:1104-1165` (the non-chunking override) are byte-for-byte identical between v2.29.1 (`2a296f1`) and v3 main (`7d9a6ab3`).

---

## 2. Measurements

Hardware: **Mac15,9 / M3 Max / 48 GB**, macOS 26.4. All runs `scripts/bench.sh quick --model qwen3.5-4b-paro`. Generation params identical (`t=1.0_p=0.95_k=20_pp=1.5`, `kvBits=8`, `kvGroupSize=64`, `prefillStepSize=256`, `maxTokens=2048`).

### 2.1 Aggregate, last three pre-VLM-switch baselines vs all post-v3 runs

| run | model name | gen tok/s | prefill tok/s¹ | p50 latency | peak mem | pass |
|---|---|---|---|---|---|---|
| 2026-03-29 13:20 | `Qwen3.5-4B PARO (INT4)` | **92.0** | **1301** | 3766 ms | 4273 MB | 7/14 |
| 2026-03-29 13:24 | `Qwen3.5-4B PARO (INT4)` | **95.1** | **1339** | 3752 ms | 4215 MB | 7/14 |
| 2026-03-29 13:32 | `Qwen3.5-4B PARO (INT4)` | **55.9**² | **1245** | 6324 ms | 4273 MB | 5/14 |
| 2026-04-09 00:34 | `Qwen3.5-4B PARO`³ | 70.7 | 379 | 10791 ms | 6213 MB | 4/14 |
| 2026-04-09 00:53 | `Qwen3.5-4B PARO`³ | 42.9 | 270 | 15263 ms | 6951 MB | 4/14 |
| 2026-04-09 04:33 | `Qwen3.5-4B PARO`³ | 69.9 | 373 | 10392 ms | 9270 MB⁴ | 3/14 |
| **median delta** | | **−25 %** | **−71 %** | **+2.8×** | **+50 %** | |

¹ Computed per-turn from `(promptTokens) / (latencyMs/1000 − genTokens/tokPerSec)`, then averaged across all turns of the run. Stable per-turn — not noise.

² Lower than the other two baselines because of thermal throttling kicking in at the back of the run, but **its prefill speed is still 1245 tok/s** — the throttling hits the GPU compute path, not the dispatch path.

³ The `(INT4)` suffix dropped because `tesseract/Features/Models/ModelDefinition.swift:92` now sets `displayName: "Qwen3.5-4B PARO"`. Same model, just a label change. Not related to perf.

⁴ Highest peak observed. The instrumented run logged per-scenario memory and showed peak hits **9270 MB during S2** (the long-prefill scenario with 4 turns and 18K-29K total prompt tokens), then stays at the high water mark for the rest of the run. Active memory holds steady at **2966 MB** throughout.

### 2.2 Per-turn determinism — S3 across 5 runs

Per-turn breakdown for S3 (3 turns, ~1100-2100 prompt tokens per turn). Prefill speed is rock-stable per-turn — confirms the regression is deterministic, not noise.

| run | turn | latency | gen | gen tps | prompt | prefill speed |
|---|---|---|---|---|---|---|
| base 13:20 | 0 | 2.64 s | 104 | 94 | 2016 | **1311 tok/s** |
| base 13:20 | 1 | 1.89 s | 104 | 95 | 1079 | **1364 tok/s** |
| base 13:20 | 2 | 1.34 s | 49 | 96 | 1119 | **1353 tok/s** |
| base 13:24 | 0 | 2.72 s | 111 | 95 | 2038 | **1306 tok/s** |
| base 13:24 | 1 | 1.91 s | 105 | 96 | 1104 | **1354 tok/s** |
| base 13:24 | 2 | 1.51 s | 64 | 96 | 1138 | **1349 tok/s** |
| base 13:32 | 0 | 2.66 s | 107 | 95 | 2018 | **1308 tok/s** |
| base 13:32 | 1 | 2.01 s | 116 | 96 | 1087 | **1360 tok/s** |
| base 13:32 | 2 | 1.42 s | 56 | 96 | 1121 | **1339 tok/s** |
| **v3 00:34** | 0 | 7.43 s | 142 | 73 | 2137 | **390 tok/s** |
| **v3 00:34** | 1 | 4.66 s | 125 | 74 | 1153 | **389 tok/s** |
| **v3 00:34** | 2 | 3.80 s | 54 | 73 | 1195 | **390 tok/s** |
| **v3 00:53** | 0 | 15.26 s | 255 | 72 | 4663 | **397 tok/s** |
| **v3 00:53** | 1 | 4.61 s | 120 | 75 | 1227 | **408 tok/s** |
| **v3 00:53** | 2 | 4.01 s | 65 | 75 | 1279 | **407 tok/s** |

Pre-VLM baseline prefill: **~1310-1360 tok/s** (extremely tight band).
Post-VLM-switch v3 prefill: **~390-410 tok/s** (also tight band, just much lower).

**Ratio: 1330 / 395 = 3.37× slower prefill.** Generation tps regressed only **1.28×** (95 → 73-75) — most of the wall-clock latency increase is the prefill phase.

### 2.3 Memory growth across scenarios (instrumented run, 2026-04-09 04:33)

Per-scenario memory snapshot — `BenchmarkRunner.swift` was patched to log `[mem active=… peak=…]` after each scenario for this investigation:

| scenario | turns | longest turn | active | **peak** |
|---|---|---|---|---|
| S1 | 3 | 12.3 s | 2966 MB | **5394 MB** |
| **S2** | 4 | **87.2 s** | 2966 MB | **9270 MB** ← jump |
| S3 | 3 | 18.4 s | 2966 MB | 9270 MB |
| S4–S15 | 2-6 | various | 2966 MB | 9270 MB |

- **Active memory holds at 2966 MB the entire run** — steady-state working set: model weights (~2.5 GB) + KV cache + overhead. Same value across all 14 scenarios. No leak.
- **Peak hits 9270 MB during S2**, the scenario with the longest single turn (87 seconds, ~7K prompt tokens, 4 sequential turns). After S2, peak doesn't grow further — `Memory.peakMemory` is a high-water mark.
- **The +5 GB delta between active and peak is transient prefill buffer accumulation** during S2's longest turn. With `prefillStepSize=256` and ~7K prompt tokens that's 27 chunks per turn — but because the **VLM `Qwen35.prepare` ignores `windowSize`** the entire 7K-token batch is materialized in one matmul, allocating multi-GB of intermediate activations.
- `Memory.cacheLimit = 2 GB` (set in `tesseract/Features/Agent/LLMActor.swift:Defaults.cacheLimitMB`) is **not bounding the peak** — the limit only triggers cache trim, the live working set during a single forward pass can exceed it freely.

### 2.4 Time Profiler hot path (Instruments, 14.6 s S3-only trace)

`xcrun xctrace record --template "Time Profiler" --attach <pid>` against `bench.sh quick --bench-scenarios S3`. 10,524 samples, 5.89 s of CPU time on the hot thread (the rest of the 14.6 s wall-clock is the thread blocked waiting for GPU dispatch — see §3 for what that implies).

| share of CPU time | call path |
|---|---|
| **50.4 %** | inside `TokenIterator.prepare` (the prefill loop) |
| 38.6 % | inside `generateLoopTask` / `TokenIterator.next` (generation) |
| 26.6 % | inside `Qwen35` model code (call → attention → matmul) |
| 62.1 % | reaches `mlx::core::async_eval` |
| 2.7 % | inside `RotateQuantizedLinear` (the ParoQuant rotation kernel) |
| ~3 % | inside `AGX::ComputeContext` / `IOGPUResourceListAddResource` (Metal dispatch) |

**~28 % of total CPU time is spent in `_xzm_free` / `_xzm_xzone_malloc` / `_platform_memmove` / `_platform_memset` / `mlx::core::array::~array` / `std::shared_ptr<ArrayDesc>` etc.** Of those 2.20 s of allocation time, the breakdown by deepest interesting Swift caller is:

| share of alloc time | caller |
|---|---|
| 38.6 % (0.85 s) | `TokenIterator.prepare` |
| 24.4 % (0.54 s) | `TokenIterator.next` |
| 22.1 % (0.49 s) | `Qwen35` model layers |

The actual GPU compute (Metal dispatch) is only ~3 % of CPU samples; the rest of the wall-clock is the CPU thread blocked waiting for GPU. **CPU dispatch can't keep up with GPU because every prefill chunk allocates fresh intermediates.** This is consistent with the Apple/MLX maintainer position quoted in `mlx-swift-lm-prefill-memory-research.md` §2.3: the buffer cache only releases under pressure, and the lazy graph keeps everything alive until `eval()`.

---

## 3. Root cause — confirmed

The previous research note (`mlx-swift-lm-prefill-memory-research.md` §2.6) called this out 24 hours before my measurements, but I'm reproducing the diff here for the next session because it's the exact 60-line diff that needs to change.

### 3.1 The two prepare implementations

**`Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift:21-36`** — the LLM (text-only) path. Inherited by `MLXLLM/Models/Qwen35.swift`'s `Qwen35Model: LLMModel`. **Chunks the prompt, calls `eval(cache)` between chunks**:

```swift
public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
    -> PrepareResult
{
    let prefillStepSize = windowSize ?? 512
    var y = input.text

    while y.tokens.size > prefillStepSize {
        let input = y[.newAxis, ..<prefillStepSize]
        _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
        eval(cache)               // ← forces evaluation, releases temporaries
        y = y[prefillStepSize...]
        // (clearCache() also missing here per the prior research note §7.1)
    }

    return .tokens(y)
}
```

**`Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:1104-1165`** — the VLM path. **Ignores `windowSize`, runs the entire prompt as one batch**:

```swift
public func prepare(
    _ input: LMInput,
    cache: [any KVCache],
    windowSize _: Int?               // ← parameter name is `_`, value discarded
) throws -> PrepareResult {
    let inputIds = input.text.tokens

    // … vision embedding merge logic, only triggered if input.image != nil …

    let typedCache = castCache(cache)
    let output = languageModel(
        inputIds,                     // ← whole prompt, all N tokens, in one shot
        inputsEmbeds: inputEmbeddings,
        cache: typedCache,
        mask: input.text.mask,
        positionIds: nil,
        pixelValues: pixelValues,
        imageGridTHW: imageFrames,
        videoGridTHW: videoFrames
    )

    return .logits(output)
}
```

**The VLM Qwen35 path is correct for image-bearing inputs** — vision embeddings need to be merged into the text embedding sequence at specific image-token positions, and that merge has to happen on the full sequence before any forward pass. **But for text-only inputs (no `input.image`, no `input.video`) the merge is skipped entirely** (the `if let pixelValues, …` branch is not taken; only the `else` calls `languageModel.resetPositionState()`), and the function still drops to a single un-chunked `languageModel(inputIds, …)` call.

### 3.2 Why this single line is the entire regression

- Tesseract loads ParoQuant Qwen3.5 via `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift:357-367` (current line numbers post-v3) which calls `MLXLMCommon.loadParoQuantModel(from:typeRegistry: VLMModelFactory.shared.typeRegistry, …)`. This was changed from `LLMModelFactory.shared.typeRegistry` in commit `4de46681` (2026-03-30 02:18:34).
- `VLMTypeRegistry.shared` (`Vendor/mlx-swift-lm/Libraries/MLXVLM/VLMModelFactory.swift:87`) maps `"qwen3_5" → MLXVLM.Qwen35.init`. So our PARO model now resolves to the VLM Qwen35 class with its own non-chunking `prepare`.
- `LLMTypeRegistry.shared` (`Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift:38`) maps `"qwen3_5" → MLXLLM.Qwen35Model.init`. The LLM Qwen35Model does **not** override `prepare`, so it inherits the chunking version from `LLMModel.swift`.
- The user prompts (per §2.3) are 1000-7000 tokens each. With `prefillStepSize=256`, the LLM path would process them in 4-27 chunks, releasing transients between each. The VLM path materializes them in a single matmul → fully populated activation tensors for every layer simultaneously → +5 GB peak, 3.4× slower per token.

### 3.3 Why generation is also slower (1.28×)

The VLM `Qwen35.callAsFunction` (`Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:1167-1180`) has:

```swift
public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
    let typedCache = castCacheOptional(cache)    // ← allocates a new [KVCache] per call
    let result = languageModel(
        inputs,
        inputsEmbeds: nil, cache: typedCache,
        mask: nil, positionIds: nil,
        pixelValues: nil, imageGridTHW: nil, videoGridTHW: nil
    )
    return result.logits
}
```

`castCacheOptional` calls `castCache` which does `cache.map { $0 }` — i.e. **rebuilds the entire cache array per generated token**. For Qwen3.5 with 32 layers, that's a 32-element array allocation every step. Per-token CPU overhead, modest but real. The LLM `Qwen35Model.callAsFunction` just does `languageModel(inputs, cache: cache)` — no per-call array re-allocation.

---

## 4. Fix options (not yet attempted)

Listed cheapest → most thorough. I have **not** implemented any of these — the user explicitly asked to defer fixing until a later session, and the existing research note `mlx-swift-lm-prefill-memory-research.md` §7.1 already lays out the patch ladder. This section adds new options that became feasible after the v3 upgrade and the measurements above.

### 4.1 [Cheapest, app-side, no fork edit] Route text-only inputs through LLM, image inputs through VLM

In `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift`, load **both** containers at startup:

- one via `LLMModelFactory.shared.typeRegistry` (for text-only)
- one via `VLMModelFactory.shared.typeRegistry` (for image inputs)

Route in `LLMActor.generate`: if `input.image == nil && input.video == nil`, dispatch to the LLM container; else dispatch to the VLM container.

**Cost:** 2× model weight memory (~5 GB extra) because both copies of Qwen3.5 are loaded. Could be mitigated by sharing the language_model module between the two via MLX's parameter sharing, but that requires fork-side changes.

**Benefit:** zero fork changes, no risk of regressing image-bearing inputs.

**Why this might be the right call:** Tesseract's actual usage today is ~99% text-only, with images being a recent addition. Paying for two model copies avoids the per-prefill cost on the common path.

### 4.2 [Cheapest, fork-side] Add chunking to `MLXVLM/Models/Qwen35.swift:1104` for text-only path

Single fork edit. After the vision merge branch, instead of one giant `languageModel(inputIds, …)` call, fall through to the chunking pattern from `LLMModel.swift`:

```swift
// Replace lines 1152-1164 with:
let typedCache = castCache(cache)
let prefillStepSize = windowSize ?? 512   // ← actually use windowSize
var y = input.text

if pixelValues == nil && inputEmbeddings == nil {
    // Pure text path — use chunking, identical to LLMModel.swift
    while y.tokens.size > prefillStepSize {
        let chunk = y[.newAxis, ..<prefillStepSize]
        _ = languageModel(
            chunk.tokens, inputsEmbeds: nil,
            cache: typedCache, mask: nil, positionIds: nil,
            pixelValues: nil, imageGridTHW: nil, videoGridTHW: nil)
        eval(typedCache ?? [])
        y = y[prefillStepSize...]
    }
    return .tokens(y)
}

// Image-bearing path — single shot is fine because the merge step
// already requires the whole sequence in memory
let output = languageModel(
    y.tokens, inputsEmbeds: inputEmbeddings, cache: typedCache,
    mask: y.mask, positionIds: nil,
    pixelValues: pixelValues, imageGridTHW: imageFrames, videoGridTHW: videoFrames)
return .logits(output)
```

**Cost:** ~20 lines in the fork, on top of the existing v3 cherry-pick chain. Have to keep this patched in our `test/tesseract-integration-v3` branch until/unless upstream accepts a similar fix.

**Benefit:** Fixes the regression for both text-only and image-light prompts. Image-heavy prompts (where the merge dominates) are unchanged.

**Verification:** rerun `scripts/bench.sh quick --model qwen3.5-4b-paro` and check prefill speed returns to >1200 tok/s. Single bench run takes ~3 minutes.

### 4.3 [Combined with the prior research note's §7.1] Also add `MLX.GPU.clearCache()` between chunks

Per the prior research note, the LLM path's chunking loop is also missing `MLX.GPU.clearCache()`. Adding chunking to the VLM path **and** adding `clearCache()` in both `LLMModel.swift` and the new chunking branch in `Qwen35.swift` is a single coordinated fork patch. Per `mlx-lm` PR #917 the clearCache fix alone took peak prefill memory **50 GB → 28 GB** on a comparable workload. We don't have crash-level pressure right now (active is stable at 3 GB), but when context grows to 16K+ this becomes load-bearing.

### 4.4 [Best long-term] Delete the VLM Qwen35 prepare override entirely

The cleanest design is to make the VLM `Qwen35.prepare` either:
- Detect text-only and delegate to `(self as LanguageModel).prepare(...)` (which would route through the LLMModel default chunked implementation), OR
- Detect text-only and just call `Qwen35Language.LanguageModel.prepare(...)` directly, bypassing the VLM wrapper

Both require a small refactor of `MLXVLM/Models/Qwen35.swift`. Probably worth doing once we've validated 4.2 works.

---

## 5. What I changed during the investigation (already committed elsewhere)

- `tesseract/Features/Agent/Benchmark/BenchmarkRunner.swift` — added one `[mem active=… peak=…]` field to the per-scenario log line. Pure observability addition, no behavioral change. Should be **kept** — it caught this regression instantly. Currently uncommitted on `feat/mlx-swift-lm-v3`.

Nothing else was modified for this investigation. The fork is unchanged from `25917da8`.

---

## 6. Trace files (saved for the next session)

```
/tmp/tesse-prof/v3_S3_attach.trace      21 MB    Time Profiler — open in Instruments.app
/tmp/tesse-prof/timeprof.xml             6 MB    xctrace XML export of the same
```

The 12 GB Allocations trace was deleted. If the next session wants to re-run Allocations, the working command is:

```bash
APP="/Users/owl/Library/Developer/Xcode/DerivedData/tesseract-buwysfpnwmzyucelgewutuddcvgv/Build/Products/Release/Tesseract Agent.app"

killall "Tesseract Agent" 2>/dev/null || true
open -W "$APP" --args --benchmark --bench-sweep quick --bench-scenarios S3 --bench-model-id qwen3.5-4b-paro &
OPEN_PID=$!
for i in $(seq 1 30); do
    AGENT_PID=$(pgrep -x "Tesseract Agent" | head -1)
    [ -n "$AGENT_PID" ] && break
    sleep 0.5
done

xcrun xctrace record \
    --template "Time Profiler" \
    --output /tmp/tesse-prof/v3_S3_NEW.trace \
    --no-prompt \
    --attach "$AGENT_PID" &
wait $OPEN_PID
```

The key trick is to launch the app first (so it goes through `open` / LaunchServices and inherits the right entitlements), then attach `xctrace` to the spawned PID. Direct `xctrace --launch -- /path/to/binary` only profiles the launcher (it exits in 1 s).

---

## 7. Concrete next-session checklist

- [ ] Decide between fix path 4.1 (app-side dual container) vs 4.2 (fork-side chunking) — see tradeoffs above. Recommend **4.2** for cleaner long-term but **4.1** for fastest path-to-shipping if the user wants to avoid more fork changes
- [ ] If 4.2: edit `Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift` lines ~1104-1164, cherry-pick the change as a new commit on `test/tesseract-integration-v3`, push to fork, bump submodule pointer
- [ ] Re-run `scripts/bench.sh quick --model qwen3.5-4b-paro` and confirm prefill returns to >1200 tok/s, peak memory drops below 5 GB
- [ ] If both numbers come back, this work is done — close the loop in `mlx-swift-lm-prefill-memory-research.md` §8 action plan
- [ ] If still slow, the residual is the missing `clearCache()` calls per §7.1 of the prior research note — add those next
- [ ] Decide whether to commit the `BenchmarkRunner.swift` per-scenario memory log (see §5 of this doc) — recommend **yes**, it took 1 line and immediately surfaced the S2 spike

---

## 8. Cross-references

- [`docs/mlx-swift-lm-prefill-memory-research.md`](./mlx-swift-lm-prefill-memory-research.md) — the prior (more comprehensive) research note. §2.5 / §2.6 / §7.1 are the directly relevant sections. This doc adds empirical confirmation and post-v3 status.
- [`docs/mlx-swift-lm-kv-cache-audit.md`](./mlx-swift-lm-kv-cache-audit.md) — adjacent KV cache audit, may be relevant for fix path 4.4 if we pursue the deeper refactor.
- v3 upgrade commit: `25917da8 build(mlx-swift-lm): upgrade fork to v3 main + cascade audio and app updates`
- VLM switch commits: `877d31d1` (link MLXVLM framework), `4de46681` (flip ParoQuant loader to VLM type registry)
- Affected file in tesseract: `tesseract/Features/Agent/ParoQuant/ParoQuantLoader.swift:357-367`
- Affected file in fork: `Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:1104-1180`
- Plain LLM path (the reference implementation): `Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift:21-36`
- Where `prefillStepSize` is set in tesseract: `tesseract/Features/Agent/AgentGeneration.swift:26` (currently 256)
