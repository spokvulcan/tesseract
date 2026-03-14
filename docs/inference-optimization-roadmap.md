# Inference Optimization Roadmap

Areas for further performance improvement in the MLX inference pipeline. Each item includes the affected file, the bottleneck, and a brief implementation sketch.

---

## Completed

### 1. GPU-only penalty processors — **+33% peak tok/s**
- **PR**: [ml-explore/mlx-swift-lm#147](https://github.com/ml-explore/mlx-swift-lm/pull/147)
- Eliminated `token.item(Int.self)` CPU←GPU sync in `didSample()` for all three penalty processors
- Restored `asyncEval()` pipelining in `TokenIterator`

### 2. TopPSampler argPartition — **+75% aggregate tok/s**
- **PR**: same as above
- O(V) `argPartition` + O(K log K) sort on K elements instead of O(V log V) full sort
- Switched `take()` → `takeAlong()` to preserve shapes

### 3. PARO rotation eval cache
- **File**: `tesseract/Features/Agent/ParoQuant/RotateQuantizedLinear.swift:171`
- Pre-materializes cos/sin/packedPairs with `eval()` so they're GPU-resident constants, not recomputed in every forward pass graph

---

## Remaining Opportunities

### 4. Fused rotation + quantized matmul kernel

**Bottleneck**: Each `RotateQuantizedLinear` dispatches 2 Metal kernels per forward pass (rotation + `quantizedMM`). With 32 layers × ~7 projections = ~196 layers, that's ~392 kernel dispatches per token vs ~196 for standard 8-bit.

**Files**:
- `tesseract/Features/Agent/ParoQuant/RotateQuantizedLinear.swift:208-220` — `callAsFunction` dispatches rotation then quantizedMM
- `z-lab/paroquant/paroquant/kernels/metal/rotation.metal` — reference Metal kernel

**Approach**: Write a single Metal kernel that reads quantized INT4 weights, dequantizes them, applies Givens rotation to the input activation, and accumulates the dot product — all in one pass. This eliminates the intermediate rotation output buffer and halves kernel dispatch count.

**Complexity**: High — requires reimplementing `quantizedMM` internals. The MLX `quantized_matmul` kernel handles complex dequantization (group-wise scales/biases, bit unpacking). The fused kernel would need to replicate this plus add rotation.

**Alternative**: Express rotation as native MLX ops (`matmul`, element-wise) instead of a custom Metal kernel. MLX's graph compiler may fuse them with the subsequent `quantizedMM` automatically. This is simpler but may not fuse effectively at batch=1.

**Expected impact**: ~5-10% tok/s improvement from halved dispatch count and eliminated intermediate buffer.

---

### 5. Group-wise precomputed rotation matrices

**Bottleneck**: The current rotation applies 8 rounds of Givens rotations per group using a custom Metal kernel with threadgroup barriers. At batch=1, each round does minimal work (64 pairs × 1 row) but pays full barrier + dispatch overhead.

**Files**:
- `tesseract/Features/Agent/ParoQuant/RotateQuantizedLinear.swift:10-83` — Metal kernel source
- `tesseract/Features/Agent/ParoQuant/RotateQuantizedLinear.swift:171-186` — `ensureCached()`

**Approach**: At model load time, compute the dense rotation matrix Q for each group by multiplying out all 8 Givens rotation matrices: `Q = G_1 · G_2 · ... · G_8`. Each group's Q is `[groupSize, groupSize]` = `[128, 128]`. Store these and replace the per-token Metal kernel with `matmul(x_group, Q_group)` calls. Total memory: 20 groups × 128 × 128 × 2 bytes (float16) = 640 KB per layer, ~140 MB total for 224 layers.

**Trade-off**: Memory (140 MB) vs compute (eliminates 8 barrier rounds and custom kernel). At batch=1, the 128×128 matmul is fast on Apple Silicon. At batch>1 (prefill), the current kernel might actually be faster due to its tiled parallelism.

**Expected impact**: ~5-15% tok/s improvement at batch=1 decoding. Needs benchmarking vs current kernel.

---

### 6. Batch rotation kernels across projections

**Bottleneck**: Within a single transformer layer, multiple projections (q, k, v, gate, up) share the same input dimension (2560) and each dispatches its own rotation kernel. These could be batched.

**Files**:
- `tesseract/Features/Agent/ParoQuant/RotateQuantizedLinear.swift:188-206` — `rotate()` method
- Qwen3.5-4B layer structure: q/k/v/gate/up all take 2560-dim input

**Approach**: Instead of 5 separate rotation dispatches for projections with the same input dim, concatenate their rotation parameters and dispatch a single batched kernel. Requires modifying the model's forward pass to group projections, which means changes to the Qwen3.5 model implementation in mlx-swift-lm.

**Complexity**: Medium-high — requires model architecture changes, not just kernel changes.

**Expected impact**: ~196 rotation dispatches → ~96 dispatches per token (5 projections batched into 1 per layer for same-dim groups).

---

### 7. `TokenIterator.next()` returns `Int` — forces one sync per token

**Bottleneck**: `TokenIterator` conforms to `IteratorProtocol` with `Element = Int`. The `next()` method must call `previousY.tokens.item(Int.self)` to return a CPU-side integer. This is currently "free" (reads the previous token which was already `asyncEval`'d), but it fundamentally limits pipelining to one token of lookahead.

**File**: `mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:712-728`

**Approach**: Change `TokenIterator` to yield `MLXArray` tokens instead of `Int`. The generation loop (`generateLoopTask`) would then handle EOS checking and detokenization on GPU-resident tokens. This requires a protocol change:
```swift
// Current:
public struct TokenIterator: Sequence, IteratorProtocol {
    mutating public func next() -> Int? { ... }
}

// Proposed:
public struct TokenIterator: Sequence, IteratorProtocol {
    mutating public func next() -> MLXArray? { ... }
}
```

The `item(Int.self)` call moves to `generateLoopTask` where it's needed for EOS checking and detokenization. This is a **breaking API change** but could enable multi-token lookahead in the future.

**Complexity**: Medium — API change affects all callers of `TokenIterator`.

**Expected impact**: Minimal for current single-token generation. Enables future speculative decoding optimizations.

---

### 8. EOS/stop token checking on GPU

**Bottleneck**: The generation loop checks `stopTokenIDs.contains(token)` on CPU for every token. With our penalty fix, this is the main remaining reason token IDs need to be on CPU.

**File**: `mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:1330`

**Approach**: Build a GPU-resident EOS mask at generation start:
```swift
let eosMask = MLXArray.zeros([vocabSize], type: .bool)
for id in stopTokenIDs { eosMask[id] = true }
```
Then check `eosMask[token].item(Bool.self)` — but this still syncs. A better approach: let the model generate freely and check EOS in batches (every N tokens), doing one sync per batch instead of per token. This is speculative execution — generate N tokens assuming none are EOS, then verify.

**Complexity**: High — changes generation semantics. Must handle the case where EOS appears mid-batch.

**Expected impact**: Potentially significant for very fast models where per-token sync overhead is proportionally larger.

---

### 9. Streaming detokenizer optimization

**Bottleneck**: `NaiveStreamingDetokenizer` processes tokens one at a time, calling into the tokenizer for each token. The tokenizer is CPU-bound Swift/Python code.

**File**: `mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:1171` — `TextToolTokenLoopHandler`

**Approach**: Batch detokenization — accumulate N tokens before calling the tokenizer. The tokenizer can decode `[Int]` arrays more efficiently than single tokens. This trades latency (text appears in bursts) for throughput.

**Expected impact**: Small — detokenization is typically <1% of total time.

---

### 10. GPU thermal throttling mitigation

**Bottleneck**: On sustained inference (>30s continuous generation), Apple Silicon GPU thermally throttles, reducing tok/s by 40-67%. This is a hardware constraint, not a software bug.

**Evidence**: Benchmark experiments showed PARO drops from 95 to 27 tok/s after ~40s of continuous generation. Adding 10s cooldown between scenarios maintained 67-68 tok/s throughout.

**Files**:
- `tesseract/Features/Agent/Benchmark/BenchmarkRunner.swift:54-75` — scenario loop
- `tesseract/Features/Agent/Core/AgentLoop.swift` — production agent loop

**Approaches**:
1. **Benchmark**: Add configurable cooldown between scenarios for accurate measurements
2. **Production**: Accept throttled performance as steady-state. Optimize for lower GPU utilization per token (items 4-6 above) to delay thermal throttling onset
3. **Investigate KV cache quantization**: `GenerateParameters.kvBits` can quantize the KV cache to reduce memory bandwidth and GPU work. This is already supported by mlx-swift-lm but not enabled in tesseract

**Expected impact**: Can't eliminate thermal throttling, but reducing per-token GPU work (items 4-6) would delay its onset and raise the throttled floor.

---

### 11. Memory cache limit tuning

**Bottleneck**: `LLMActor` sets `Memory.cacheLimit = 128 MB` before each generation. Experiments showed 2 GB was optimal — large enough for buffer reuse, small enough to avoid memory pressure.

**File**: `tesseract/Features/Agent/LLMActor.swift:15`

**Approach**: Change `cacheLimitMB` from 128 to a tuned value (512-2048 MB depending on device memory). On 8 GB devices, keep it small. On 48 GB+, use 1-2 GB. Could be made adaptive based on `ProcessInfo.processInfo.physicalMemory`.

**Expected impact**: Small but consistent — prevents buffer reallocation overhead in multi-turn conversations.

---

## Priority Order

| # | Optimization | Impact | Complexity | Target |
|---|-------------|--------|------------|--------|
| 11 | Cache limit tuning | Small | Trivial | tesseract |
| 5 | Precomputed rotation matrices | 5-15% | Medium | tesseract |
| 4 | Fused rotation+quantizedMM | 5-10% | High | tesseract |
| 10 | Thermal mitigation (KV quant) | Variable | Low | tesseract |
| 7 | TokenIterator yields MLXArray | Enabling | Medium | mlx-swift-lm PR |
| 6 | Batched rotation kernels | ~10% | Medium-high | tesseract |
| 8 | GPU-side EOS checking | Small | High | mlx-swift-lm PR |
| 9 | Batch detokenization | Tiny | Low | mlx-swift-lm PR |

---

## References

- [ParoQuant paper](https://arxiv.org/abs/2511.10645) — rotation overhead benchmarks (Table 3)
- [z-lab/paroquant](https://github.com/z-lab/paroquant) — reference CUDA/Metal kernels
- [mlx-swift-lm PR #147](https://github.com/ml-explore/mlx-swift-lm/pull/147) — our GPU-only penalties + argPartition PR
- [MLX Memory.swift](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Memory.swift) — cache limit documentation
- [Experimental results](docs/paro-inference-degradation-research.md) — thermal throttling evidence from 5 benchmark experiments
