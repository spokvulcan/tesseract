# Local-inference allocation inventory

This records the September 12 baseline. The subsequent fixes and measurements
are documented in [the September 13 investigations](2026-09-13-allocation-investigations.md).

Issue [#506](https://github.com/spokvulcan/tesseract/issues/506), first evidence
deliverable for [#505](https://github.com/spokvulcan/tesseract/issues/505).
Source baseline: `b1f0dca32ec9baead778221a8001fe4aa59e1850` (#503). The inventory
is a map of allocation purposes and lifetimes, not a claim that every retained
buffer is waste. No production optimization is implemented in this change.

The original brief excluded loaded-model runs. The maintainer subsequently
explicitly requested production-model measurements and new diagnostics where
needed. The bounded campaign below follows that revised scope; it does not
authorize the separate 45k/75k/93k comparison on this 48 GiB host.

## Evidence and accounting rules

- **Source-established:** ownership, array shapes, allocation operations and
  release intent traced at the baseline and the accompanying diagnostic diff.
  Swift scope exit and logical byte counters do not prove an OS footprint drop.
- **Historical measurement:** preserved results from earlier builds, with their
  source/build and workload limitations. They are not measurements of this change.
- **New production measurement:** only the instrumented Release campaign whose
  binary/model hashes, exact diff, runner and scalar observations are preserved
  in the [capture directory](../../benchmarks/allocation-inventory/2026-09-12/).
- **Unvalidated:** an explicit gap, including allocation-stack identity, backing
  retained outside a view's extent, precision/layout variants and long contexts
  not exercised by the campaign. Potential savings remain unknown.

| Quantity | Meaning and limitation |
| --- | --- |
| Logical token offset | Valid conversation rows. Leaf Rewind restores this without necessarily shrinking storage. |
| Charged snapshot bytes | Sum of snapshot array `nbytes` used by cache policy. May describe views, exclude slack, or remain charged while the tree has transferred the body. |
| Array extent bytes | Shape product × element width. `innerState()` exposes cache capacity arrays where implemented, but not allocator padding or a larger backing hidden behind a view. |
| Unique physical backing | Requires allocation identity/allocator evidence. Several views, snapshot wrappers, full payload aliases and model module references must not be counted again. This audit has no complete allocation-stack census. |
| Active/cached MLX bytes | Live allocator allocations versus reusable free buffers; neither is independently additive to process footprint. |
| Lifetime MLX peak / sampled request maximum | An all-process high-water mark versus the largest sampled value during the request. Neither supplies attribution to an old peak or guarantees capture of every transient. |
| Footprint / resident / compressed / system swap | Different OS accounting views. System swap includes other processes and can stay elevated after allocation release. RSS alone omits much of this MLX workload's footprint. |
| Pending SSD payload bytes | Outstanding logical queue charge, including the current writer, not unique physical staging bytes. Queue count counts waiting entries only. |

Sources: [`RequestMemoryTelemetry`](../../tesseract/Features/Server/Telemetry/RequestMemoryTelemetry.swift),
[`HybridCacheSnapshot`](../../tesseract/Features/Server/HybridCacheSnapshot.swift),
[`LeafCheckout`](../../tesseract/Features/Server/LeafCheckout.swift),
[`SSDSnapshotStore`](../../tesseract/Features/Server/SSDSnapshotStore.swift).
Component facts retain `requestCacheMeasuredAtPhase` / `treeMeasuredAtPhase`;
use these freshness fields rather than attributing an old count to a later phase.

## Baseline protections and preserved results

#503 already implements eligible Leaf Handoff and exact Leaf Rewind, clears
`FinalGenerationCache` ownership after transfer, and constructs production
caches with `alphaTuner: nil`. These are existing protections. The inventory
does not claim their savings again. Structural, pending-full-payload, rotating,
image-key-space and quantized cases still have justified copy fallbacks.
See [ADR-0064](../adr/0064-leaf-handoff.md),
[`ServerCompletion`](../../tesseract/Features/Server/ServerCompletion.swift),
and [`LeafCheckout`](../../tesseract/Features/Server/LeafCheckout.swift).

The [bounded checkout evidence](../../benchmarks/leaf-checkout/2026-09-12/README.md)
used two 4,096 × 64 float32 attention arrays (2,097,152 bytes) and a 64-byte
recurrent backup. Across 24 cycles, each checkout added 64 active MLX bytes;
copied restore added 2,097,232 bytes. All 24 retired owners released cache state.
The final 131,072-byte active increase matched one 256-row capacity step, not
an unexplained leaked owner. These are historical toy-cache measurements,
not loaded-model peak/settled estimates. The
[review follow-up](../../benchmarks/leaf-checkout/2026-09-12-review/README.md)
records the later validation and the opt-in isolated-allocation gate.

The [AlphaTuner incident](../../benchmarks/incidents/2026-09-12-alpha-tuner/README.md)
strongly attributed 16,716,988,416 additional active bytes (15.57 GiB) and a
64.636-second post-generation tail to synthetic replay allocations. This is
historical source/timing/allocation evidence, not a profiler-stack capture;
the production path is now disconnected. [#504](https://github.com/spokvulcan/tesseract/issues/504)
owns any redesign. The provisional 4–8 GiB allowance remains an unvalidated
estimate, not a measured pool of recoverable bytes.

## Inference and cache allocation families

Use `B` for batch size, `N` for logical rows, `C` for allocated row extent,
`Hkv` for KV heads, `Dk/Dv` for head dimensions, `b` for element bytes,
`S` for prefill chunk/verify rows and `V` for vocabulary size. Formula values
must come from the loaded model's shapes/dtypes, not its checkpoint file size.

| Family; creating operation and owner | Size, aliasing and bound | Last consumer and release boundary; disposition |
| --- | --- | --- |
| Full-attention cache; `KVCacheSimple.update` / DFlash2 `writeRows`; request or radix leaf owns cache objects | Per layer: `B*C*Hkv*(Dk+Dv)*b`; logical content substitutes `N` for `C`. Normal growth uses 256-row steps; `writeRows` also reserves lookahead. Old arrays, zeros, concatenated output and dependent views may overlap during growth. No fixed context-independent physical bound. | Model forwards read valid rows. Eligible check-in moves the same objects into the tree; cancellation/failure trims offsets and returns the original leaf. Supersession/demotion may leave payload aliases alive. Quiescence is required before move/rewind. Required state; excess capacity is a #501 candidate. |
| Recurrent live state; model GDN/Mamba layers; live cache owner | Sum of convolution state and recurrent matrices, plus lengths/padding/slots. Matrix term scales with recurrent layer count, value heads and key/value head dimensions, usually independent of conversation length; actual dtype matters. | Last forward/rewind reader must finish. Successful handoff transfers state; cancel/failure reinstalls independent backup including metadata. Required state, not disposable overhead. |
| Independent Leaf Rewind backup; `LeafCheckout.init` owns copied recurrent `LayerState` | `sum(saved recurrent arrays.nbytes)`; deliberately independent via evaluated deep copies. Does not copy full attention. Metadata is extra host storage. | Last possible cancel/failure rewind; `checkIn` or `rewindIfNeeded` clears checkout ownership after return. Cannot use speculative capture aliases as this backup. Already addressed by #503; production size observable separately. |
| Immutable checkpoints and copy restore; `HybridCacheSnapshot.capture/restore`, prefill and boundary owners | Sum of copied array extents at each captured offset. `array * 1` in its own dtype creates lazy independent state, then one hoisted `eval` realizes all copies. Source and destination coexist; several necessary checkpoints can overlap. | Structural/think-strip restores and post-answer seeds may still need snapshots after the foreground response. Transfer/admission or task completion releases local owners; failure/cancel must first settle GPU work. Required copies remain unless a specific last-consumer violation is demonstrated. |
| Prefill activations, logits and chunk state; `PrefillExecutor` and model forward | Embeddings/hidden/MLP terms scale with `B*S*hidden/intermediate`; logits can scale with `B*S*V*b`; attention workspace depends on kernel, prompt and chunk shape. This is not an exact allocator peak formula. Lazy graph inputs can survive lexical scopes. | Chunk/cache evaluation and the output's final consumer settle the graph; pipelined mode overlaps work intentionally. Checked-synchronous and final evaluation are existing boundaries. #501 owns allocation attribution; do not insert per-token synchronization to improve a memory sample. |
| Post-answer speculative canonical prefill; seed, warm state and task registry | Holds completed-conversation host data, restored cache, growing suffix cache and potentially captured progress. Copy restore can overlap the source leaf. Bound by planned residual path, not a tiny fixed scratch allowance. | Foreground entry/unload cancels and awaits the task, including eligible progress admission. Natural completion admits the leaf and releases the warm cache. ADR-0009's think-strip performance contract requires this work; study lifetimes without dropping required checkpoints/progress. |
| Completion handles, start registries, streams and closures; `ServerCompletion` / `FinalGenerationCache` | Several handles may refer to one owner; counting handles as caches is wrong. Start/drive tasks retain Model Session access through cleanup. Stream/host buffer bytes are separate from live KV. | Drain waits for starts, generation and speculative prefill. A retained completed handle must not retain cache after move/check-in/rewind. Historical toy assertions cover this; long-model terminal release timing remains a separate observation, not proven by niling one field. |

Sources: [`KVCacheSimple`](../../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift),
[`DFlash2 cache writes`](../../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/DFlash2DrafterModel.swift),
[`LeafCheckout`](../../tesseract/Features/Server/LeafCheckout.swift),
[`HybridCacheSnapshot`](../../tesseract/Features/Server/HybridCacheSnapshot.swift),
[`PrefillExecutor`](../../tesseract/Features/Agent/PrefillExecutor.swift),
[`SpeculativeCanonicalPrefill`](../../tesseract/Features/Server/SpeculativePrefill.swift),
[`ServerCompletion`](../../tesseract/Features/Server/ServerCompletion.swift),
[ADR-0009](../adr/0009-speculative-canonical-prefill-post-answer-gpu.md).

### DFlash2 working state

The currently inspected draft config specifies width 8, five capture layers,
five draft layers, eight KV heads, head dimension 128 and sliding window 2,048
(context window 2,047). Config hashes are recorded with production evidence.
Use current source plus [ADR-0061](../adr/0061-dflash2-upstream-shape.md): older
ADR-0057 descriptions of state-key interfaces and rewind were superseded.

| Owner / allocation family | Size and overlap | End of lifetime / next evidence |
| --- | --- | --- |
| Iterator `prepare`: captured target hidden outputs, concatenated hidden window and last logits | Retained logical hidden window is `B*min(rows,2047)*sum(captured hidden widths)*b`. With five 5,120-wide layers and 2-byte elements this is 104,806,400 bytes, a formula example rather than an observed unique allocation. Concatenation with a new chunk, slicing and final logits can retain larger backings/graphs. | Local window feeds the first draft round. Intermediate chunks evaluate cache/hidden arrays; final dependencies may survive into that round. #501 needs operation-level measurements to establish physical overlap. |
| Iterator `drafterState.contextCaches`: projected sliding K/V plus row metadata | Each store initially reserves `max(newRows,window+256)` rows. Compact gathers newest valid rows into fresh padded buffers; `withBlock` borrows slack or concatenates on fallback. Window + slack + current append bounds logical shape for this configuration, not allocator/GPU overlap. | Retained across rounds, released with iterator state. Compaction temporarily needs old and new storage. Already bounded reuse; measure before proposing a pool or different compaction. |
| Iterator `inFlight` and `committedCaptures`: verify logits/hidden and GDN replay inputs | Verify logits scale with `width*V*b`; sampled acceptance adds float32 probability/filter/residual arrays. `GatedDeltaCapture` retains conv input, k/v, gates and initial recurrent state. Next-round graphs are built before packed acceptance synchronizes, so old captures and next-round work intentionally overlap. | Old round becomes obsolete after commit/next graph consumption; committed captures remain for undrained-token finalization. Finalizing rewind alone does not prove iterator captures were deallocated. Required lookahead/correctness state; measure #501 lifetime/peak, not a simple sum of references. |
| Iterator host pending tokens and logit processor scratch copy | Pending output is at most a round's drained/undrained tokens, reused with `keepingCapacity`; processor scratch is copied per processed verify sequence. Candidate tables scale with width × top-K; distribution arrays with vocabulary. | Next round replaces pending storage contents; iterator/processor release ends ownership. Full-vocabulary repeated materialization is a measured-investigation candidate, not an established savings figure. |
| Loaded draft modules: compiled segments, dynamic-convolution tap caches and stacked projections | Tap dictionaries are keyed by two kernel slots and replace dtype variants. Stacked draft attention intentionally holds K/V weights in both QKV and KV stacks after releasing original modules. Compiled trace/kernel workspace retention has no complete byte account here. | Model/drafter destruction after session drain releases module references; runtime compile caches may outlive them. Projection duplication is a documented speed tradeoff, not automatically unnecessary memory. |

Sources: [`DFlash2SpeculativeTokenIterator`](../../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/DFlash2SpeculativeTokenIterator.swift),
[`DFlash2DrafterState`, `GatedDeltaCapture`, `DFlash2ContextCache`](../../Vendor/mlx-swift-lm/Libraries/MLXLMCommon/DFlash2DrafterModel.swift),
[`DFlash2 draft model`](../../Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/DFlash2.swift).

## Allocator, model lifecycle and host data

| Family / owner | Size, bound and aliasing | Release / evidence status |
| --- | --- | --- |
| Target and drafter immutable weights; `LLMActor` containers and active Model Sessions | Actual parameter-array bytes plus auxiliary caches; checkpoint file lengths include metadata and can differ from loaded quantization/layout. `modelWeightBytes` is recorded during target verification before later projection stacking; draft array bytes are read at request preparation. They are not a unique physical census. | Loaded features require residency through all active consumers. The ordinary path drains completion, flushes SSD via `AgentEngine`, then unloads actor state. Reentrant unload protects a newly installed container. No active-model unloading is proposed as an optimization. |
| Weight-loading/conversion/staging; factory locals and PARO preparation | Source arrays, converted arrays, quantization/stacking outputs, prepared-file buffers and compiled/module caches can overlap. Exact peak depends on checkpoint loader and layout; no model-file-size multiplier is established. | Required loading/evaluation consumers must finish before old locals/module references disappear. This capture uses the existing affine checkpoint; PARO conversion and model-replacement failure peaks remain explicit gaps. |
| Reusable MLX allocator buffers; global allocator | `LLMActor.Defaults.cacheLimitMB` is 2,048 at this revision, not a live-array cap. Clear points exist at request/prefill boundaries, speculative prefill completion and projection stacking. Retained free buffers can avoid allocation churn. | `Memory.clearCache()` reclaims eligible free buffers, not still-owned arrays or unfinished GPU work. Do not sum cached bytes with footprint or infer a leak from unchanged process footprint. Required GPU workspaces and driver/compile caches remain separate unknowns. |
| Conversation render/token data; request keying, `RenderTokenCache`, Emitted Path Index | Render cache has one entry containing UTF-8 bytes, `[Int]` ids and digests; on arm64 the token element payload is `8*N` bytes before capacity/headers. It is count-bounded, not independently byte-capped. Emitted Path Index has a 32 MiB id-byte budget; dictionary/hash metadata and Swift capacity are additional. Temporary prefix/suffix arrays may share COW storage until mutation. | Entry replacement/unload reset releases render state; fingerprint index eviction/replacement/unload clears ids. Complete history is required behavior; measure repeated-copy/transient overhead without silently truncating it. |
| Generation accumulation, HTTP inbound/serialized JSON and stream buffers | Request/response strings, parsed objects, token lists, tool deltas and diagnostic pretty-print copies scale with input/output. Default stream buffering and retained handles require consumer-specific investigation; no blanket fixed-memory claim is made. | Last response/leaf-store/parser consumer and request completion release transient owners. Slow consumers, failed streams and retained finished handles require separate bounded probes. |
| UI activity traces; `ServerGenerationLog` | At most 20 traces; individual text spans have 64 KiB head + 192 KiB tail budgets plus hysteresis. Trace-count and per-span limits do not prove a global byte bound on inbound captures, span counts, tool data or pending appends. | Oldest trace eviction, clear and terminal cancel-action removal release owners. Existing bounded display behavior is preserved; remaining whole-history/slow-consumer retention is a host-data investigation. |
| Diagnostic/log queues and disk records; `RotatingJSONLWriter`, file sinks, `HTTPRequestLogger` | Diagnostic day files rotate at 8 MiB with seven retained days. The async writer captures encoding closures/events; disk retention limits do not bound queue backlog. HTTP request logging temporarily holds parsed JSON, pretty JSON and header Data. | Serial writer consumes queued events; request logging writes then releases locals. Slow/disk-failure behavior and byte growth need measurement before asserting bounded host memory. Diagnostic metadata must never materialize synthetic model tensors. |

Inbound UI capture already keeps at most 8 KiB head + 24 KiB tail per message
and 512 KiB of message content per trace. These are retained-content bounds;
flattening before the cap, object overhead and pending/span/tool storage still
need separate accounting. MTP, Batch Engine and vision-specific scratch are
unmapped variants, as are complete quantized/rotating cache layouts; the
production campaign below cannot close those gaps.

Sources: [`LLMActor`](../../tesseract/Features/Agent/LLMActor.swift),
[`AgentEngine.unloadModel`](../../tesseract/Features/Agent/AgentEngine.swift),
[`RenderTokenCache`](../../tesseract/Features/Server/RenderTokenCache.swift),
[`EmittedPathIndex`](../../tesseract/Features/Server/EmittedPathIndex.swift),
[`GenerationAccumulator`](../../tesseract/Features/Agent/Core/GenerationAccumulator.swift),
[`ServerGenerationLog`](../../tesseract/Features/Server/ServerGenerationLog.swift),
[`HTTPRequestLogger`](../../tesseract/Features/Server/HTTPRequestLogger.swift),
[`RotatingJSONLWriter`](../../tesseract/Features/Server/Telemetry/RotatingJSONLWriter.swift),
[`PromptCacheDiagnosticsFileSink`](../../tesseract/Features/Server/Telemetry/PromptCacheDiagnosticsFileSink.swift).

## SSD hydration, demotion and payload staging

This section is a source inventory at repository revision
`b1f0dca32ec9baead778221a8001fe4aa59e1850`, with preserved evidence identified
separately below. It does not establish current production peaks, physical bytes
freed, or measured optimization benefits. Current validation and production
observations are listed separately below.

The constraints are [Deferred Payload Extraction, Snapshot Hydrating and Segment
Chain terminology](../../CONTEXT.md), the [prefix-cache architecture](../../ARCHITECTURE.md),
[ADR-0001](../adr/0001-ssd-hydration-handle-stays-off-main.md),
[ADR-0019](../adr/0019-recoverable-eviction-leaf-home-guarantee.md) and
[ADR-0064](../adr/0064-leaf-handoff.md). Hydration and extension detachment remain
inside the Metal-affine Model Session. A full payload may retain immutable body
arrays, but a running generation cannot own those arrays while the writer reads
them. The Leaf Home Guarantee bypasses incidental pending limits only after a
lease has been checked in or rewound.

### Allocation families and byte meanings

Let `B` be one payload's serialized array bytes, `L` its largest layer's array
bytes, `H` its encoded JSON header bytes, and `F` the total bytes of the segment
files selected for a hydration. These count logical content. They do not include
allocator capacity, alignment, shared parent backings, VM residency or page-cache
effects. Array shape products use the array's actual dtype; the generic formula
is `sum(array.nbytes) = sum(product(shape) * dtype.size)`.

| Allocation family | Owner and aliases | Byte formula / physical uncertainty | Allocation and release boundary |
| --- | --- | --- | --- |
| Full deferred payload array references | `SnapshotPayload.LayerSource` retains its materializer closure, which retains `ServerCompletion.DeferredLayers`. Each owed layer references `HybridCacheSnapshot.layers[].state`. The snapshot/RAM body or other snapshot views may own the same MLX backings. | `B = sum(state.nbytes)`. The additional payload starts as references, not another `B`-byte tensor copy. A prefix view can retain a larger backing than its `nbytes`, and multiple owners cannot be summed as physical bytes. | `ServerCompletion.deferredPayload(for:extending:)` with no validated extension reads shape/dtype facts and retains array handles. During `DeferredLayers.materialize()`, each layer is removed from `owed`, copied, then its local source references go out of scope. Other owners and allocator caching can prolong physical residency. |
| Detached extension arrays | `DeferredLayers` owns independent attention suffix arrays and independent whole-state copies of recurrent, rotating and chunked layers. The array wrapper lists used during extraction share those detached backings. | `B = sum(sliceable suffix nbytes) + sum(non-sliceable whole-state nbytes)`. For a normal K/V pair of shape `[batch, heads, tokens, headDim]`, suffix bytes are `2 * batch * heads * (headOffset - baseOffset) * headDim * itemSize`; use actual state arrays for quantized/mixed layouts. | Every retained extension array is deep-copied and one `eval(detached)` completes on the Model Session before payload return. The original body and these detached arrays overlap. Materialization later replaces the latter with host bytes layer by layer. An invalid/near-full extension can degrade to a full payload; inspect `payload.extending`, not requested intent. |
| Materialized host array data | `SnapshotPayload.LayerSource.ready([LayerPayload])` owns each `ArrayPayload.data`. Struct copies of `SnapshotPayload` share the same reference-type `LayerSource`; they do not rematerialize. | Sum of `Data.count` is `B`. `asData(access: .copy)` allocates independent contiguous host bytes even on unified memory. Metadata/list storage is extra. | `DeferredLayers.materialize()` calls `asData(access: .copy)` for every array. Host data remains through encoding and file I/O, and until the last `LayerSource` owner releases it; completing the writer does not release copies held elsewhere. A sole-owner, compact demotion has approximate source-plus-host *logical content* `B + up to L` during this conversion, not zero overlap. This is not a physical peak bound. |
| Encoded file staging | `encodePlaceholderContainer(payload:descriptor:)` retains a list of existing blobs and creates a separate contiguous `out: Data`; `SSDSnapshotStore.writePayload` owns the returned container. The blob list aliases existing Data values. | Encoded count is exactly `8 + H + B`. At encoder completion the materialized payload's `B` and the encoded container's `8 + H + B` are simultaneously live logical content. Header encoding and capacity overhead are additional. A full payload may also coexist with the original resident MLX body. | `out` is reserved and populated by appending all array blobs before any file write. It is required through the chunked write’s last consumer. Source scope alone does not establish whether ARC releases it before later synchronize/close/rename operations or only at return/throw. The file writer's 1 GiB `Data(bytesNoCopy:deallocator:.none)` chunks borrow this container; chunking does not bound the full staging allocation. |
| Hydration file buffers and parsed metadata | `loadSync`/`loadSyncPrefix` own `[Data]` for all selected files; `decodeSegmentChain`'s parsed tuples retain those Data values and decoded headers. | `F = sum(selected file lengths)`, including inherited segments and headers. `.mappedIfSafe` is a requested read strategy, not a promise of zero physical memory or zero copying. Residency depends on touched pages and Foundation/OS behavior. Header objects and token-path metadata are extra host allocations. | Every selected file is read before chain composition. Data references remain through `decodeSegmentChain` return/throw. Full load selects the chain; prefix load selects only the files through the historical boundary. Interrupted reads release local references and preserve the backing chain. |
| Hydration constituent MLX arrays and concatenation graphs | `materializeLayerArrays` constructs fresh MLX arrays from file blob slices. Per-layer `pieces` supplies `concatenated(..., axis: -2)`; `snapshotLayers` retains merged results and their unevaluated dependencies. | `memoryBytes = sum(byteSize of contributing arrays)`. For whole-state layers only the last whole copy and later suffix contributors count; redundant earlier recurrent states in `F` are skipped. This does not count resident mapped pages, metadata, backing capacity, or lazy concatenation source/output overlap. | Arrays are allocated during composition in the Model Session. Multiple suffix pieces create a lazy concatenation. There is **no explicit `eval` inside `decodeSegmentChain`**; local `pieces` scope ending does not prove that dependent graph inputs have been released. The later `HybridCacheSnapshot.restore` deep-copies every restored array and hoists `eval(copiedArrays)`, so hydrated snapshot, graph inputs/results and live restore copies may overlap. The comment claiming about one snapshot plus one layer is an unmeasured expectation, not an established upper bound. |
| Queue, manifest and chain bookkeeping | `SSDSnapshotStore.pending`/current writer item own payloads, descriptors, body-access handles and extension-transfer claims. `SnapshotLedger` owns resident descriptors, partition metadata and chain ownership. `LeafBodyAccess` holds scalars and weak materialization probes, not payloads or cache arrays. | Queue/descriptors include token paths and segment metadata in host memory, not `B`. `PersistedSnapshotDescriptor.bytes` counts own array payload; `totalBytes` includes inherited segment array payloads. Neither includes the binary header or exact on-disk allocation. | Pending items leave the queue at writer selection but their payloads remain in the in-flight item. Resident descriptors retain no model tensors. Transfer claims shield the base until commit/drop and have a deinit backstop. Completed materialization can end body exclusion before file durability. |

Sources: [`SnapshotPayload`](../../tesseract/Features/Server/SnapshotManifest.swift),
[`ServerCompletion.deferredPayload` / `DeferredLayers`](../../tesseract/Features/Server/ServerCompletion.swift),
[`encodePlaceholderContainer`](../../tesseract/Features/Server/PlaceholderContainer.swift),
[`SSDSnapshotStore`](../../tesseract/Features/Server/SSDSnapshotStore.swift),
[`HybridCacheSnapshot.deepCopyState` / `restore`](../../tesseract/Features/Server/HybridCacheSnapshot.swift),
[`LeafBodyAccess`](../../tesseract/Features/Server/LeafLease.swift),
[`SnapshotLedger`](../../tesseract/Features/Server/SnapshotLedger.swift).
The resolved MLX source was also inspected locally at dependency revision
`6058402c676de25560051acda772e80d86d696d1`, matching
[`Package.resolved`](../../tesseract.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved):
[`MLXArray.asDataCopy/asData`](https://github.com/spokvulcan/mlx-swift/blob/6058402c676de25560051acda772e80d86d696d1/Source/MLX/MLXArray%2BBytes.swift)
allocates `Data(count: nbytes)` after `self.eval()`, and
[`MLXArray(Data,shape,dtype:)`](https://github.com/spokvulcan/mlx-swift/blob/6058402c676de25560051acda772e80d86d696d1/Source/MLX/MLXArray%2BInit.swift)
calls `mlx_array_new_data`. These links identify the inspected primary source;
the audit did not fetch a newer library revision.

### Pending accounting is not a physical memory bound

`SSDSnapshotStore.pendingBytes` increases by `payload.totalBytes` at enqueue and
remains charged after `popNextPending` removes the item from the waiting queue.
`releasePendingBytes` runs on terminal writer outcomes. Consequently
`PrefixCacheManager.memoryTelemetryFacts()["ssdPendingPayloadBytes"]` includes
the in-flight writer, while `ssdPendingPayloadCount` counts only waiting items.
A count of zero can coexist with a nonzero byte charge. The ledger and queue
observations use different locks; `diagnosticsSnapshot()` explicitly permits a
mixed pre/post-commit view. Neither pending bytes nor tree bytes can simply be
added to MLX active memory or process footprint.

The production `SSDPrefixCacheConfig.withAutoPendingCap` cap is
`min(4 GiB, physicalMemory / 16)` (3 GiB on a 48 GiB machine). An individual
nonmandatory payload larger than this is rejected. Backpressure first removes
queued deferrable writes, then other queued nonmandatory writes. Mandatory
guarantee writes bypass individual size rejection, cannot be backpressure victims
and cannot be made deferrable. They still require valid metadata/partition/base
and cannot violate a Leaf Lease.

There is another source-visible exception to a strict pending cap:
`enqueueApplyingBackPressure` always appends the incoming item after its victim
passes. A popped in-flight item is still charged but cannot be a queued victim.
An individually eligible nonmandatory incoming payload can therefore push the
combined charge above the cap even without mandatory writes. This is a candidate
for measurement and accounting clarification, not evidence of an unbounded
production leak. The encoded `8 + H + B` container is not charged at all.

`PrefixCacheManager.demoteBeforeDrop` does not request `mandatory: true`; its
accepted deferred full payload can retain a victim after tree `dropBody` removes
the logical RAM charge. A falling `treeSnapshotBytes` is therefore not evidence
of immediate physical relief. It skips a new write for an existing Snapshot Ref
or Chain-Prefix Restore point. Admission refusal still reaches the subsequent
body drop in `evictToFitBudget`, so cap/refusal outcomes must be recorded when
testing Recoverable Eviction rather than assuming every attempted demotion was
accepted. See [`PrefixCacheManager`](../../tesseract/Features/Server/PrefixCacheManager.swift),
[`TieredSnapshotStore.admitSnapshot`](../../tesseract/Features/Server/TieredSnapshotStore.swift)
and [`SSDPrefixCacheConfig`](../../tesseract/Features/Server/SSDPrefixCacheConfig.swift).

### Ownership, interruption and release boundaries

1. **Extraction and enqueue.** Full extraction performs no full array copy. The
   source must already be evaluated and detached from active generation; moved
   snapshot capture evaluates state views, and extension extraction evaluates
   its private copies. `SnapshotPayload.layers` itself can trigger
   materialization on its caller, so it must not be read by new MainActor or
   phase-observation code. The one-shot `LayerSource` lock serializes this copy;
   probing it from another thread can wait for the copy to complete.
2. **Lease exclusion.** `TieredSnapshotStore.admitSnapshot` supplies a body-access
   handle only for unmaterialized full payloads. `popNextPending` atomically
   claims `LeafBodyAccess.beginRead` before removing such an item. An existing
   lease keeps it queued and charged; forced flush, write class and the 30-second
   activity-gate timeout cannot override it. A suffix cannot overtake its blocked
   base, but unrelated work can proceed. After return, the existing 500 ms writer
   recheck resumes work. Production move checkout rejects pending full payloads;
   tree-side tests that acquire a lease over a queued full payload exercise a
   separate exclusion boundary, not normal production eligibility.
3. **Materialization and file write.** `processPendingItem` releases the body-read
   claim immediately after all arrays have been copied/released, before file I/O.
   Its `defer` releases the claim on earlier drop paths. A disk-full retry reuses
   the materialized array Data but calls `writePayload` again, rebuilding its
   encoded container. Success, budget refusal, lost extension base, tombstone or
   I/O failure settle the pending charge and transfer claim on their respective
   branches. Actual host-data lifetime ends only after remaining payload owners
   release the shared `LayerSource`.
4. **Cancellation and deletion.** Request cancellation does not directly cancel
   the SSD writer; admitted durability work may outlive the request. Explicit
   `deleteSnapshot` removes a queued payload and charge immediately, or tombstones
   an in-flight write for the writer's pre-write/post-write/commit checks. It does
   not interrupt a synchronous host memcpy or `FileHandle` write already running.
   Hydration polls interruption between files and once before decoding; neither
   `decodeSegmentChain` nor its per-layer reconstruction takes the interruption
   closure. Cancellation arriving during composition therefore waits for that
   synchronous phase. Interrupted hydration preserves the chain; read/decode
   failure drops the broken chain and triggers `.hydrationFailure` cleanup.
5. **Unload and durable drain.** The production
   `AgentEngine.unloadModel` task awaits `LLMActor.drainServerCompletion`, then
   `prefixCacheAdmin.flushSSDWrites`, then `LLMActor.unloadModel`. Await
   `AgentEngine.awaitPendingUnload` when observing this path; the immediate
   "Model unloaded" log precedes completion of the task. The actor method by
   itself drains completion but does not flush SSD. `flushAsync` waits for writer
   drain cycles, persists the manifest, and bypasses only storage-activity
   deferral. It is not evidence of cache allocator release or callback settlement
   on the MainActor. `SSDSnapshotStore.deinit` intends to finish/cancel its writer;
   the weak task capture followed by `await self?.writerLoop()` needs an explicit
   weak-owner observation before claiming store deallocation at unload.

Sources: [`SSDSnapshotStore`](../../tesseract/Features/Server/SSDSnapshotStore.swift),
[`AgentEngine`](../../tesseract/Features/Agent/AgentEngine.swift),
[`LLMActor`](../../tesseract/Features/Agent/LLMActor.swift),
[`PrefixCacheAdmin`](../../tesseract/Features/Server/PrefixCacheAdmin.swift),
[`LeafBodyAccess`](../../tesseract/Features/Server/LeafLease.swift).

### Evidence already present and what it establishes

The following tests were read, not rerun:

- [`ServerCompletionExtractSnapshotPayloadsTests.extractionDefersTheHostCopyAndFixesTheByteTotal`,
  `totalBytesMatchesSumOfArrayNbytes`, `byteRoundTripMatchesSourceArrays`](../../tesseractTests/ServerCompletionExtractSnapshotPayloadsTests.swift)
  check deferred materialization, exact logical byte totals and content.
- [`LeafExtensionAdmissionTests.pendingExtensionPayloadRetainsNoBodyArray`,
  `detachedExtensionPayloadWritesTheSameBytes`](../../tesseractTests/LeafExtensionAdmissionTests.swift)
  compare real backing addresses of hybrid attention/recurrent arrays and verify
  empty owed arrays after writer drain, without changing serialized content.
- [`LeafLeaseTests.pendingFullWriterWaitsForLeaseEvenWhenFlushForcesDrain`,
  `writerFailureReleasesReadClaim`, `mandatoryAdmissionReportsLeaseRefusalThenPersistsAfterReturn`,
  `leasedBaseDefersItsSuffixButNotUnrelatedWrites`](../../tesseractTests/LeafLeaseTests.swift)
  pin exclusion, pending-charge settlement, failure cleanup and required ordering.
- [`SSDSnapshotStoreTests.writerMaterializesADeferredPayloadBeforeTheWrite`,
  `mandatoryWriteBypassesThePendingSizeCap`, `backPressureNeverDropsMandatoryPendingItems`,
  `flushAsyncDrainsWriterAndPersistsManifest`, `writerCommitsPayloadPastIntMaxBytes`](../../tesseractTests/SSDSnapshotStoreTests.swift)
  exercise off-main materialization, accounting policy, flush durability and the
  large single-write syscall regression. They do not establish a streaming
  encoding memory bound. [`SSDWriteEagernessTests`](../../tesseractTests/SSDWriteEagernessTests.swift)
  cover busy-gate deferral and its flush/write-through exceptions.
- [`SnapshotDemotionTests`](../../tesseractTests/SnapshotDemotionTests.swift)
  exercise recovery, already-backed skip and stale-recency preservation;
  [`ServerCompletionSSDRestoreTests`](../../tesseractTests/ServerCompletionSSDRestoreTests.swift)
  exercise the completion restore path with its fixture rather than production
  model memory.

The preserved [tree-side lease evidence](../../benchmarks/leaf-lease/2026-09-12/README.md)
and [review follow-up](../../benchmarks/leaf-lease/2026-09-12-review/README.md) report
those bounded tests at their recorded revisions. They explicitly separate array
release from whole-process footprint, identify the pending-byte/count asymmetry,
and state that delayed `afterRelease` samples are not SSD-drain guarantees.
The review run records 890 focused passes plus one isolated ownership test;
this is historical validation, not validation of this audit's checkout.

The preserved [AlphaTuner incident](../../benchmarks/incidents/2026-09-12-alpha-tuner/README.md)
and [raw events](../../benchmarks/incidents/2026-09-12-alpha-tuner/incident-events.jsonl)
contain an SSD extension materialization of 204,472,320 bytes taking 7,508.058 ms,
completed at `2026-09-12T21:05:49Z`. This supplies a real pressured timing, not an
isolated SSD allocation peak. The incident's repeated approximately 16.7 GB
active-memory pattern continued afterwards and was attributed to AlphaTuner;
that small pending suffix cannot be reassigned as its cause. Its binary/source
provenance caveat remains applicable.

### Prioritized gaps and smallest next probes

| Priority / candidate | Smallest useful probe | Proposed owner and acceptance boundary |
| --- | --- | --- |
| 1. Full container staging overlaps materialized payload data; the source establishes an extra `B + H + 8` logical buffer, but not its measured resident peak. | At `processPendingItem` materialization begin/end and `writePayload` encoding begin/end/write end, record scalar payload ID, full/extension class, `B`, encoded `Data.count`, writer phase and existing process/MLX readings. First observe one naturally admitted full leaf and one suffix on the authorized production model. Do not retain arrays/Data in the event or force extra evaluation. | #505 SSD subwork. Consider bounded container encoding/file streaming only after observing its contribution. Preserve current file bytes, atomic temp/write/synchronize/rename, large-write chunk limit and error/retry behavior. This candidate is separate from #501. |
| 2. Pending charge conflates queued/in-flight and source-reference/host-data phases; encoded bytes and mandatory exceptions are invisible. | Under the queue lock track scalar queued and in-flight payload bytes/counts, mandatory bytes, full/extension bytes and cap overshoot; expose body-source bytes and host-data bytes as ownership categories rather than physical sums. Sample before enqueue, after pop, after materialization, after terminal charge release. Update immutable scalar phase state on the writer; do not call `payload.layers` or a contended materialization probe from telemetry. | #505 SSD subwork. Required guarantee bypass and enqueue-before-delete remain intact. Correlate with demotion/refusal IDs so a tree-byte drop is not misreported as released resident memory. |
| 3. Hydration's lazy concatenation and subsequent restore may retain more than the stated per-layer working set. | On the existing Model Session path, record selected file bytes/segment count, contributing array bytes and per-layer largest contributor sum immediately before/after `decodeSegmentChain`, then use the existing restore `eval` boundary to sample evaluated MLX/process state. A forced extra `eval` would change lifetime and timing, so treat any such experiment separately. | #505 SSD hydration subwork. #501 may consume the same before/after-restore samples when separating warm-prefill/DFlash2 from restore, but should not own a duplicate SSD investigation. Byte-exact full/chain-prefix restore and existing cancellation semantics are required. |
| 4. Physical relief at demotion, explicit deletion and unload is not attributable from current logical counters. | Correlate body-drop, payload materialization and terminal writer IDs with process/MLX observations after an explicit flush and callback settlement. Add weak scalar observations of the store/source lifetimes at existing teardown rather than retaining objects for diagnostics; specifically test whether the suspended `writerLoop` keeps the store alive. | #505 SSD lifecycle subwork, sharing unload measurements with the main inventory. No leak claim or allocator-policy change until the surviving owner is identified. |
| 5. Hydration cancellation during composition has no intra-decode poll, and mandatory/write-through traffic can overlap hydration despite the activity gate. | Timestamp cancellation signal, decode begin/end, writer phase and request quiescence on one bounded cancel; record write class and hydration-active flag on the same timeline. | #505 SSD responsiveness subwork. Decide whether changes are justified by measured latency/memory. Preserve the Leaf Home Guarantee and off-MainActor hydration discipline. |

These are investigation candidates, not authorization to change restore,
rewind, checkpoint, writer ordering, or mandatory admission semantics.

## New production observations on the 48 GiB Mac

The [preserved campaign and reproduction notes](../../benchmarks/allocation-inventory/2026-09-12/README.md)
include every attempt, including failures, and the
[successful scalar summary](../../benchmarks/allocation-inventory/2026-09-12/production-final/summary.json).
All sizes below use binary GiB/MiB unless explicitly stated as bytes.

The final run used the current Release source plus the archived diagnostic diff,
Apple M3 Max / 48 GiB, Qwen3.8-27B affine 4-bit weights, the installed DFlash2
checkpoint loaded at 4-bit, unquantized KV, temperature zero and a 128-token
output ceiling. Every successful response confirmed `dflash2Engaged=true`;
`mtpLoaded=false` and `visionMode=false` reflect the existing automatic selection.
Target parameter accounting was 15,132,802,048 bytes before projection stacking;
request-time draft parameter arrays were 1,112,353,380 bytes. These are logical
parameter totals, not unique physical residency. Target/draft file and binary
hashes are preserved. No other model test or build ran concurrently.

The final plan sampled every 250 ms, stopped at critical/unknown pressure,
32 GiB app footprint, 2 GiB additional system swap, 180 seconds per HTTP response or
900 seconds overall, with a separate 60-second release wait. Seven scenarios finished in 171.809 seconds: six completed
responses and one intentional disconnect. No resource stop fired. All 641 OS
samples reported normal pressure; sampled swap stayed below the 6,747,062,272-byte
starting value. Earlier runs stopped at warning pressure or 512 MiB swap growth.
Their differing OS state and allowances prevent treating the final run as a
controlled performance or memory comparison. The first startup attempt sent
no inference because its runner incorrectly waited for automatic model loading;
the corrected runner waits for model availability before sending the first request.

### Loading transient versus inference

The largest external footprint sample was **31,871,681,024 bytes (29.683 GiB)**,
inside the interval after draft loading and before projection stacking completed.
At completed load, footprint was 17,967,409,712 bytes (16.733 GiB), active MLX
16,670,850,332 bytes and cached MLX zero. The existing stacking path evaluates
its new arrays and clears reusable allocator buffers; this change did not add
that clear. Phase samples locate the transient, but do not identify every backing
allocation or prove that the difference is recoverable without a latency tradeoff.

| Phase boundary | Process footprint GiB | Active MLX GiB | Cached MLX GiB |
| --- | ---: | ---: | ---: |
| Target loaded | 19.478 | 14.489 | 4.365 |
| Draft loaded | 23.518 | 15.501 | 7.391 |
| Projection stacking complete | 16.761 | 15.526 | 0 |
| Model load complete | 16.733 | 15.526 | 0 |

After completed loading, the largest 250 ms OS sample was 20.144 GiB. The denser
request-boundary/periodic telemetry observed 20.543 GiB during resend; these
sampling schemes catch different instants, so neither is a guaranteed maximum.
The 23,719,247,460-byte lifetime MLX peak occurred during loading and stayed
unchanged throughout the later requests. It must not be reported as a new
per-request allocation peak. Final ten-second idle samples ended at
18,964,311,104 bytes (17.662 GiB) of process footprint.

### Request behavior and capacity

“Cold” means a fresh app/model load and no reusable snapshot for this synthetic
prefix, not an erased SSD store. Existing persisted metadata was left intact.
The first lookup reported `missNoSnapshotInPrefix`; all six later lookups used
Leaf Handoff with no SSD hydration. Client timing includes HTTP work and, for
the cold request, loading. “First delta” is the first content/reasoning/tool delta,
not a calibrated model-only TTFT. No reference build or output-correctness
comparison was performed.

| Scenario | Prompt / cached tokens | First delta s | Response s | Request telemetry peak footprint GiB | Last external settle sample GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold | 3,430 / 0 | 22.231 | 25.342 | 18.424 (excludes loading) | 16.809 |
| Warm | 3,577 / 3,558 | 0.357 | 1.382 | 18.088 | 17.110 |
| Grow | 7,009 / 3,631 | 16.235 | 18.620 | 19.420 | 17.438 |
| Cancel growth | 11,954 / 7,136 from lookup; no usage response | — | 3.008 to disconnect | 20.409, including cleanup | 17.380 |
| Resend growth | 11,954 / 7,136 | 25.216 | 27.734 | 20.543 | 17.864 |
| Warm repeat 1 | 12,106 / 12,082 | 0.518 | 2.038 | 18.615 | 17.656 |
| Warm repeat 2 | 12,219 / 12,198 | 0.489 | 1.323 | 18.701 | 17.662 |

The 16 full-attention layers exposed valid versus allocated extents. Before
cancelled growth they held 467,664,896 logical bytes in 469,762,048 array bytes.
After rewind, the next resend restored the same 7,136 logical rows in
788,529,152 array bytes: **320,864,256 bytes (306 MiB) of unused extent**.
This establishes retained shape capacity across rewind, not a leaked owner or
306 MiB of proven recoverable physical memory. Resend reused those arrays and
the final successful leaf had only 1,507,328 unused extent bytes. The measured
recurrent state and independent rewind backup were each 153,944,064 bytes
(146.8125 MiB); the backup fell to zero after rewind/check-in.

All six leases had matching end events, with one `leafRewind` to offset 7,136.
Cancelled request-owned cache facts reset to zero at `rewoundLeaf`. Successful
handoff events retained the `capturingLeaf` byte facts with that freshness tag
and `requestCacheLayerCountAfterCapture=0`; they describe the moved leaf, not a
second request-owned cache. The last three after-release active-MLX readings
were identical at 17,788,591,204 bytes while the conversation grew. These few
cycles do not prove a general leak bound or unload/reload behavior.

The client disconnect occurred at 3.008 seconds with no output delta. Server
`cancelSignal` was not observed until 25.958 seconds, after `prefilled`; request
terminal was at 26.042 seconds and after-release at 27.089 seconds. Request
telemetry begins after loading, which was already complete for this scenario.
The trace establishes a roughly 23-second disconnect-to-observed-cancellation
gap, but cannot separate network delivery, handler setup/scheduling and prefill
cancellation cooperation. Follow up with existing transport/handler and prefill
boundaries under #501; do not attribute the delay to Leaf Rewind itself, which
followed generation quiescence promptly.

### SSD staging observed without changing admission

Six naturally admitted payloads were materialized and committed: one full
payload and five extensions. The full payload contained 387,121,152 bytes and
its encoded container 387,153,971 bytes. Both logical buffers coexist at encode
completion. Footprint rose by 387,416,328 bytes between its encode begin/end
samples while active/cached MLX stayed unchanged, consistent with the separate
host container. This correlation supports the source-established staging
candidate but is not an allocation-stack proof of exclusive attribution.

The largest extension contained 478,085,120 bytes; its container was
478,158,633 bytes. Other encode deltas varied, including one with no observed
footprint increase despite a new 158,409,501-byte container. Reuse, compression,
other releases and sample timing prevent treating a `Data.count` as an equal
resident delta. No artificial eviction, hydration, explicit flush/drain or
streaming encoder change was made. A quiet interval and `afterRelease` are not
SSD-drain guarantees, and carried pending counters can be stale.

### Diagnostic validation and cost

The [focused validation](../../benchmarks/allocation-inventory/2026-09-12/validation/focused-tests-summary.txt)
passed **40 tests in three suites** with allocation diagnostics enabled:
`RequestMemoryTelemetryTests`, `SSDSnapshotStoreTests` and
`AgentEngineLoadStateTests`. Coverage includes trimmed capacity versus valid
bytes, clearing carried bytes on request release, deferred materialization,
large SSD writes and load-state behavior. The Release build, strict Swift format
lint, Python syntax and repository doc/link checks passed. Historical checkout
and incident checksum manifests were also verified; their tests were not
rerun or counted as current passes.

The new scalar allocation events never retain model arrays, materialize payloads
for observation or force evaluation. In the successful run, 30 such events
reported 0.020 ms median / 0.101 ms maximum observation cost, 0.785 ms total.
The 641 external OS reads took 0.065 ms median / 0.997 ms maximum, 60.121 ms total.
These timings exclude logging dispatch/serialization/I/O, Python bookkeeping,
existing request telemetry and observer effects on scheduling. They do not
establish zero overhead or substitute for a diagnostics-off comparison.

## Follow-up ownership and validation boundaries

| Area | Next observation needed | Existing issue / constraint |
| --- | --- | --- |
| Warm prefill, DFlash2 working state and attention capacity | Correlate restored length, valid/capacity array bytes, recurrent backup and iterator round lifetimes with evaluated MLX and OS samples. Separate restore, warm prefill, generation, finalization and settled idle. | [#501](https://github.com/spokvulcan/tesseract/issues/501). Preserve active DFlash2, exact rewind, checkpoint and ADR-0009 behavior. |
| Loaded Leaf Handoff parity and cleanup | Compare copied restore against handoff at identical inputs/settings, continuation cache/logit bytes, TTFT, final-token tail and terminal ownership; cover success, failure and cancel/resend. | [#480](https://github.com/spokvulcan/tesseract/issues/480). This inventory's single-build scalar campaign is not a parity gate. |
| Model loading and projection stacking | Observe target load, draft load and stack construction separately, then identify the retained source/destination/cache owners inside any large transient. | [#505](https://github.com/spokvulcan/tesseract/issues/505). Keep checkpoint precision, stacked execution and active draft configuration unchanged while measuring. |
| SSD host staging, hydration, demotion and writer lifetime | Use the smaller probes listed above; demonstrate which owners survive each existing materialize, write, restore and flush boundary. | #505 SSD work, sharing restore observations with #501. Preserve recoverability, file bytes and lease exclusion. |
| Host history, streaming and diagnostic queues | Measure capped retained content separately from pre-cap copies, object overhead and slow-consumer backlog; confirm replacement/unload releases. | #505 host/lifecycle work. Required history and behavior stay intact. |
| Disabled AlphaTuner and budget allowances | Any synthetic tuning redesign needs bounded isolated validation; reserve calibration needs actual remaining workspace peaks across supported workloads. | [#504](https://github.com/spokvulcan/tesseract/issues/504) and [#238](https://github.com/spokvulcan/tesseract/issues/238). Do not rerun the disconnected production tuner or claim the provisional allowance as savings. |

A useful validation sequence is: isolated accounting/ownership tests, one
bounded production run at current settings, then controlled repeat/comparison
only where the first result identifies a question. Repeated small growth and
cancel/resend can expose accumulating ownership, but cannot establish a 45k,
75k or 93k context peak. Larger contexts, explicit SSD hydration and drain,
model replacement/unload, allocation-stack capture, alternate cache layouts,
MTP, Batch Engine and vision need their own measured coverage. No model
precision, context/history requirement, speculative mode or correctness copy
was reduced to obtain the observations in this change.
