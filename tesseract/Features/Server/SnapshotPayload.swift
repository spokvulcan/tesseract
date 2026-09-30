//
//  SnapshotPayload.swift
//  tesseract
//
//  The **Snapshot Payload**: one snapshot's SSD byte form, and the one place
//  that builds it. The value below is what crosses to the SSD writer. The
//  builders after it are **Deferred Payload Extraction** for whole, extension
//  and view payloads, together with the extension worth-it gate, and the dtype
//  table names each array's dtype on disk in both directions. The container
//  framing around the bytes lives in `PlaceholderContainer.swift`.
//

import Foundation
import MLX
import os

// MARK: - In-memory transport payload (Sendable, not Codable)

/// The bytes of a `HybridCacheSnapshot` on their way to the SSD tier, plus
/// the metadata needed to reconstruct the snapshot on hydration. The only
/// shape that crosses the LLMActor → `SSDSnapshotStore` boundary.
///
/// The bytes are owed, not carried (**Deferred Payload Extraction**): the
/// extraction edge builds the value from the snapshot's arrays with only
/// their byte total; the SSD writer prepares borrowed host views right before
/// writing them. Each view owns its evaluated array for the lifetime of its
/// Data. A full payload shares arrays with the body; an extension owns detached
/// arrays. The writer consumes borrowed layers, releasing each after its bytes
/// are written. Ready host-byte payloads (fixtures) remain repeatable.
///
/// Not `Codable`. The payload is the input to the placeholder-container
/// writer, which consumes it byte-for-byte inside the writer task and
/// never round-trips it through JSON. Making this `Codable` would
/// encourage callers to persist the payload without going through that
/// path, which is exactly what the Metal-affinity rules forbid.
nonisolated struct SnapshotPayload: Sendable {

    /// One array's bytes, either owned host Data or a view retaining its MLX owner.
    struct ArrayPayload: Sendable {
        /// Contiguous bytes. Borrowed production views retain the evaluated
        /// array through Data's deallocator; ready fixtures own host storage.
        let data: Data

        /// MLX dtype name as reported by the vendor (e.g. `"bfloat16"`,
        /// `"int8"`, `"float16"`). Stored as a String so the payload
        /// type does not have to import the MLX enum.
        let dtype: String

        /// Array shape. Preserved for the safetensors header so the
        /// reader can reconstruct the array before MLX gets involved.
        let shape: [Int]
        /// The Data retains evaluated MLX storage until its bytes are written.
        let borrowsArray: Bool

        init(data: Data, dtype: String, shape: [Int], borrowsArray: Bool = false) {
            self.data = data
            self.dtype = dtype
            self.shape = shape
            self.borrowsArray = borrowsArray
        }
    }

    /// One `HybridCacheSnapshot.LayerState`, mirrored as a flat value
    /// type. The vendor type is `@unchecked Sendable` because it
    /// holds `[MLXArray]`; this mirror is genuinely `Sendable`.
    struct LayerPayload: Sendable {
        /// Cache class name matching the `savePromptCache` convention
        /// (e.g. `"KVCache"`, `"QuantizedKVCache"`, `"RotatingKVCache"`,
        /// `"ChunkedKVCache"`, `"MambaCache"`, `"ArraysCache"`).
        let className: String

        /// Per-array extracted bytes. One entry per
        /// `HybridCacheSnapshot.LayerState.state` element, in the
        /// same order.
        let state: [ArrayPayload]

        /// `HybridCacheSnapshot.LayerState.metaState`, verbatim.
        /// Stable `String` values that the vendor reconstructs from.
        let metaState: [String]

        /// Absolute token offset captured from
        /// `HybridCacheSnapshot.LayerState.offset`. Mirrors the
        /// vendor's explicit-offset contract.
        let offset: Int

        /// Non-nil when this layer's arrays hold only the suffix token
        /// range `(suffixBaseOffset..offset]` along the token axis — a
        /// **Leaf Extension Admission** sliced a sliceable-attention
        /// layer (`HybridCacheSnapshot.LayerState.Kind`). `nil` means the
        /// arrays are the layer's whole state (always for whole-state
        /// layers, and for every non-extension payload).
        let suffixBaseOffset: Int?

        init(
            className: String,
            state: [ArrayPayload],
            metaState: [String],
            offset: Int,
            suffixBaseOffset: Int? = nil
        ) {
            self.className = className
            self.state = state
            self.metaState = metaState
            self.offset = offset
            self.suffixBaseOffset = suffixBaseOffset
        }
    }

    /// The snapshot's absolute prompt token offset. Mirrors
    /// `HybridCacheSnapshot.tokenOffset`.
    let tokenOffset: Int

    /// Mirrors `HybridCacheSnapshot.checkpointType`. Kept as the enum
    /// (not the wire String) because the payload is in-memory only.
    let checkpointType: HybridCacheSnapshot.CheckpointType

    /// Non-nil when this payload is a **Leaf Extension Admission**: the
    /// sliceable layers carry only the suffix past `extending.baseOffset`
    /// and the SSD front door must validate-and-transfer the base's
    /// **Segment Chain**. `nil` for every full payload.
    let extending: SnapshotExtension?

    /// Sum of every `state` array's byte count across every layer, fixed
    /// at construction so the front door's `maxPendingBytes` check and the
    /// ledger's descriptor never need the bytes themselves. A deferred
    /// payload's materializer must produce exactly this many.
    let totalBytes: Int

    /// Whether this payload's arrays are the tree body's own. View and
    /// extension payloads own detached device arrays, so they never retain
    /// the body; a full payload retains it until the writer releases each
    /// borrowed layer. This is independent of the segment format.
    private let mayRetainBodyArrays: Bool

    private let source: LayerSource

    /// Per-layer payloads, in the same order as the vendor snapshot's
    /// `layers` array. Reading a deferred payload runs its materializer on
    /// the calling thread — once, the result is cached — so in production
    /// only the SSD writer reads this.
    var layers: [LayerPayload] { source.layers() }

    /// `true` once the bytes exist as `Data`: from construction for a
    /// ready payload, after the first `layers` read (or `materialize()`)
    /// for a deferred one.
    var isMaterialized: Bool { source.isMaterialized }

    /// Preparing borrowed Data does not detach a full payload from its body:
    /// `true` while a full payload still borrows a tree body array, `false`
    /// for view and extension payloads (their arrays are detached copies)
    /// and once every borrowed layer has been written and released.
    var retainsBodyArrays: Bool { mayRetainBodyArrays && source.retainsBodyArrays }

    /// The writer consumes one layer at a time, releasing its borrowed owner
    /// after the final chunk. In-memory fixture encoding remains repeatable.
    func consumeLayers(_ consume: (LayerPayload) throws -> Void) rethrows {
        try source.consumeLayers(consume)
    }

    func discardLayers() { source.discardLayers() }

    /// Observes pending array ownership without retaining the payload or its Data.
    var materializationProbe: @Sendable () -> Bool {
        let mayRetainBodyArrays = mayRetainBodyArrays
        return { [weak source] in
            !(mayRetainBodyArrays && (source?.retainsBodyArrays ?? false))
        }
    }

    /// Prepare deferred byte views now, once. This does not detach a borrowed
    /// full payload from its tree body; `retainsBodyArrays` tracks that lifetime.
    @discardableResult
    func materialize() -> Bool {
        source.materialize()
    }

    /// A payload whose bytes are already in hand.
    init(
        tokenOffset: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType,
        layers: [LayerPayload],
        extending: SnapshotExtension? = nil
    ) {
        self.tokenOffset = tokenOffset
        self.checkpointType = checkpointType
        self.extending = extending
        self.totalBytes = Self.byteCount(of: layers)
        self.mayRetainBodyArrays = false
        self.source = LayerSource(ready: layers)
    }

    /// A deferred payload: `materialize` produces the layers on the first
    /// read and must yield exactly `totalBytes` bytes.
    init(
        tokenOffset: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType,
        extending: SnapshotExtension? = nil,
        totalBytes: Int,
        retainsBodyArrays: Bool = true,
        materialize: @escaping @Sendable () -> [LayerPayload]
    ) {
        self.tokenOffset = tokenOffset
        self.checkpointType = checkpointType
        self.extending = extending
        self.totalBytes = totalBytes
        self.mayRetainBodyArrays = retainsBodyArrays && extending == nil
        self.source = LayerSource(deferred: materialize)
    }

    /// Sum of every `state` array's byte count across `layers`.
    static func byteCount(of layers: [LayerPayload]) -> Int {
        var total = 0
        for layer in layers {
            for array in layer.state {
                total += array.data.count
            }
        }
        return total
    }

    /// The one-shot holder behind `layers`: ready from construction, or
    /// produced by a materializer the first reader runs under the lock (a
    /// concurrent reader blocks until preparation finishes and sees the
    /// cached result). Reference semantics on purpose — copies of the
    /// payload share one materialization.
    private final class LayerSource: Sendable {
        private enum State {
            case deferred(@Sendable () -> [LayerPayload])
            case ready([LayerPayload])
        }

        private let state: OSAllocatedUnfairLock<State>

        init(ready layers: [LayerPayload]) {
            state = OSAllocatedUnfairLock(initialState: .ready(layers))
        }

        init(deferred materialize: @escaping @Sendable () -> [LayerPayload]) {
            state = OSAllocatedUnfairLock(initialState: .deferred(materialize))
        }

        var isMaterialized: Bool {
            state.withLock { state in
                if case .ready = state { return true }
                return false
            }
        }

        var retainsBodyArrays: Bool {
            state.withLock { state in
                switch state {
                case .deferred: return true
                case .ready(let layers):
                    return layers.contains { $0.state.contains { $0.borrowsArray } }
                }
            }
        }

        func discardLayers() {
            state.withLock { state in
                switch state {
                case .deferred: state = .ready([])
                case .ready(let layers):
                    if layers.contains(where: { $0.state.contains { $0.borrowsArray } }) {
                        state = .ready([])
                    }
                }
            }
        }

        func consumeLayers(_ consume: (LayerPayload) throws -> Void) rethrows {
            _ = materialize()
            // Ready host-byte fixtures remain repeatable. Only borrowed
            // device owners need a consuming write to release per layer.
            guard retainsBodyArrays else {
                for layer in layers() { try consume(layer) }
                return
            }
            while let layer = state.withLock({ state -> LayerPayload? in
                guard case .ready(var layers) = state, !layers.isEmpty else { return nil }
                let next = layers.removeFirst()
                state = .ready(layers)
                return next
            }) {
                try consume(layer)
            }
        }

        /// `true` when this call ran the materializer.
        func materialize() -> Bool {
            state.withLock { state in
                guard case .deferred(let materialize) = state else { return false }
                state = .ready(materialize())
                return true
            }
        }

        func layers() -> [LayerPayload] {
            state.withLock { state in
                switch state {
                case .ready(let layers):
                    return layers
                case .deferred(let materialize):
                    let layers = materialize()
                    state = .ready(layers)
                    return layers
                }
            }
        }
    }
}

// MARK: - Building a payload (Deferred Payload Extraction)

nonisolated extension SnapshotPayload {
    /// Worth-it gate for a **Leaf Extension Admission**: when the
    /// estimated suffix payload exceeds this fraction of the full
    /// payload (a model whose layers are mostly non-sliceable, or a
    /// near-root base), the leaf admits full — a "delta" that rivals
    /// the full write buys chain complexity for nothing.
    static let extensionMaxSuffixFraction = 0.9

    /// The validated, worth-it extension for `snapshot`, or `nil` when
    /// the payload should admit full. Pure metadata arithmetic — no
    /// array bytes move here. Which layers would slice is each layer's
    /// ``HybridCacheSnapshot/LayerState/Kind``, derived at capture; a
    /// whole-state layer (recurrent, rotating, chunked, or an attention
    /// layer the shape guard kept whole) counts whole.
    private static func validatedExtension(
        _ extending: SnapshotExtension?,
        for snapshot: HybridCacheSnapshot
    ) -> SnapshotExtension? {
        guard let extending,
            extending.baseOffset > 0,
            extending.baseOffset < snapshot.tokenOffset
        else { return nil }

        var fullBytes = 0
        var suffixBytes = 0
        let suffixFraction =
            Double(snapshot.tokenOffset - extending.baseOffset)
            / Double(snapshot.tokenOffset)
        for layer in snapshot.layers {
            let layerBytes = layer.state.reduce(0) { $0 + $1.nbytes }
            fullBytes += layerBytes
            switch layer.kind {
            case .sliceableAttention:
                suffixBytes += Int(Double(layerBytes) * suffixFraction)
            case .wholeState:
                suffixBytes += layerBytes
            }
        }
        guard fullBytes > 0,
            Double(suffixBytes) <= extensionMaxSuffixFraction * Double(fullBytes)
        else { return nil }
        return extending
    }

    /// Build the SSD payload for `snapshot` — **Deferred Payload
    /// Extraction**, so this call moves no array bytes on the calling
    /// thread. It settles the extension and fixes the byte total; the
    /// host views are prepared when the payload's `layers` are first read, on
    /// the SSD writer's task. Callable from the MainActor too (**Snapshot
    /// Demotion**'s extractor passes no extension, so that path reads
    /// shapes only): the arrays are evaluated deep copies, never live
    /// model state.
    static func extract(
        _ snapshot: HybridCacheSnapshot,
        extending: SnapshotExtension? = nil
    ) -> SnapshotPayload {
        deferred(for: snapshot, extending: extending).payload
    }

    /// `extract(_:extending:)` together with the box of arrays the
    /// payload still owes the SSD writer. What a pending payload retains
    /// is a memory-safety claim (ADR-0064: an extension payload retains
    /// no body array) that only a physical-address comparison can check,
    /// so the box is returned for tests; production reads the payload.
    ///
    /// For a **Leaf Extension Admission** every retained array is
    /// detached here, on the Metal-affine caller, by the layer's kind: a
    /// sliceable-attention layer's suffix past the base is sliced into
    /// its own contiguous device buffer, and every whole-state layer
    /// (recurrent, rotating, chunked — small next to the attention
    /// suffix) is deep-copied whole, all evaluated in one sync, so the
    /// later host view can borrow contiguous storage and never references
    /// the body it was built from. A full payload retains the body's own
    /// arrays: copying them is what the deferral exists to avoid.
    static func deferred(
        for snapshot: HybridCacheSnapshot,
        extending: SnapshotExtension? = nil
    ) -> (payload: SnapshotPayload, owed: DeferredLayers) {
        precondition(!snapshot.isPrefixView, "a view payload requires its Backing Leaf")
        let activeExtension = validatedExtension(extending, for: snapshot)
        return deferred(
            for: snapshot, layers: snapshot.layers, extending: activeExtension, detaching: false)
    }

    /// A full-format view payload owns every retained array independently
    /// of both the Backing Leaf and the view. Run inside the Model Session
    /// while the backer is protected; only the later host copy is deferred.
    static func deferred(
        for view: HybridCacheSnapshot, backingLeaf: HybridCacheSnapshot
    ) throws -> (payload: SnapshotPayload, owed: DeferredLayers) {
        precondition(view.isPrefixView)
        if backingLeaf.isWarm {
            // Stored Form remains full until #531. View restore dequantizes
            // only the prefix and copies whole-state layers into private
            // buffers, so this payload can take those buffers without a
            // second copy or retaining any tree body array.
            var cache = try view.restore(backingLeaf: backingLeaf)
            guard
                let materialized = HybridCacheSnapshot.captureMoving(
                    cache: &cache, offset: view.tokenOffset)
            else { throw HybridCacheSnapshot.ViewRestoreError.invalidBackingLeaf }
            return deferred(
                for: view, layers: materialized.layers, extending: nil, detaching: false)
        }
        return deferred(
            for: view, layers: try view.materializationLayers(backingLeaf: backingLeaf),
            extending: nil, detaching: true)
    }

    private static func deferred(
        for snapshot: HybridCacheSnapshot, layers: [HybridCacheSnapshot.LayerState],
        extending activeExtension: SnapshotExtension?, detaching: Bool
    ) -> (payload: SnapshotPayload, owed: DeferredLayers) {

        var owed: [DeferredLayers.Layer] = []
        owed.reserveCapacity(snapshot.layers.count)
        var detached: [MLXArray] = []
        var totalBytes = 0

        for layer in layers {
            var suffixBaseOffset: Int?
            var arrays = layer.state
            if detaching {
                arrays = layer.state.map { HybridCacheSnapshot.deepCopyState($0) }
                detached.append(contentsOf: arrays)
            } else if let activeExtension {
                switch layer.kind {
                case .sliceableAttention:
                    suffixBaseOffset = activeExtension.baseOffset
                    arrays = layer.state.map { array in
                        HybridCacheSnapshot.deepCopyState(
                            array[
                                .ellipsis, activeExtension.baseOffset..<snapshot.tokenOffset, 0...]
                        )
                    }
                case .wholeState:
                    // A whole-state layer rides whole in the segment and
                    // is detached whole: an extension payload retains no
                    // body array.
                    arrays = layer.state.map { HybridCacheSnapshot.deepCopyState($0) }
                }
                detached.append(contentsOf: arrays)
            }
            totalBytes += arrays.reduce(0) { $0 + $1.nbytes }
            owed.append(
                DeferredLayers.Layer(
                    className: layer.className,
                    arrays: arrays,
                    metaState: layer.metaState,
                    offset: layer.offset,
                    suffixBaseOffset: suffixBaseOffset
                ))
        }
        // One sync for every detached array, as `HybridCacheSnapshot.capture`
        // does for its copies; a full payload has nothing to evaluate.
        if !detached.isEmpty {
            eval(detached)
        }

        let deferred = DeferredLayers(owed)
        let payload = SnapshotPayload(
            tokenOffset: snapshot.tokenOffset,
            checkpointType: snapshot.checkpointType,
            extending: activeExtension,
            totalBytes: totalBytes,
            retainsBodyArrays: !snapshot.isPrefixView && !detaching && activeExtension == nil,
            materialize: { deferred.materialize() }
        )
        return (payload, deferred)
    }

    /// The evaluated arrays a deferred payload owes the SSD writer. Preparation
    /// transfers each array into its Data view's lifetime owner; the consuming
    /// writer releases that owner after the layer's last chunk. The lock guards
    /// the pending list; no live generation state enters this box.
    nonisolated final class DeferredLayers: @unchecked Sendable {
        struct Layer {
            let className: String
            let arrays: [MLXArray]
            let metaState: [String]
            let offset: Int
            let suffixBaseOffset: Int?
        }

        private let owed: OSAllocatedUnfairLock<[Layer]>

        init(_ layers: [Layer]) {
            owed = OSAllocatedUnfairLock(uncheckedState: layers)
        }

        /// Arrays not yet transferred to borrowed Data owners, in layer order.
        /// Empty once prepared; the Data still retains each array until written. Read by tests that
        /// check, by physical address, what a pending payload retains.
        var retainedArrays: [MLXArray] {
            owed.withLockUnchecked { $0.flatMap(\.arrays) }
        }

        /// Data's custom deallocator owns this box for exactly the lifetime
        /// of its borrowed bytes. It also retains a fallback contiguous Data
        /// if the vendor had to copy a non-contiguous input.
        private final class BorrowedArrayBytes: @unchecked Sendable {
            let array: MLXArray
            let data: Data

            init(_ array: MLXArray) {
                self.array = array
                // MLX's no-copy path force-unwraps the zero-size pointer.
                data = array.size == 0 ? Data() : array.asData(access: .noCopyIfContiguous).data
            }

            func view() -> Data {
                guard !data.isEmpty else { return Data() }
                return data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
                    Data(
                        bytesNoCopy: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
                        count: buffer.count,
                        deallocator: .custom { [self] _, _ in
                            withExtendedLifetime(self) {}
                        })
                }
            }
        }

        func materialize() -> [SnapshotPayload.LayerPayload] {
            var layers: [SnapshotPayload.LayerPayload] = []
            layers.reserveCapacity(owed.withLockUnchecked { $0.count })
            while let layer = owed.withLockUnchecked({ $0.isEmpty ? nil : $0.removeFirst() }) {
                var arrays: [SnapshotPayload.ArrayPayload] = []
                arrays.reserveCapacity(layer.arrays.count)
                for array in layer.arrays {
                    let bytes = BorrowedArrayBytes(array)
                    arrays.append(
                        SnapshotPayload.ArrayPayload(
                            data: bytes.view(),
                            dtype: SnapshotPayload.dtypeWireString(array.dtype),
                            shape: array.shape, borrowsArray: array.size > 0
                        ))
                }
                layers.append(
                    SnapshotPayload.LayerPayload(
                        className: layer.className,
                        state: arrays,
                        metaState: layer.metaState,
                        offset: layer.offset,
                        suffixBaseOffset: layer.suffixBaseOffset
                    ))
            }
            return layers
        }
    }

    /// Stable wire-format name for an MLX `DType`. Load-bearing: the
    /// result is written into the SSD snapshot header at
    /// `encodePlaceholderContainer(payload:descriptor:)` (in
    /// `PlaceholderContainer.swift`), so the mapping is part of the
    /// on-disk contract. A vendor-side rename of any `DType` case label
    /// would silently corrupt files without this explicit table.
    ///
    /// `@unknown default` traps via `fatalError` rather than inventing
    /// a placeholder string, because reaching it means the vendor
    /// shipped a new case that this table hasn't audited — inventing
    /// a wire name would persist an unreadable header under a claim of
    /// success. The remediation is always "add the case", not "paper
    /// over with a sentinel." Mirrors `DType.init(_ cmlxDtype:)` at
    /// `Vendor/.../mlx-swift/Source/MLX/DType.swift:61`, which uses
    /// the same loud-failure pattern for the C → Swift direction.
    static func dtypeWireString(_ dtype: DType) -> String {
        switch dtype {
        case .bool: return "bool"
        case .uint8: return "uint8"
        case .uint16: return "uint16"
        case .uint32: return "uint32"
        case .uint64: return "uint64"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .float16: return "float16"
        case .float32: return "float32"
        case .bfloat16: return "bfloat16"
        case .complex64: return "complex64"
        case .float64: return "float64"
        @unknown default:
            fatalError(
                "dtypeWireString missing case for MLX DType \(dtype) — "
                    + "extend the switch to preserve the SSD wire-format contract."
            )
        }
    }

    /// Inverse of ``dtypeWireString``. Must stay exhaustive against
    /// the forward table so round-tripping an SSD-resident snapshot
    /// cannot silently lose dtype information; every branch in
    /// `dtypeWireString` has a matching branch here. Returns `nil`
    /// for unknown wire strings so the `SSDSnapshotStore` decoder
    /// can distinguish a parse error from a supported dtype.
    static func dtypeFromWireString(_ wire: String) -> DType? {
        switch wire {
        case "bool": return .bool
        case "uint8": return .uint8
        case "uint16": return .uint16
        case "uint32": return .uint32
        case "uint64": return .uint64
        case "int8": return .int8
        case "int16": return .int16
        case "int32": return .int32
        case "int64": return .int64
        case "float16": return .float16
        case "float32": return .float32
        case "bfloat16": return .bfloat16
        case "complex64": return .complex64
        case "float64": return .float64
        default: return nil
        }
    }
}
