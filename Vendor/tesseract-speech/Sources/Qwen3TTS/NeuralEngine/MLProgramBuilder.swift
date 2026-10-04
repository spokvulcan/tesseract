import Foundation
@preconcurrency import MLX

// A small writer for Core ML ML Program packages: the model specification
// (Model.proto and MIL.proto, coremltools' `mlmodel/format`, BSD-3), its
// weight blob file (MILBlob storage format v2) and the `.mlpackage` layout.
//
// It knows what our fixed, static-shaped fp16 graphs need: tensor inputs and
// outputs, states (Core ML 8), `const` ops (immediate or blob-backed), weights
// stored as 8-bit blocks, and ordinary ops whose output types the caller
// states. It is not a MIL compiler; Core ML's own compiler validates and
// optimizes what it writes.

// MARK: - Protobuf

/// Protocol Buffers wire format, the parts the Core ML specification uses.
struct ProtobufWriter {
    private(set) var data = Data()

    mutating func varint(_ value: UInt64) {
        var v = value
        while v >= 0x80 {
            data.append(UInt8(v & 0x7F) | 0x80)
            v >>= 7
        }
        data.append(UInt8(v))
    }

    private mutating func tag(_ field: Int, _ wireType: Int) {
        varint(UInt64(field << 3 | wireType))
    }

    mutating func int(_ field: Int, _ value: Int64) {
        tag(field, 0)
        varint(UInt64(bitPattern: value))
    }

    mutating func bool(_ field: Int, _ value: Bool) {
        tag(field, 0)
        varint(value ? 1 : 0)
    }

    mutating func bytes(_ field: Int, _ value: Data) {
        tag(field, 2)
        varint(UInt64(value.count))
        data.append(value)
    }

    mutating func string(_ field: Int, _ value: String) {
        bytes(field, Data(value.utf8))
    }

    mutating func message(_ field: Int, _ build: (inout ProtobufWriter) -> Void) {
        var inner = ProtobufWriter()
        build(&inner)
        bytes(field, inner.data)
    }

    /// A packed repeated varint field (int32/int64/bool).
    mutating func packed(_ field: Int, _ values: [Int64]) {
        var inner = ProtobufWriter()
        for v in values { inner.varint(UInt64(bitPattern: v)) }
        bytes(field, inner.data)
    }

    /// A packed repeated float field.
    mutating func packedFloats(_ field: Int, _ values: [Float]) {
        var raw = Data(capacity: values.count * 4)
        for v in values { withUnsafeBytes(of: v.bitPattern.littleEndian) { raw.append(contentsOf: $0) } }
        bytes(field, raw)
    }

    /// A `map<string, V>` entry.
    mutating func mapEntry(_ field: Int, key: String, _ value: (inout ProtobufWriter) -> Void) {
        message(field) { entry in
            entry.string(1, key)
            entry.message(2, value)
        }
    }
}

// MARK: - MIL

/// A tensor (or a state wrapping one) in a MIL program: its name and static type.
package struct MILVar {
    package let name: String
    package let type: MILType
}

package struct MILType: Equatable {
    package enum DataType: Int64 {
        case bool = 1
        case string = 2
        case float16 = 10
        case float32 = 11
        case int8 = 21
        case int32 = 23
        case uint8 = 31
        case uint4 = 35
    }
    package let dataType: DataType
    package let shape: [Int]
    /// A state wrapping this tensor type (Core ML 8): an input the model reads
    /// with `read_state` and writes with `write_state`, kept between calls.
    package var isState = false

    package init(dataType: DataType, shape: [Int], isState: Bool = false) {
        self.dataType = dataType
        self.shape = shape
        self.isState = isState
    }

    package static func fp16(_ shape: [Int]) -> MILType { .init(dataType: .float16, shape: shape) }
    package static func fp32(_ shape: [Int]) -> MILType { .init(dataType: .float32, shape: shape) }
    package static func int32(_ shape: [Int]) -> MILType { .init(dataType: .int32, shape: shape) }
    package static func bool(_ shape: [Int]) -> MILType { .init(dataType: .bool, shape: shape) }
    /// An fp16 tensor state.
    package static func state(_ shape: [Int]) -> MILType {
        .init(dataType: .float16, shape: shape, isState: true)
    }

    /// The tensor a state wraps.
    package var wrapped: MILType { .init(dataType: dataType, shape: shape) }

    /// MIL.proto `ValueType` { tensorType { dataType rank dimensions } }, or
    /// `stateType { wrappedType { tensorType … } }`.
    func write(_ w: inout ProtobufWriter) {
        guard !isState else {
            w.message(5) { state in state.message(1) { wrapped.write(&$0) } }
            return
        }
        w.message(1) { t in
            t.int(1, dataType.rawValue)
            t.int(2, Int64(shape.count))
            for size in shape {
                t.message(3) { d in d.message(1) { c in c.int(1, Int64(size)) } }
            }
        }
    }
}

/// Builds one MIL function: typed inputs, ops in topological order, named
/// outputs. Weights go to a `MILBlobWriter`.
package final class MILFunctionBuilder {
    package private(set) var inputs: [MILVar] = []
    private var operations: [Data] = []
    package private(set) var outputs: [MILVar] = []
    private var count = 0
    package let blobs: MILBlobWriter

    package init(blobs: MILBlobWriter) {
        self.blobs = blobs
    }

    private func fresh(_ prefix: String) -> String {
        count += 1
        return "\(prefix)_\(count)"
    }

    package func input(_ name: String, _ type: MILType) -> MILVar {
        let v = MILVar(name: name, type: type)
        inputs.append(v)
        return v
    }

    package func output(_ v: MILVar) {
        outputs.append(v)
    }

    // MARK: Constants

    /// A const op: `attributes { name, val }`, one output.
    private func const(_ type: MILType, value: (inout ProtobufWriter) -> Void) -> MILVar {
        let v = MILVar(name: fresh("const"), type: type)
        var w = ProtobufWriter()
        w.string(1, "const")
        w.message(3) { o in
            o.string(1, v.name)
            o.message(2) { type.write(&$0) }
        }
        w.mapEntry(5, key: "name") { Self.stringValue(v.name, &$0) }
        w.mapEntry(5, key: "val") { value(&$0) }
        operations.append(w.data)
        return v
    }

    package func ints(_ values: [Int]) -> MILVar {
        let type = MILType(dataType: .int32, shape: [values.count])
        return const(type) {
            Self.immediateValue(&$0, type, field: 2) { $0.packed(1, values.map(Int64.init)) }
        }
    }

    package func int(_ value: Int) -> MILVar {
        let type = MILType(dataType: .int32, shape: [])
        return const(type) { Self.immediateValue(&$0, type, field: 2) { $0.packed(1, [Int64(value)]) } }
    }

    package func bool(_ value: Bool) -> MILVar {
        let type = MILType(dataType: .bool, shape: [])
        return const(type) { Self.immediateValue(&$0, type, field: 3) { $0.packed(1, [value ? 1 : 0]) } }
    }

    package func string(_ value: String) -> MILVar {
        const(MILType(dataType: .string, shape: [])) { Self.stringValue(value, &$0) }
    }

    /// An fp16 scalar, stored as its two bytes (how MIL writes fp16).
    package func half(_ value: Float) -> MILVar {
        let type = MILType.fp16([])
        let bits = Float16(value).bitPattern
        return const(type) {
            Self.immediateValue(&$0, type, field: 7) {
                $0.bytes(1, Data([UInt8(bits & 0xFF), UInt8(bits >> 8)]))
            }
        }
    }

    /// A small fp16 tensor written inline: positions' rotations, masks.
    package func halves(_ values: [Float], shape: [Int]) -> MILVar {
        precondition(values.count == shape.reduce(1, *), "\(values.count) values vs \(shape)")
        let type = MILType.fp16(shape)
        var raw = Data(capacity: values.count * 2)
        for v in values {
            let bits = Float16(v).bitPattern
            raw.append(UInt8(bits & 0xFF))
            raw.append(UInt8(bits >> 8))
        }
        return const(type) { Self.immediateValue(&$0, type, field: 7) { $0.bytes(1, raw) } }
    }

    /// An fp16 tensor `shape` from the blob file: `values`' bytes.
    package func weight(_ values: MLXArray, shape: [Int]) -> MILVar {
        precondition(values.size == shape.reduce(1, *), "weight size \(values.size) vs \(shape)")
        let offset = blobs.append(values)
        return blobConst(MILType.fp16(shape), offset: offset)
    }

    /// A tensor of `type` whose bytes are at `offset` in the blob file.
    package func blobConst(_ type: MILType, offset: UInt64) -> MILVar {
        const(type) { v in
            v.message(2) { type.write(&$0) }
            v.message(5) { blob in
                blob.string(1, "@model_path/weights/weight.bin")
                blob.int(2, Int64(offset))
            }
        }
    }

    /// A weight kept as 8-bit blocks (Core ML 8's `constexpr_blockwise_shift_scale`):
    /// `scale * (data - offset)` per block, fp16 out. `data` is `shape`'s
    /// unsigned bytes; `scale` and `offset` are fp16 with `blocks` per row,
    /// shaped like `data` with its last axis divided by the block size.
    package func blockwiseWeight(
        data: Data, scale: Data, offset: Data, shape: [Int], blockShape: [Int]
    ) -> MILVar {
        precondition(data.count == shape.reduce(1, *), "data \(data.count) vs \(shape)")
        precondition(scale.count == blockShape.reduce(1, *) * 2 && offset.count == scale.count)
        let q = blobConst(
            MILType(dataType: .uint8, shape: shape), offset: blobs.append(bytes: data, type: .uint8))
        let s = blobConst(.fp16(blockShape), offset: blobs.append(bytes: scale, type: .float16))
        let o = blobConst(.fp16(blockShape), offset: blobs.append(bytes: offset, type: .float16))
        return op(
            "constexpr_blockwise_shift_scale", [("data", [q]), ("scale", [s]), ("offset", [o])],
            .fp16(shape))
    }

    /// A palettized weight (Core ML 8's `constexpr_lut_to_dense`): `indices`,
    /// one byte per value of `shape`, into fp16 tables of 256 values. `lutShape`
    /// is `shape` divided into the groups that share a table, then 256, then
    /// 1 (scalar entries).
    package func palettizedWeight(
        indices: Data, lut: Data, shape: [Int], lutShape: [Int]
    ) -> MILVar {
        precondition(indices.count == shape.reduce(1, *), "indices \(indices.count) vs \(shape)")
        precondition(lut.count == lutShape.reduce(1, *) * 2, "lut \(lut.count) vs \(lutShape)")
        let i = blobConst(
            MILType(dataType: .uint8, shape: shape), offset: blobs.append(bytes: indices, type: .uint8))
        let l = blobConst(.fp16(lutShape), offset: blobs.append(bytes: lut, type: .float16))
        return op("constexpr_lut_to_dense", [("indices", [i]), ("lut", [l])], .fp16(shape))
    }

    /// An 8-bit weight with one fp16 scale per block and no offset
    /// (`constexpr_blockwise_shift_scale`, signed data).
    package func symmetricWeight(data: Data, scale: Data, shape: [Int], blockShape: [Int]) -> MILVar {
        precondition(data.count == shape.reduce(1, *))
        let q = blobConst(
            MILType(dataType: .int8, shape: shape), offset: blobs.append(bytes: data, type: .int8))
        let s = blobConst(.fp16(blockShape), offset: blobs.append(bytes: scale, type: .float16))
        return op("constexpr_blockwise_shift_scale", [("data", [q]), ("scale", [s])], .fp16(shape))
    }

    /// `Value { type, immediateValue { tensor { <field>: payload } } }`, the
    /// field naming the tensor's value list (ints, bools, strings, bytes).
    private static func immediateValue(
        _ v: inout ProtobufWriter, _ type: MILType, field: Int,
        _ payload: (inout ProtobufWriter) -> Void
    ) {
        v.message(2) { type.write(&$0) }
        v.message(3) { imm in imm.message(1) { t in t.message(field) { payload(&$0) } } }
    }

    private static func stringValue(_ s: String, _ v: inout ProtobufWriter) {
        immediateValue(&v, MILType(dataType: .string, shape: []), field: 4) { $0.string(1, s) }
    }

    // MARK: Operations

    /// An op `type(inputs…)` with one output of `outputType`. Each input
    /// binds one variable, or several for a variadic input (`values`).
    @discardableResult
    package func op(
        _ type: String, _ inputs: [(String, [MILVar])], _ outputType: MILType,
        name: String? = nil
    ) -> MILVar {
        ops(type, inputs, [outputType], names: name.map { [$0] })[0]
    }

    /// An op with any number of outputs (`topk`'s values and indices; none
    /// for `write_state`).
    @discardableResult
    package func ops(
        _ type: String, _ inputs: [(String, [MILVar])], _ outputTypes: [MILType],
        names: [String]? = nil
    ) -> [MILVar] {
        let outs = outputTypes.enumerated().map { i, t in
            MILVar(name: names?[i] ?? fresh(type), type: t)
        }
        var w = ProtobufWriter()
        w.string(1, type)
        for (name, vars) in inputs {
            w.mapEntry(2, key: name) { argument in
                for v in vars {
                    argument.message(1) { binding in binding.string(1, v.name) }
                }
            }
        }
        for out in outs {
            w.message(3) { o in
                o.string(1, out.name)
                o.message(2) { out.type.write(&$0) }
            }
        }
        w.mapEntry(5, key: "name") { Self.stringValue(outs.first?.name ?? fresh(type), &$0) }
        operations.append(w.data)
        return outs
    }

    /// MIL.proto `Function` with one block specialization for `opset`.
    func writeFunction(_ w: inout ProtobufWriter, opset: String) {
        for v in inputs {
            w.message(1) { n in
                n.string(1, v.name)
                n.message(2) { v.type.write(&$0) }
            }
        }
        w.string(2, opset)
        w.mapEntry(3, key: opset) { block in
            for v in outputs { block.string(2, v.name) }
            for op in operations { block.bytes(3, op) }
        }
    }
}

// MARK: - Weight blobs

/// MILBlob storage v2: a 64-byte header (count, version 2), then per blob a
/// 64-byte metadata record (sentinel 0xDEADBEEF, dtype, size, data offset)
/// and the data, each 64-byte aligned. A blob is referenced by the offset of
/// its metadata record. Each blob goes to the file as it comes, and the
/// header at the end, so building holds one weight at a time; the first
/// write error is thrown by `finish()`.
package final class MILBlobWriter {
    /// MILBlob's storage types (`BlobDataType`).
    package enum StorageType: UInt32 {
        case float16 = 1
        case float32 = 2
        case uint8 = 3
        case int8 = 4
    }

    private let handle: FileHandle
    private var size: UInt64 = 64
    private var blobCount: UInt32 = 0
    private var error: Error?

    package init(url: URL) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: Data(count: 64)) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path])
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    /// Appends an fp16 array's values; returns the offset to reference them by.
    package func append(_ values: MLXArray) -> UInt64 {
        precondition(values.dtype == .float16)
        return append(bytes: values.asData(access: .noCopyIfContiguous).data, type: .float16)
    }

    /// Appends raw values of `type`; returns the offset to reference them by.
    package func append(bytes: Data, type: StorageType) -> UInt64 {
        let metadataOffset = size
        var record = Data(count: 64)
        record.withUnsafeMutableBytes { p in
            p.storeBytes(of: UInt32(0xDEAD_BEEF).littleEndian, toByteOffset: 0, as: UInt32.self)
            p.storeBytes(of: type.rawValue.littleEndian, toByteOffset: 4, as: UInt32.self)
            p.storeBytes(of: UInt64(bytes.count).littleEndian, toByteOffset: 8, as: UInt64.self)
            p.storeBytes(of: (metadataOffset + 64).littleEndian, toByteOffset: 16, as: UInt64.self)
        }
        let padding = (64 - bytes.count % 64) % 64
        write(record)
        write(bytes)
        if padding > 0 { write(Data(count: padding)) }
        size += UInt64(64 + bytes.count + padding)
        blobCount += 1
        return metadataOffset
    }

    /// Writes the header and closes the file.
    package func finish() throws {
        defer { try? handle.close() }
        if let error { throw error }
        var header = Data(count: 64)
        header.withUnsafeMutableBytes { p in
            p.storeBytes(of: blobCount.littleEndian, toByteOffset: 0, as: UInt32.self)
            p.storeBytes(of: UInt32(2).littleEndian, toByteOffset: 4, as: UInt32.self)
        }
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: header)
    }

    private func write(_ data: Data) {
        guard error == nil else { return }
        do { try handle.write(contentsOf: data) } catch { self.error = error }
    }
}

// MARK: - Model and package

package enum MLProgramPackage {
    /// The Core ML release a program targets: its specification version and
    /// MIL opset.
    package enum Target {
        /// Core ML 7 (iOS 17 / macOS 14): specification 8, opset `CoreML7`.
        case coreML7
        /// Core ML 8 (iOS 18 / macOS 15): specification 9, opset `CoreML8`.
        /// States, blockwise weights and functions need it.
        case coreML8

        var specificationVersion: Int64 { self == .coreML7 ? 8 : 9 }
        var opset: String { self == .coreML7 ? "CoreML7" : "CoreML8" }
    }

    /// Model.proto for a program with one function, `main`.
    package static func specification(
        _ function: MILFunctionBuilder, metadata: [String: String], target: Target = .coreML7
    ) -> Data {
        specification(functions: [("main", function)], metadata: metadata, target: target)
    }

    /// Model.proto: the description (each function's features) and the ML
    /// program. With one function the description is the classic one; with
    /// several it lists them (Core ML 8), the first being the default.
    package static func specification(
        functions: [(name: String, builder: MILFunctionBuilder)], metadata: [String: String],
        target: Target
    ) -> Data {
        precondition(!functions.isEmpty)
        precondition(functions.count == 1 || target == .coreML8, "several functions need Core ML 8")
        var model = ProtobufWriter()
        model.int(1, target.specificationVersion)
        model.message(2) { description in
            if functions.count == 1 {
                let f = functions[0].builder
                for v in f.inputs where !v.type.isState { feature(&description, 1, v) }
                for v in f.outputs { feature(&description, 10, v) }
                for v in f.inputs where v.type.isState { feature(&description, 13, v) }
            } else {
                for (name, f) in functions {
                    description.message(20) { fd in
                        fd.string(1, name)
                        for v in f.inputs where !v.type.isState { feature(&fd, 2, v) }
                        for v in f.outputs { feature(&fd, 3, v) }
                        for v in f.inputs where v.type.isState { feature(&fd, 6, v) }
                    }
                }
                description.string(21, functions[0].name)
            }
            description.message(100) { meta in
                for (key, value) in metadata.sorted(by: { $0.key < $1.key }) {
                    meta.message(100) { entry in
                        entry.string(1, key)
                        entry.string(2, value)
                    }
                }
            }
        }
        model.message(502) { program in
            program.int(1, 1)
            for (name, f) in functions {
                program.mapEntry(2, key: name) { f.writeFunction(&$0, opset: target.opset) }
            }
        }
        return model.data
    }

    /// A feature's description: a multi-array of the var's type, or a state
    /// wrapping one.
    private static func feature(_ w: inout ProtobufWriter, _ field: Int, _ v: MILVar) {
        func array(_ a: inout ProtobufWriter) {
            a.packed(1, v.type.shape.map(Int64.init))
            a.int(2, arrayDataType(v.type.dataType))
        }
        w.message(field) { f in
            f.string(1, v.name)
            f.message(3) { type in
                if v.type.isState {
                    type.message(8) { state in state.message(1) { array(&$0) } }
                } else {
                    type.message(5) { array(&$0) }
                }
            }
        }
    }

    /// FeatureTypes.proto `ArrayDataType`.
    private static func arrayDataType(_ type: MILType.DataType) -> Int64 {
        switch type {
        case .float32: 65568
        case .int32: 131104
        case .int8: 131080
        default: 65552  // FLOAT16
        }
    }

    /// The weight file of the `.mlpackage` at `url`, its directories made.
    package static func weightsURL(in url: URL) throws -> URL {
        let weights = url.appendingPathComponent("Data/com.apple.CoreML/weights", isDirectory: true)
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        return weights.appendingPathComponent("weight.bin")
    }

    /// Completes the `.mlpackage` at `url`, whose weights are written
    /// (`weightsURL`): the specification and the manifest.
    package static func write(specification: Data, to url: URL) throws {
        let root = url.appendingPathComponent("Data/com.apple.CoreML", isDirectory: true)
        try specification.write(to: root.appendingPathComponent("model.mlmodel"))
        let modelID = UUID().uuidString
        let weightsID = UUID().uuidString
        let manifest: [String: Any] = [
            "fileFormatVersion": "1.0.0",
            "itemInfoEntries": [
                modelID: [
                    "author": "com.apple.CoreML", "description": "CoreML Model Specification",
                    "name": "model.mlmodel", "path": "com.apple.CoreML/model.mlmodel",
                ],
                weightsID: [
                    "author": "com.apple.CoreML", "description": "CoreML Model Weights",
                    "name": "weights", "path": "com.apple.CoreML/weights",
                ],
            ],
            "rootModelIdentifier": modelID,
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
            .write(to: url.appendingPathComponent("Manifest.json"))
    }
}
