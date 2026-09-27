import Foundation
@preconcurrency import MLX

// A small writer for Core ML ML Program packages: the model specification
// (Model.proto and MIL.proto, coremltools' `mlmodel/format`, BSD-3), its
// weight blob file (MILBlob storage format v2) and the `.mlpackage` layout.
//
// It knows exactly what one fixed, static-shaped fp16 graph needs: tensor
// inputs and outputs, `const` ops (immediate or blob-backed) and ordinary
// ops whose output types the caller states. It is not a MIL compiler; Core
// ML's own compiler validates and optimizes what it writes.

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

    /// A `map<string, V>` entry.
    mutating func mapEntry(_ field: Int, key: String, _ value: (inout ProtobufWriter) -> Void) {
        message(field) { entry in
            entry.string(1, key)
            entry.message(2, value)
        }
    }
}

// MARK: - MIL

/// A tensor in a MIL program: its name and static type.
struct MILVar {
    let name: String
    let type: MILType
}

struct MILType: Equatable {
    enum DataType: Int64 {
        case bool = 1
        case string = 2
        case float16 = 10
        case float32 = 11
        case int32 = 23
    }
    let dataType: DataType
    let shape: [Int]

    static func fp16(_ shape: [Int]) -> MILType { .init(dataType: .float16, shape: shape) }

    /// MIL.proto `ValueType` { tensorType { dataType rank dimensions } }.
    func write(_ w: inout ProtobufWriter) {
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
final class MILFunctionBuilder {
    private(set) var inputs: [MILVar] = []
    private var operations: [Data] = []
    private(set) var outputs: [MILVar] = []
    private var count = 0
    let blobs: MILBlobWriter

    init(blobs: MILBlobWriter) {
        self.blobs = blobs
    }

    private func fresh(_ prefix: String) -> String {
        count += 1
        return "\(prefix)_\(count)"
    }

    func input(_ name: String, _ type: MILType) -> MILVar {
        let v = MILVar(name: name, type: type)
        inputs.append(v)
        return v
    }

    func output(_ v: MILVar) {
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

    func ints(_ values: [Int]) -> MILVar {
        let type = MILType(dataType: .int32, shape: [values.count])
        return const(type) {
            Self.immediateValue(&$0, type, field: 2) { $0.packed(1, values.map(Int64.init)) }
        }
    }

    func int(_ value: Int) -> MILVar {
        let type = MILType(dataType: .int32, shape: [])
        return const(type) { Self.immediateValue(&$0, type, field: 2) { $0.packed(1, [Int64(value)]) } }
    }

    func bool(_ value: Bool) -> MILVar {
        let type = MILType(dataType: .bool, shape: [])
        return const(type) { Self.immediateValue(&$0, type, field: 3) { $0.packed(1, [value ? 1 : 0]) } }
    }

    func string(_ value: String) -> MILVar {
        const(MILType(dataType: .string, shape: [])) { Self.stringValue(value, &$0) }
    }

    /// An fp16 scalar, stored as its two bytes (how MIL writes fp16).
    func half(_ value: Float) -> MILVar {
        let type = MILType.fp16([])
        let bits = Float16(value).bitPattern
        return const(type) {
            Self.immediateValue(&$0, type, field: 7) {
                $0.bytes(1, Data([UInt8(bits & 0xFF), UInt8(bits >> 8)]))
            }
        }
    }

    /// An fp16 tensor `shape` from the blob file: `values`' bytes.
    func weight(_ values: MLXArray, shape: [Int]) -> MILVar {
        precondition(values.size == shape.reduce(1, *), "weight size \(values.size) vs \(shape)")
        let offset = blobs.append(values)
        let type = MILType.fp16(shape)
        return const(type) { v in
            v.message(2) { type.write(&$0) }
            v.message(5) { blob in
                blob.string(1, "@model_path/weights/weight.bin")
                blob.int(2, Int64(offset))
            }
        }
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
    func op(
        _ type: String, _ inputs: [(String, [MILVar])], _ outputType: MILType,
        name: String? = nil
    ) -> MILVar {
        let out = MILVar(name: name ?? fresh(type), type: outputType)
        var w = ProtobufWriter()
        w.string(1, type)
        for (name, vars) in inputs {
            w.mapEntry(2, key: name) { argument in
                for v in vars {
                    argument.message(1) { binding in binding.string(1, v.name) }
                }
            }
        }
        w.message(3) { o in
            o.string(1, out.name)
            o.message(2) { outputType.write(&$0) }
        }
        w.mapEntry(5, key: "name") { Self.stringValue(out.name, &$0) }
        operations.append(w.data)
        return out
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
final class MILBlobWriter {
    private let handle: FileHandle
    private var size: UInt64 = 64
    private var blobCount: UInt32 = 0
    private var error: Error?

    init(url: URL) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: Data(count: 64)) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path])
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    /// Appends an fp16 array's values; returns the offset to reference them by.
    func append(_ values: MLXArray) -> UInt64 {
        precondition(values.dtype == .float16)
        let bytes = values.asData(access: .noCopyIfContiguous).data
        let metadataOffset = size
        var record = Data(count: 64)
        record.withUnsafeMutableBytes { p in
            p.storeBytes(of: UInt32(0xDEAD_BEEF).littleEndian, toByteOffset: 0, as: UInt32.self)
            p.storeBytes(of: UInt32(1).littleEndian, toByteOffset: 4, as: UInt32.self)  // Float16
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
    func finish() throws {
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

enum MLProgramPackage {
    /// Core ML 7 (iOS 17 / macOS 14): specification version 8, opset
    /// `CoreML7`.
    static let specificationVersion: Int64 = 8
    static let opset = "CoreML7"

    /// Model.proto: the description (fp16 multi-array features matching the
    /// function's inputs and outputs) and the ML program with its one
    /// function, `main`.
    static func specification(
        _ function: MILFunctionBuilder, metadata: [String: String]
    ) -> Data {
        func feature(_ w: inout ProtobufWriter, _ field: Int, _ v: MILVar) {
            w.message(field) { f in
                f.string(1, v.name)
                f.message(3) { type in
                    type.message(5) { array in
                        array.packed(1, v.type.shape.map(Int64.init))
                        array.int(2, 65552)  // FLOAT16
                    }
                }
            }
        }
        var model = ProtobufWriter()
        model.int(1, specificationVersion)
        model.message(2) { description in
            for v in function.inputs { feature(&description, 1, v) }
            for v in function.outputs { feature(&description, 10, v) }
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
            program.mapEntry(2, key: "main") { function.writeFunction(&$0, opset: opset) }
        }
        return model.data
    }

    /// The weight file of the `.mlpackage` at `url`, its directories made.
    static func weightsURL(in url: URL) throws -> URL {
        let weights = url.appendingPathComponent("Data/com.apple.CoreML/weights", isDirectory: true)
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        return weights.appendingPathComponent("weight.bin")
    }

    /// Completes the `.mlpackage` at `url`, whose weights are written
    /// (`weightsURL`): the specification and the manifest.
    static func write(specification: Data, to url: URL) throws {
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
