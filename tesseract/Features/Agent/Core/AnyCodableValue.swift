import Foundation

/// Type-safe JSON value wrapper for the tagged message encoding system.
/// Used by `MessageCodecRegistry` (Epic 2) for heterogeneous message persistence.
///
/// Encodes and decodes native JSON types via `singleValueContainer`.
///
/// **JSON number limitation**: JSON has a single number type, so `.int(1)` and
/// `.double(1.0)` both encode as `1`. On decode, integers are preferred for whole
/// numbers — meaning `.double(1.0)` will round-trip as `.int(1)`. Non-integral
/// doubles (e.g. `.double(3.14)`) round-trip correctly. The `init(_ value: Any)`
/// factory (from `JSONSerialization`) does NOT have this limitation because
/// `NSNumber` preserves the original type via `objCType`.
enum AnyCodableValue: Sendable, Hashable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([AnyCodableValue])
    case object([String: AnyCodableValue])

    // MARK: - Factory from JSONSerialization output

    /// Convert an untyped `Any` (e.g. from `JSONSerialization`) into a typed value.
    /// Returns `.null` for unrecognized types.
    ///
    /// Unlike `Codable` round-tripping, this factory correctly distinguishes
    /// `.int` from `.double` for integral values because `JSONSerialization`
    /// produces `NSNumber` instances whose `objCType` reflects the original
    /// JSON token (integer literal → `"q"`, float literal → `"d"`).
    /// Swift's `as Int` / `as Double` pattern matching does NOT preserve this
    /// distinction, so we inspect `objCType` directly on `NSNumber`.
    init(_ value: Any) {
        switch value {
        case is NSNull:
            self = .null
        case let n as NSNumber:
            // CFBoolean is bridged as NSNumber with objCType "c". Identity-check
            // against the CFBoolean singletons to avoid confusing Bool with Int8.
            if n === kCFBooleanTrue || n === kCFBooleanFalse {
                self = .bool(n.boolValue)
            } else {
                // objCType "d" = Double, "f" = Float. Everything else is integral.
                let t = n.objCType.pointee
                if t == UInt8(ascii: "d") || t == UInt8(ascii: "f") {
                    self = .double(n.doubleValue)
                } else {
                    self = .int(n.intValue)
                }
            }
        case let s as String:
            self = .string(s)
        case let arr as [Any]:
            self = .array(arr.map { AnyCodableValue($0) })
        case let dict as [String: Any]:
            self = .object(dict.mapValues { AnyCodableValue($0) })
        default:
            self = .null
        }
    }

    // MARK: - Convenience accessors

    var boolValue: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }

    var intValue: Int? {
        if case .int(let v) = self { return v }
        return nil
    }

    var doubleValue: Double? {
        switch self {
        case .double(let v): return v
        case .int(let v): return Double(v)
        default: return nil
        }
    }

    var stringValue: String? {
        if case .string(let v) = self { return v }
        return nil
    }

    var arrayValue: [AnyCodableValue]? {
        if case .array(let v) = self { return v }
        return nil
    }

    var objectValue: [String: AnyCodableValue]? {
        if case .object(let v) = self { return v }
        return nil
    }

    var isNull: Bool {
        if case .null = self { return true }
        return false
    }
}

// MARK: - Codable (native JSON)

extension AnyCodableValue: Codable {
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            // Bool before Int: JSONDecoder correctly rejects JSON `1` for Bool
            // (it checks NSNumber.objCType), so this only matches `true`/`false`.
            self = .bool(b)
        } else if let i = try? container.decode(Int.self) {
            self = .int(i)
        } else if let d = try? container.decode(Double.self) {
            self = .double(d)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let arr = try? container.decode([AnyCodableValue].self) {
            self = .array(arr)
        } else if let obj = try? container.decode([String: AnyCodableValue].self) {
            self = .object(obj)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "AnyCodableValue: unsupported JSON type")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let v):
            try container.encode(v)
        case .int(let v):
            try container.encode(v)
        case .double(let v):
            try container.encode(v)
        case .string(let v):
            try container.encode(v)
        case .array(let v):
            try container.encode(v)
        case .object(let v):
            try container.encode(v)
        }
    }
}
