import Foundation
import MLXLMCommon

// MARK: - JSONValue Extraction Helpers

extension [String: JSONValue] {
    func string(for key: String) -> String? {
        guard let value = self[key] else { return nil }
        switch value {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return String(b)
        case .null: return nil
        default: return nil
        }
    }

    func int(for key: String) -> Int? {
        guard let value = self[key] else { return nil }
        switch value {
        case .int(let i): return i
        case .double(let d): return Int(d)
        case .string(let s): return Int(s)
        case .null: return nil
        default: return nil
        }
    }

    func intArray(for key: String) -> [Int]? {
        guard let value = self[key] else { return nil }
        switch value {
        case .array(let arr):
            return arr.compactMap { element -> Int? in
                switch element {
                case .int(let i): return i
                case .double(let d): return Int(d)
                case .string(let s): return Int(s)
                default: return nil
                }
            }
        case .int(let i): return [i]  // Single int → treat as one-element array
        case .string(let s):
            // Accept comma-separated string like "3, 4, 5"
            let parts = s.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            return parts.isEmpty ? nil : parts
        default: return nil
        }
    }

    func bool(for key: String) -> Bool? {
        guard let value = self[key] else { return nil }
        switch value {
        case .bool(let b): return b
        case .int(let i): return i != 0
        case .string(let s): return s == "true" || s == "1"
        case .null: return nil
        default: return nil
        }
    }
}
