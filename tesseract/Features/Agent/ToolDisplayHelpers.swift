import Foundation
import MLXLMCommon
import os

/// The display-ready properties of a single tool call — the output of
/// ``ToolDisplayHelpers/displayProps(for:)``. A value type so it can be cached
/// and compared.
nonisolated struct ToolDisplayProps: Sendable, Equatable {
    let title: String
    let icon: String
    let argsFormatted: String
    let filePath: String?
}

/// Shared helpers for rendering tool calls — pure value logic in the model layer
/// (no SwiftUI), so the `nonisolated` Chat Transcript projection can call it
/// without inverting layering up into `Views`.
nonisolated enum ToolDisplayHelpers {

    /// Maps a tool name to an SF Symbol icon name.
    static func iconForTool(_ name: String) -> String {
        switch name.lowercased() {
        case let n where n.contains("read"): return "doc.text"
        case let n where n.contains("write"): return "square.and.pencil"
        case let n where n.contains("edit"): return "pencil.line"
        case "ls": return "folder"
        case let n where n.contains("list"): return "folder"
        case let n where n.contains("memory"): return "brain"
        case let n where n.contains("task") || n.contains("goal"): return "checklist"
        case let n where n.contains("search"): return "magnifyingglass"
        default: return "wrench.adjustable"
        }
    }

    /// Human-readable title for a tool call, extracting filenames from arguments.
    static func titleForTool(_ name: String, arguments: [String: JSONValue]?) -> String {
        guard let args = arguments else { return name }

        switch name.lowercased() {
        case "read_file":
            if case .string(let path)? = args["path"] {
                return "Reading \((path as NSString).lastPathComponent)"
            }
            return "Reading file"
        case "write_file":
            if case .string(let path)? = args["path"] {
                return "Writing to \((path as NSString).lastPathComponent)"
            }
            return "Writing file"
        case "edit_file":
            if case .string(let path)? = args["path"] {
                return "Editing \((path as NSString).lastPathComponent)"
            }
            return "Editing file"
        case "ls", "list", "list_files", "list_directory":
            if case .string(let path)? = args["path"] {
                return "Listing files in \((path as NSString).lastPathComponent)"
            }
            return "Listing files"
        case "search_files":
            if case .string(let query)? = args["query"] {
                return "Searching for \"\(query)\""
            }
            return "Searching files"
        case "memory_save":
            return "Saving memory"
        case "task_create":
            return "Creating task"
        case "task_complete":
            return "Completing task"
        default:
            return name
        }
    }

    /// Pretty-prints tool call arguments as JSON. Allocates a `JSONEncoder` per
    /// call rather than sharing one: ``displayProps(for:)`` memoizes its result
    /// by `ToolCallInfo`, so a committed tool call is encoded once (not on every
    /// streaming tick), and a fresh encoder means there is never a concurrent
    /// `encode()` on a shared instance to reason about. (`Sendable` would only
    /// say the encoder can be *moved* across isolation domains — not that
    /// concurrent `encode()` is safe, which is the property sharing would need.)
    static func formatArguments(_ arguments: [String: JSONValue]) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            let data = try encoder.encode(arguments)
            if let jsonString = String(data: data, encoding: .utf8) {
                return jsonString == "{}" ? "No arguments" : jsonString
            }
        } catch {
            return "\(arguments)"
        }

        return "No arguments"
    }

    /// Extracts the file path from tool arguments for file-centric tools.
    static func filePath(forTool name: String, arguments: [String: JSONValue]?) -> String? {
        guard let args = arguments else { return nil }
        switch name.lowercased() {
        case "read", "read_file", "write", "write_file", "edit", "edit_file":
            if let path = args.string(for: "path"), !path.isEmpty { return path }
            return nil
        default:
            return nil
        }
    }

    /// Memoized display properties, keyed by the whole `ToolCallInfo` (id + name
    /// + `argumentsJSON`). The streaming tail-patch reprojects the active turn
    /// ~20×/sec, and a committed tool call's `argumentsJSON` is frozen — so
    /// without memoization every committed tool in the active turn is re-decoded
    /// and re-encoded on every tick. A frozen tool call hits the cache; a live
    /// streaming tool call's arguments grow, so its key changes and it is
    /// correctly recomputed (and the bounded cache drops the stale states).
    ///
    /// Guarded by a lock because the projection is `nonisolated`: the app calls
    /// it MainActor-serialized, but the parallel test bundle calls it
    /// concurrently. Keyed by a value, the entry for a given `ToolCallInfo` is
    /// always identical, so the cache can never return a wrong result.
    private static let displayCache = OSAllocatedUnfairLock(
        initialState: [ToolCallInfo: ToolDisplayProps]()
    )

    /// Soft cap on cached entries. Committed tool calls per conversation stay
    /// well under this; the cap only bounds the transient states a long stream
    /// of tool-call arguments accumulates. On overflow the cache is cleared
    /// wholesale — committed entries re-warm on the next tick for a few cents.
    private static let displayCacheCap = 256

    /// All display properties for a tool call, computed once per distinct
    /// `ToolCallInfo` and memoized (see ``displayCache``).
    static func displayProps(for info: ToolCallInfo) -> ToolDisplayProps {
        displayCache.withLock { cache in
            if let cached = cache[info] { return cached }
            let props = computeDisplayProps(for: info)
            if cache.count >= displayCacheCap { cache.removeAll(keepingCapacity: true) }
            cache[info] = props
            return props
        }
    }

    private static func computeDisplayProps(for info: ToolCallInfo) -> ToolDisplayProps {
        let args = info.parsedArguments
        return ToolDisplayProps(
            title: titleForTool(info.name, arguments: args),
            icon: iconForTool(info.name),
            argsFormatted: formatArguments(args),
            filePath: filePath(forTool: info.name, arguments: args)
        )
    }
}
