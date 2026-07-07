import Foundation
import MLXLMCommon
import os

/// The display-ready properties of a single tool call — the output of
/// ``ToolDisplayHelpers/displayProps(for:)``. A value type so it can be cached
/// and compared.
nonisolated struct ToolDisplayProps: Sendable, Equatable {
    let title: String
    let argsFormatted: String
    let filePath: String?
}

/// Shared helpers for rendering tool calls — pure value logic in the model layer
/// (no SwiftUI), so the `nonisolated` Chat Transcript projection can call it
/// without inverting layering up into `Views`.
nonisolated enum ToolDisplayHelpers {

    /// The Tool Row Title: imperative verb + workspace-relative target,
    /// matched against the actual tool registry (`read`, `write`, `edit`,
    /// `ls`, `use_skill`, `browser.search`, `browser.fetch`). Imperative because
    /// a committed row describes a completed act — the spinner owns "running".
    /// Unknown tools fall back to the raw name.
    static func titleForTool(_ name: String, arguments: [String: JSONValue]?) -> String {
        switch name.lowercased() {
        case "read":
            return fileTitle("Read", arguments)
        case "write":
            return fileTitle("Write", arguments)
        case "edit":
            return fileTitle("Edit", arguments)
        case "ls":
            guard let path = arguments?.string(for: "path"), !path.isEmpty else {
                return "List workspace"
            }
            let display = workspaceRelative(path)
            return display == "." ? "List workspace" : "List \(display)"
        case "use_skill":
            if let skill = arguments?.string(for: "name"), !skill.isEmpty {
                return "Load skill \(skill)"
            }
            return "Load skill"
        case "browser.search":
            if let query = arguments?.string(for: "query"), !query.isEmpty {
                return "Search \u{201C}\(query)\u{201D}"
            }
            return "Search the web"
        case "browser.fetch":
            if let url = arguments?.string(for: "url"), !url.isEmpty {
                return "Fetch \(displayURL(url))"
            }
            return "Fetch URL"
        default:
            return name
        }
    }

    /// `"Read notes/todo.md"` — bare verb while arguments are still streaming.
    private static func fileTitle(_ verb: String, _ arguments: [String: JSONValue]?) -> String {
        guard let path = arguments?.string(for: "path"), !path.isEmpty else { return verb }
        return "\(verb) \(workspaceRelative(path))"
    }

    /// Render a tool path relative to the Workspace (the sandbox root):
    /// absolute paths inside it are stripped to their relative form; relative
    /// paths lose any leading "./".
    private static func workspaceRelative(_ path: String) -> String {
        if path.hasPrefix("/") {
            return PathSandbox(root: PathSandbox.defaultRoot)
                .displayPath(URL(fileURLWithPath: path))
        }
        var trimmed = path
        while trimmed.hasPrefix("./") { trimmed.removeFirst(2) }
        return trimmed.isEmpty ? "." : trimmed
    }

    /// A URL compacted for a one-line title: scheme and trailing slash dropped.
    private static func displayURL(_ url: String) -> String {
        var display = url
        for scheme in ["https://", "http://"] where display.hasPrefix(scheme) {
            display.removeFirst(scheme.count)
        }
        if display.hasSuffix("/") { display.removeLast() }
        return display
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
        case "read", "write", "edit":
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
            argsFormatted: formatArguments(args),
            filePath: filePath(forTool: info.name, arguments: args)
        )
    }
}
