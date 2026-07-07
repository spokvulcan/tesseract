import Foundation

// MARK: - MCPServerConfig

/// One configured MCP server the agent connects to. Persisted (user servers) via
/// the Settings Catalogue; the built-in Browser server is synthesized, not
/// stored (US #2 — pre-registered, un-removable). `Codable` for persistence,
/// `Hashable` so the manager can detect a materially-changed config and
/// reconnect.
nonisolated struct MCPServerConfig: Codable, Sendable, Hashable, Identifiable {

    /// How the client reaches the server.
    enum Transport: String, Codable, Sendable {
        /// Streamable HTTP (+ SSE) over a URL — arbitrary user-configured servers.
        case http
        /// The in-app Browser MCP server, reached in-process (no socket), so
        /// browser-use in chat never depends on the inference HTTP listener.
        case inProcessBrowser
    }

    var id: String
    var name: String
    var url: String
    var enabled: Bool
    var headers: [String: String]
    var transport: Transport

    init(
        id: String = UUID().uuidString,
        name: String,
        url: String,
        enabled: Bool = false,
        headers: [String: String] = [:],
        transport: Transport = .http
    ) {
        self.id = id
        self.name = name
        self.url = url
        self.enabled = enabled
        self.headers = headers
        self.transport = transport
    }

    // MARK: - Namespacing

    /// The tool namespace for this server (US #8), derived from its name so it is
    /// stable and human-legible in the tool list. The built-in Browser server is
    /// pinned to ``browserNamespace`` regardless of its display name, so the
    /// web-access gated set (US #16) can never drift from the tools it
    /// materializes even if that name is later changed.
    var namespace: String { isBuiltInBrowser ? Self.browserNamespace : Self.sanitize(name) }

    /// Separator between a server's namespace and a tool's own name. The single
    /// source of truth — the tool adapter and the web-access gating both build
    /// namespaced names through ``namespacedToolName(namespace:tool:)`` so they
    /// can never drift.
    static let namespaceSeparator = "."

    static func namespacedToolName(namespace: String, tool: String) -> String {
        namespace.isEmpty ? tool : "\(namespace)\(namespaceSeparator)\(tool)"
    }

    // MARK: - Built-in Browser server

    static let browserServerID = "builtin-browser"
    static let browserNamespace = "browser"

    var isBuiltInBrowser: Bool { id == Self.browserServerID }

    /// The synthetic Browser MCP server entry. Its `enabled` tracks
    /// `browserMCPServerEnabled` (the one "Browser Access" switch): flip it on
    /// and the agent's 12 browser tools appear in chat; off and they vanish.
    static func builtInBrowser(enabled: Bool) -> MCPServerConfig {
        MCPServerConfig(
            id: browserServerID, name: "Browser", url: "", enabled: enabled, headers: [:],
            transport: .inProcessBrowser)
    }

    /// The namespaced names of the built-in Browser server's tools — the single
    /// source of truth for the tool set the web-access switch gates (US #16).
    /// Static because gating applies by name whether or not the server is
    /// currently connected; ``namespace`` pins the live tools to the same
    /// ``browserNamespace``, so the two match by construction (a test asserts it).
    static var browserToolNames: Set<String> {
        Set(
            BrowserToolCatalog.all.map {
                namespacedToolName(namespace: browserNamespace, tool: $0.name)
            })
    }

    // MARK: - Helpers

    /// ASCII-lowercase, non-alphanumerics → `_`, collapsed and trimmed. Keeps a
    /// server namespace safe to compose into a tool name.
    static func sanitize(_ raw: String) -> String {
        let mapped = raw.lowercased().map { character -> Character in
            (character.isASCII && (character.isLetter || character.isNumber)) ? character : "_"
        }
        var result = String(mapped)
        while result.contains("__") {
            result = result.replacingOccurrences(of: "__", with: "_")
        }
        result = result.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return result.isEmpty ? "server" : result
    }
}
