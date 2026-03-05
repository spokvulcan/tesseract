import Foundation

// MARK: - ExtensionHost

/// Central registry for agent extensions. Manages lifecycle and aggregates tools.
/// Extensions are registered at app startup; registration order determines
/// event dispatch order and tool conflict resolution (first wins).
@MainActor
final class ExtensionHost {
    private var extensions: [any AgentExtension] = []

    /// Register an extension. First registration wins on tool name conflicts.
    func register(_ ext: any AgentExtension) {
        if extensions.contains(where: { $0.path == ext.path }) {
            Log.agent.warning("[ExtensionHost] Extension already registered: \(ext.path)")
            return
        }
        extensions.append(ext)
        Log.agent.info("[ExtensionHost] Registered extension: \(ext.path)")
    }

    /// Unregister an extension by its path.
    func unregister(path: String) {
        extensions.removeAll { $0.path == path }
        Log.agent.info("[ExtensionHost] Unregistered extension: \(path)")
    }

    /// All registered extensions in registration order.
    var registeredExtensions: [any AgentExtension] {
        extensions
    }

    /// Look up a specific extension by path.
    func getExtension(path: String) -> (any AgentExtension)? {
        extensions.first { $0.path == path }
    }

    /// Collect tools from all extensions in registration order.
    /// Deduplication keys off `tool.name` (the canonical identity).
    /// First registration wins on name conflicts (duplicates log a warning).
    func aggregatedTools() -> [AgentToolDefinition] {
        var seen = Set<String>()
        var result: [AgentToolDefinition] = []

        for ext in extensions {
            for (key, tool) in ext.tools {
                if key != tool.name {
                    Log.agent.warning(
                        "[ExtensionHost] Key/name mismatch in '\(ext.path)': key='\(key)' name='\(tool.name)' — using tool.name"
                    )
                }
                if seen.contains(tool.name) {
                    Log.agent.warning(
                        "[ExtensionHost] Duplicate tool '\(tool.name)' from extension '\(ext.path)' — skipped"
                    )
                    continue
                }
                seen.insert(tool.name)
                result.append(tool)
            }
        }

        return result
    }
}
