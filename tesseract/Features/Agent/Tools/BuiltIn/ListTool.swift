import Foundation
import MLXLMCommon

// MARK: - ListToolDetails

nonisolated struct ListToolDetails: Sendable, Hashable {
    let path: String
    let entryCount: Int
    let wasTruncated: Bool
}

// MARK: - ListTool Factory

private nonisolated enum Defaults {
    static let defaultLimit = 200
    static let hardCap = 500
}

nonisolated func createListTool(sandbox: PathSandbox) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "list",
        label: "List Directory",
        description: "List files and directories. Shows tree-style output with file sizes.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "path": PropertySchema(
                    type: "string",
                    description: "Directory path (defaults to working directory root)"
                ),
                "recursive": PropertySchema(
                    type: "boolean",
                    description: "List recursively (default: false)"
                ),
                "limit": PropertySchema(
                    type: "integer",
                    description: "Maximum entries to return (default: 200, max: 500)"
                ),
            ],
            required: []
        ),
        execute: { _, argsJSON, _, _ in
            let pathArg = ToolArgExtractor.string(argsJSON, key: "path")
            let recursive = ToolArgExtractor.bool(argsJSON, key: "recursive") ?? false
            let limitArg = ToolArgExtractor.int(argsJSON, key: "limit")
            let limit = max(0, min(limitArg ?? Defaults.defaultLimit, Defaults.hardCap))

            let url: URL
            if let pathArg {
                url = try sandbox.resolve(pathArg)
            } else {
                url = sandbox.root.standardizedFileURL
            }

            // Verify it's a directory
            var isDir: ObjCBool = false
            let fm = FileManager.default
            guard fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
                throw PathSandboxError.notDirectory(sandbox.displayPath(url))
            }

            let displayRoot = sandbox.displayPath(url)

            if recursive {
                return ListToolHelper.listRecursive(
                    url: url, sandbox: sandbox, displayRoot: displayRoot,
                    limit: limit, fm: fm)
            } else {
                return ListToolHelper.listFlat(
                    url: url, sandbox: sandbox, displayRoot: displayRoot,
                    limit: limit, fm: fm)
            }
        }
    )
}

// MARK: - Helper (nonisolated)

private nonisolated enum ListToolHelper: Sendable {

    // MARK: Non-recursive listing

    static func listFlat(
        url: URL, sandbox: PathSandbox, displayRoot: String,
        limit: Int, fm: FileManager
    ) -> AgentToolResult {
        let keys: [URLResourceKey] = [.isDirectoryKey, .fileSizeKey]
        let contents: [URL]
        do {
            contents = try fm.contentsOfDirectory(
                at: url, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles]
            ).sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
        } catch {
            return .error("Failed to list directory: \(error.localizedDescription)")
        }

        let totalCount = contents.count
        let entries = Array(contents.prefix(limit))
        let wasTruncated = totalCount > limit

        var lines: [String] = ["\(displayRoot)/"]
        for (i, entry) in entries.enumerated() {
            let isLast = (i == entries.count - 1) && !wasTruncated
            let connector = isLast ? "└── " : "├── "
            lines.append(connector + entryLabel(entry, fm: fm))
        }

        if wasTruncated {
            lines.append("└── [Showing \(entries.count) of \(totalCount) entries. Use limit=\(min(totalCount, Defaults.hardCap)) to see more.]")
        }

        return AgentToolResult(
            content: [.text(lines.joined(separator: "\n"))],
            details: ListToolDetails(
                path: displayRoot,
                entryCount: entries.count,
                wasTruncated: wasTruncated
            )
        )
    }

    // MARK: Recursive listing

    /// Represents a node in the directory tree for recursive output.
    struct TreeNode {
        let name: String
        let isDirectory: Bool
        let size: Int?
        var children: [TreeNode] = []
    }

    static func listRecursive(
        url: URL, sandbox: PathSandbox, displayRoot: String,
        limit: Int, fm: FileManager
    ) -> AgentToolResult {
        let keys: [URLResourceKey] = [.isDirectoryKey, .fileSizeKey]
        guard let enumerator = fm.enumerator(
            at: url, includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else {
            return .error("Failed to enumerate directory")
        }

        // Collect entries up to limit
        var collected: [(url: URL, depth: [String])] = []
        var totalCount = 0
        let rootPath = url.standardizedFileURL.path
        let rootSlash = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"

        for case let entry as URL in enumerator {
            totalCount += 1
            if collected.count < limit {
                let entryPath = entry.standardizedFileURL.path
                let relative: String
                if entryPath.hasPrefix(rootSlash) {
                    relative = String(entryPath.dropFirst(rootSlash.count))
                } else {
                    relative = entry.lastPathComponent
                }
                let components = relative.split(separator: "/").map(String.init)
                collected.append((url: entry, depth: components))
            }
        }

        let wasTruncated = totalCount > limit

        // Build tree structure
        var root = TreeNode(name: displayRoot, isDirectory: true, size: nil)
        for item in collected {
            insertIntoTree(node: &root, components: item.depth, url: item.url, fm: fm)
        }

        // Render
        var lines: [String] = ["\(displayRoot)/"]
        renderTree(node: root, prefix: "", isRoot: true, lines: &lines)

        if wasTruncated {
            lines.append("[Showing \(collected.count) of \(totalCount) entries. Use limit=\(min(totalCount, Defaults.hardCap)) to see more.]")
        }

        return AgentToolResult(
            content: [.text(lines.joined(separator: "\n"))],
            details: ListToolDetails(
                path: displayRoot,
                entryCount: collected.count,
                wasTruncated: wasTruncated
            )
        )
    }

    static func insertIntoTree(
        node: inout TreeNode, components: [String], url: URL, fm: FileManager
    ) {
        guard let first = components.first else { return }

        if let idx = node.children.firstIndex(where: { $0.name == first }) {
            if components.count > 1 {
                insertIntoTree(
                    node: &node.children[idx],
                    components: Array(components.dropFirst()),
                    url: url, fm: fm)
            }
        } else {
            if components.count == 1 {
                // Leaf entry
                var isDir: ObjCBool = false
                fm.fileExists(atPath: url.path, isDirectory: &isDir)
                let size: Int?
                if !isDir.boolValue {
                    size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
                } else {
                    size = nil
                }
                node.children.append(TreeNode(
                    name: first, isDirectory: isDir.boolValue, size: size))
            } else {
                // Intermediate directory
                var dirNode = TreeNode(name: first, isDirectory: true, size: nil)
                insertIntoTree(
                    node: &dirNode,
                    components: Array(components.dropFirst()),
                    url: url, fm: fm)
                node.children.append(dirNode)
            }
        }
    }

    static func renderTree(
        node: TreeNode, prefix: String, isRoot: Bool, lines: inout [String]
    ) {
        let sorted = node.children.sorted {
            // Directories first, then alphabetical
            if $0.isDirectory != $1.isDirectory { return $0.isDirectory }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }

        for (i, child) in sorted.enumerated() {
            let isLast = i == sorted.count - 1
            let connector = isLast ? "└── " : "├── "
            let childPrefix = isLast ? "    " : "│   "
            lines.append(prefix + connector + entryLabel(for: child))

            if child.isDirectory {
                renderTree(
                    node: child, prefix: prefix + childPrefix,
                    isRoot: false, lines: &lines)
            }
        }
    }

    // MARK: Formatting

    static func entryLabel(for node: TreeNode) -> String {
        if node.isDirectory {
            return "\(node.name)/"
        }
        if let size = node.size {
            return "\(node.name) (\(formatSize(size)))"
        }
        return node.name
    }

    static func entryLabel(_ url: URL, fm: FileManager) -> String {
        var isDir: ObjCBool = false
        fm.fileExists(atPath: url.path, isDirectory: &isDir)
        let name = url.lastPathComponent
        if isDir.boolValue {
            return "\(name)/"
        }
        if let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
           let size = values.fileSize
        {
            return "\(name) (\(formatSize(size)))"
        }
        return name
    }

    static func formatSize(_ bytes: Int) -> String {
        if bytes < 1_000 {
            return "\(bytes) B"
        } else if bytes < 1_000_000 {
            let kb = Double(bytes) / 1_000.0
            return kb < 10 ? String(format: "%.1f KB", kb) : String(format: "%.0f KB", kb)
        } else {
            let mb = Double(bytes) / 1_000_000.0
            return mb < 10 ? String(format: "%.1f MB", mb) : String(format: "%.0f MB", mb)
        }
    }
}
