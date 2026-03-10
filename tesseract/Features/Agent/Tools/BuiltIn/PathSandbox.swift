import Foundation

// MARK: - PathSandboxError

nonisolated enum PathSandboxError: LocalizedError {
    case outsideSandbox(String)
    case pathTraversal(String)
    case symlinkEscape(String, target: String)
    case fileNotFound(String)
    case isDirectory(String)
    case notDirectory(String)

    var errorDescription: String? {
        switch self {
        case .outsideSandbox(let path):
            "Path is outside the sandbox: \(path)"
        case .pathTraversal(let path):
            "Path traversal detected: \(path)"
        case .symlinkEscape(let path, let target):
            "Symlink escapes sandbox: \(path) → \(target)"
        case .fileNotFound(let path):
            "File not found: \(path)"
        case .isDirectory(let path):
            "Path is a directory, not a file: \(path)"
        case .notDirectory(let path):
            "Path is not a directory: \(path)"
        }
    }
}

// MARK: - PathSandbox

/// Sandboxed path resolution for agent file tools.
/// All paths are confined to a root directory — escapes via traversal or symlinks are rejected.
nonisolated struct PathSandbox: Sendable {
    let root: URL

    private static nonisolated(unsafe) let fileManager = FileManager.default

    /// Default sandbox root for the agent's working directory.
    static var defaultRoot: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("Tesseract Agent/agent", isDirectory: true)
    }

    /// Resolve a user-provided path to an absolute URL within the sandbox.
    /// - Relative paths resolve against root
    /// - Absolute paths must be within root
    /// - Path traversal that escapes sandbox is rejected
    /// - Symlinks pointing outside sandbox are rejected
    func resolve(_ path: String) throws -> URL {
        // Trailing slash ensures "agent2" doesn't prefix-match "agent/"
        let rootPath = root.standardizedFileURL.path.hasSuffix("/")
            ? root.standardizedFileURL.path
            : root.standardizedFileURL.path + "/"

        // Build the absolute URL
        let absolute: URL
        if path.hasPrefix("/") {
            absolute = URL(fileURLWithPath: path)
        } else {
            absolute = root.appendingPathComponent(path)
        }

        // Standardize to resolve . and ..
        let standardized = absolute.standardizedFileURL

        // Check that the standardized path is within the sandbox.
        // Allow exact root match, or require the path starts with root + "/"
        let standardizedPath = standardized.path
        guard standardizedPath == root.standardizedFileURL.path
            || standardizedPath.hasPrefix(rootPath) else
        {
            if path.contains("..") {
                throw PathSandboxError.pathTraversal(path)
            }
            throw PathSandboxError.outsideSandbox(path)
        }

        // Walk existing path components and resolve symlinks to catch symlinked
        // directories that point outside the sandbox — even when the final file
        // doesn't exist yet (important for writes through symlinked dirs).
        try checkSymlinks(standardized, rootPath: rootPath, userPath: path)

        return standardized
    }

    /// Walk up from the target path resolving symlinks on each existing ancestor.
    /// This catches symlinked intermediate directories even when the leaf doesn't exist.
    private func checkSymlinks(_ url: URL, rootPath: String, userPath: String) throws {
        let resolvedRoot = root.resolvingSymlinksInPath().path
        let resolvedRootSlash = resolvedRoot.hasSuffix("/") ? resolvedRoot : resolvedRoot + "/"

        // Resolve the deepest existing ancestor
        var current = url
        while !Self.fileManager.fileExists(atPath: current.path) {
            let parent = current.deletingLastPathComponent()
            if parent.path == current.path { break } // reached filesystem root
            current = parent
        }

        let resolved = current.resolvingSymlinksInPath()
        let resolvedPath = resolved.path

        // The deepest existing ancestor is inside the sandbox — valid.
        if resolvedPath == resolvedRoot || resolvedPath.hasPrefix(resolvedRootSlash) {
            return
        }

        // The deepest existing ancestor is an ancestor of the resolved sandbox
        // root.  This means the root hasn't been fully created yet, but the
        // existing portion of the path resolves to a real prefix of it —
        // no symlink escape.
        let resolvedPathSlash = resolvedPath.hasSuffix("/") ? resolvedPath : resolvedPath + "/"
        if resolvedRoot.hasPrefix(resolvedPathSlash) {
            return
        }

        throw PathSandboxError.symlinkEscape(userPath, target: resolved.path)
    }

    /// Resolve and verify the file exists (and is not a directory).
    func resolveExisting(_ path: String) throws -> URL {
        let url = try resolve(path)

        var isDir: ObjCBool = false
        guard Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDir) else {
            throw PathSandboxError.fileNotFound(path)
        }
        if isDir.boolValue {
            throw PathSandboxError.isDirectory(path)
        }

        return url
    }

    /// Resolve for write — parent directory must be within sandbox.
    func resolveForWrite(_ path: String) throws -> URL {
        let url = try resolve(path)
        let parent = url.deletingLastPathComponent().standardizedFileURL
        let rootPath = root.standardizedFileURL.path
        let rootPathSlash = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"

        guard parent.path == rootPath || parent.path.hasPrefix(rootPathSlash) else {
            throw PathSandboxError.outsideSandbox(path)
        }

        return url
    }

    /// Get a display path relative to sandbox root.
    func displayPath(_ absolutePath: URL) -> String {
        let fullPath = absolutePath.standardizedFileURL.path
        let rootPath = root.standardizedFileURL.path
        let rootPathSlash = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"
        if fullPath == rootPath {
            return "."
        }
        if fullPath.hasPrefix(rootPathSlash) {
            return String(fullPath.dropFirst(rootPathSlash.count))
        }
        return fullPath
    }
}
