import Foundation

// MARK: - PackageBootstrap

/// Coordinates package loading, data seeding, and extension registration at startup.
///
/// Called once during app initialization (or when packages change). Bridges the
/// `PackageRegistry` to the `ExtensionHost`, `SkillRegistry`, and prompt systems.
@MainActor
enum PackageBootstrap {

    /// Extension paths registered by a previous bootstrap call, used for cleanup on reload.
    private static var packageManagedExtensionPaths: [String] = []

    /// Load packages, seed data, and register extensions.
    static func bootstrap(
        packageRegistry: PackageRegistry,
        extensionHost: ExtensionHost,
        agentRoot: URL,
        settingsManager: SettingsManager
    ) {
        // 1. Discover and load packages from bundled and user directories.
        let bundledDir = Bundle.main.url(forResource: "AgentPackages", withExtension: nil)
        let userDir = agentRoot.appendingPathComponent("packages")
        packageRegistry.loadPackages(
            bundledPackagesDir: bundledDir ?? agentRoot,
            userPackagesDir: userDir
        )

        // 2. Seed data files for enabled packages (never overwrites existing).
        for package in packageRegistry.enabledPackages {
            packageRegistry.seedDataIfNeeded(package: package, agentRoot: agentRoot)
        }

        // 2.5 Copy skill files into sandbox cache (package-managed, always overwritten).
        for package in packageRegistry.enabledPackages {
            copySkillsToCache(package: package, agentRoot: agentRoot)
        }

        // 3. Unregister previously package-managed extensions, then re-register current ones.
        for path in packageManagedExtensionPaths {
            extensionHost.unregister(path: path)
        }
        let currentExtIds = packageRegistry.allExtensionIdentifiers
        var registeredPaths: [String] = []
        for extId in currentExtIds {
            switch extId {
            case "PersonalAssistantExtension":
                let ext = PersonalAssistantExtension()
                extensionHost.register(ext)
                registeredPaths.append(ext.path)
            default:
                Log.agent.warning("[PackageBootstrap] Unknown extension identifier: \(extId)")
            }
        }

        // Register standalone extensions (not package-managed)
        if settingsManager.webAccessEnabled {
            let webExt = WebToolsExtension()
            extensionHost.register(webExt)
            registeredPaths.append(webExt.path)
        }

        packageManagedExtensionPaths = registeredPaths
    }

    /// Collect skill file paths from all enabled packages.
    static func skillPaths(from registry: PackageRegistry) -> [URL] {
        registry.allSkillPaths
    }

    /// Read and return prompt append content from all enabled packages.
    static func promptAppends(from registry: PackageRegistry) -> [String] {
        registry.allPromptAppendPaths.compactMap { url -> String? in
            do {
                return try String(contentsOf: url, encoding: .utf8)
            } catch {
                Log.agent.warning("[PackageBootstrap] Failed to read prompt append \(url.lastPathComponent): \(error.localizedDescription)")
                return nil
            }
        }
    }

    // MARK: - Skill Cache

    /// Return sandbox-local cached paths for all skill files from enabled packages.
    static func cachedSkillPaths(from registry: PackageRegistry, agentRoot: URL) -> [URL] {
        let fm = FileManager.default
        let cacheRoot = agentRoot.appendingPathComponent(".cache/packages", isDirectory: true)
        return registry.enabledPackages.flatMap { package -> [URL] in
            let packageCache = cacheRoot.appendingPathComponent(package.manifest.name, isDirectory: true)
            return package.skillPaths.compactMap { originalURL -> URL? in
                guard let relativePath = relativePath(of: originalURL, to: package.baseURL) else {
                    return nil
                }
                let cached = packageCache.appendingPathComponent(relativePath)
                return fm.fileExists(atPath: cached.path) ? cached : nil
            }
        }
    }

    /// Copy skill directories from the bundle into `{agentRoot}/.cache/packages/{name}/`.
    ///
    /// Each skill file's containing directory is copied wholesale so sibling resources
    /// (templates, references) are available for the agent's `read` tool.
    private static func copySkillsToCache(package: ResolvedPackage, agentRoot: URL) {
        let fm = FileManager.default
        let packageCache = agentRoot
            .appendingPathComponent(".cache/packages", isDirectory: true)
            .appendingPathComponent(package.manifest.name, isDirectory: true)

        // Clean stale cache from previous versions.
        if fm.fileExists(atPath: packageCache.path) {
            try? fm.removeItem(at: packageCache)
        }

        // Collect unique skill directories to copy.
        var copiedDirs = Set<String>()
        for skillURL in package.skillPaths {
            guard let relative = relativePath(of: skillURL, to: package.baseURL) else { continue }

            // Skill directory is the parent of the skill file.
            let skillRelativeDir = (relative as NSString).deletingLastPathComponent
            guard !skillRelativeDir.isEmpty, !copiedDirs.contains(skillRelativeDir) else { continue }
            copiedDirs.insert(skillRelativeDir)

            let sourceDir = package.baseURL.appendingPathComponent(skillRelativeDir, isDirectory: true)
            let destDir = packageCache.appendingPathComponent(skillRelativeDir, isDirectory: true)

            do {
                try fm.createDirectory(at: destDir.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fm.copyItem(at: sourceDir, to: destDir)
            } catch {
                Log.agent.warning("[PackageBootstrap] Failed to cache skill dir \(skillRelativeDir): \(error.localizedDescription)")
            }
        }
    }

    /// Compute the relative path of `url` within `base`, or nil if not contained.
    private static func relativePath(of url: URL, to base: URL) -> String? {
        let urlPath = url.standardizedFileURL.path
        let basePath = base.standardizedFileURL.path.hasSuffix("/")
            ? base.standardizedFileURL.path
            : base.standardizedFileURL.path + "/"
        guard urlPath.hasPrefix(basePath) else { return nil }
        return String(urlPath.dropFirst(basePath.count))
    }
}
