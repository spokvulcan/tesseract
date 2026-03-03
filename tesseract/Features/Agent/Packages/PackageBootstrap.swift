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
        agentRoot: URL
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
}
