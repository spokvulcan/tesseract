import Foundation

// MARK: - PackageRegistry

/// Discovers, loads, and manages agent packages from bundled and user directories.
///
/// Packages are directories containing a `package.json` manifest. The registry scans
/// known locations, resolves relative paths, and aggregates resources (skills, prompts,
/// extensions, seed data) from all enabled packages.
@MainActor
final class PackageRegistry {

    private enum Defaults {
        static let manifestFilename = "package.json"
        static let packageOverridesKey = "agent.packageOverrides"
    }

    /// Loaded packages keyed by name, in insertion order.
    private var packages: [(name: String, resolved: ResolvedPackage)] = []

    /// User overrides: true = force-enabled, false = force-disabled.
    /// Packages without an override use their manifest `enabled` value.
    private var userOverrides: [String: Bool]

    init() {
        if let stored = UserDefaults.standard.dictionary(forKey: Defaults.packageOverridesKey) as? [String: Bool] {
            self.userOverrides = stored
        } else {
            self.userOverrides = [:]
        }
    }

    // MARK: - Loading

    /// Scan bundled and user directories for packages, replacing any previously loaded state.
    func loadPackages(bundledPackagesDir: URL, userPackagesDir: URL) {
        packages.removeAll()

        let bundled = discoverPackages(in: bundledPackagesDir)
        let user = discoverPackages(in: userPackagesDir)

        var seenNames = Set<String>()
        for resolved in bundled + user {
            // Deduplicate: first occurrence wins (bundled beats user).
            guard seenNames.insert(resolved.manifest.name).inserted else {
                Log.agent.warning("[PackageRegistry] Skipping duplicate package: \(resolved.manifest.name)")
                continue
            }
            packages.append((name: resolved.manifest.name, resolved: resolved))
        }

        Log.agent.info("[PackageRegistry] Loaded \(packages.count) package(s): \(packages.map(\.name).joined(separator: ", "))")
    }

    // MARK: - Queries

    /// All packages that are currently enabled.
    /// User overrides take precedence over the manifest `enabled` value.
    var enabledPackages: [ResolvedPackage] {
        packages
            .filter { userOverrides[$0.name] ?? $0.resolved.manifest.enabled }
            .map(\.resolved)
    }

    var allSkillPaths: [URL] {
        enabledPackages.flatMap(\.skillPaths)
    }

    var allPromptAppendPaths: [URL] {
        enabledPackages.flatMap(\.promptAppendPaths)
    }

    var allContextFilePaths: [URL] {
        enabledPackages.flatMap(\.contextFilePaths)
    }

    var allSeedFilePaths: [URL] {
        enabledPackages.flatMap(\.seedFilePaths)
    }

    var allExtensionIdentifiers: [String] {
        enabledPackages.flatMap(\.extensionIdentifiers)
    }

    // MARK: - Enable / Disable

    /// Enable or disable a package by name. Persists the override to UserDefaults.
    func setEnabled(_ name: String, enabled: Bool) {
        userOverrides[name] = enabled
        UserDefaults.standard.set(userOverrides, forKey: Defaults.packageOverridesKey)
    }

    // MARK: - Seed Data

    /// Copy seed files for a package into `agentRoot` if they don't already exist.
    func seedDataIfNeeded(package: ResolvedPackage, agentRoot: URL) {
        let fm = FileManager.default
        for seedPath in package.seedFilePaths {
            let targetName = seedPath.lastPathComponent
            let targetURL = agentRoot.appendingPathComponent(targetName)
            guard !fm.fileExists(atPath: targetURL.path) else { continue }
            do {
                try fm.copyItem(at: seedPath, to: targetURL)
                Log.agent.info("[PackageRegistry] Seeded \(targetName)")
            } catch {
                Log.agent.warning("[PackageRegistry] Failed to seed \(targetName): \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Private — Discovery

    /// Scan a directory for subdirectories containing `package.json`, sorted alphabetically.
    private func discoverPackages(in directory: URL) -> [ResolvedPackage] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.path) else { return [] }

        guard let entries = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return entries
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { packageDir -> ResolvedPackage? in
                let manifestURL = packageDir.appendingPathComponent(Defaults.manifestFilename)
                return loadManifest(at: manifestURL, baseURL: packageDir)
            }
    }

    /// Decode a single `package.json` and resolve its paths.
    private func loadManifest(at url: URL, baseURL: URL) -> ResolvedPackage? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        do {
            let manifest = try JSONDecoder().decode(AgentPackageManifest.self, from: data)
            return ResolvedPackage(manifest: manifest, baseURL: baseURL)
        } catch {
            Log.agent.error("[PackageRegistry] Failed to decode \(url.path): \(error.localizedDescription)")
            return nil
        }
    }
}
