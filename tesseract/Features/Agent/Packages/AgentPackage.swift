import Foundation

// MARK: - AgentPackageManifest

/// Decoded representation of a package's `package.json` manifest.
struct AgentPackageManifest: Codable, Sendable {
    let name: String
    let enabled: Bool
    let skills: [String]?
    let promptAppendFiles: [String]?
    let extensions: [String]?
    let seedFiles: [String]?
    let contextFiles: [String]?
}

// MARK: - ResolvedPackage

/// A package manifest with all relative paths resolved to absolute URLs.
struct ResolvedPackage: Sendable {
    let manifest: AgentPackageManifest
    /// Directory containing `package.json`.
    let baseURL: URL

    let skillPaths: [URL]
    let promptAppendPaths: [URL]
    let extensionIdentifiers: [String]
    let seedFilePaths: [URL]
    let contextFilePaths: [URL]
}

extension ResolvedPackage {
    /// Resolve a manifest's relative paths against the package base directory.
    init(manifest: AgentPackageManifest, baseURL: URL) {
        self.manifest = manifest
        self.baseURL = baseURL
        self.skillPaths = (manifest.skills ?? []).map { baseURL.appendingPathComponent($0) }
        self.promptAppendPaths = (manifest.promptAppendFiles ?? []).map { baseURL.appendingPathComponent($0) }
        self.extensionIdentifiers = manifest.extensions ?? []
        self.seedFilePaths = (manifest.seedFiles ?? []).map { baseURL.appendingPathComponent($0) }
        self.contextFilePaths = (manifest.contextFiles ?? []).map { baseURL.appendingPathComponent($0) }
    }
}
