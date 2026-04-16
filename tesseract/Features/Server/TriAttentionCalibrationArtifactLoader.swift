import Foundation
import MLXLMCommon

enum TriAttentionCalibrationArtifactLoaderError: LocalizedError {
    case unreadableArtifact(URL, underlying: Error)

    var errorDescription: String? {
        switch self {
        case .unreadableArtifact(let url, let underlying):
            return "Failed to read TriAttention calibration artifact at \(url.path): \(underlying.localizedDescription)"
        }
    }
}

enum TriAttentionCalibrationArtifactLookupResult: Sendable {
    case loaded(
        artifact: TriAttentionCalibrationArtifact,
        identity: TriAttentionCalibrationArtifactIdentity,
        relativeResourcePath: String
    )
    case missing(expectedModelFingerprint: String, expectedURL: URL)
    case fingerprintMismatch(expectedModelFingerprint: String, actualModelFingerprint: String, url: URL)
    case unavailable(expectedModelFingerprint: String, attemptedURL: URL, errorDescription: String)
}

struct TriAttentionCalibrationArtifactLoader: Sendable {
    nonisolated static let resourceDirectoryName = "TriAttention/v1"

    let rootURL: URL

    nonisolated init(rootURL: URL? = nil) {
        self.rootURL = rootURL ?? Self.defaultRootURL()
    }

    nonisolated func lookup(
        modelFingerprint: String
    ) throws -> TriAttentionCalibrationArtifactLookupResult {
        let url = expectedURL(for: modelFingerprint)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .missing(expectedModelFingerprint: modelFingerprint, expectedURL: url)
        }
        return try load(url: url, expectedModelFingerprint: modelFingerprint)
    }

    nonisolated func expectedURL(for modelFingerprint: String) -> URL {
        rootURL.appendingPathComponent("\(modelFingerprint).pt", isDirectory: false)
    }

    nonisolated func load(
        url: URL,
        expectedModelFingerprint: String
    ) throws -> TriAttentionCalibrationArtifactLookupResult {
        let actualModelFingerprint = url.deletingPathExtension().lastPathComponent
        guard actualModelFingerprint == expectedModelFingerprint else {
            return .fingerprintMismatch(
                expectedModelFingerprint: expectedModelFingerprint,
                actualModelFingerprint: actualModelFingerprint,
                url: url
            )
        }

        let artifactData: Data
        do {
            artifactData = try Data(contentsOf: url, options: [.mappedIfSafe])
        } catch {
            throw TriAttentionCalibrationArtifactLoaderError.unreadableArtifact(url, underlying: error)
        }

        let artifact = try TriAttentionCalibrationArtifact.load(contentsOf: url)
        let identity = TriAttentionCalibrationArtifactIdentity.sha256(of: artifactData)
        return .loaded(
            artifact: artifact,
            identity: identity,
            relativeResourcePath: Self.relativeResourcePath(for: url)
        )
    }

    nonisolated static func defaultRootURL() -> URL {
        final class BundleLocator {}
        let bundle = Bundle(for: BundleLocator.self)
        let baseURL = bundle.resourceURL
            ?? bundle.bundleURL.appendingPathComponent("Contents/Resources", isDirectory: true)
        return baseURL.appendingPathComponent(resourceDirectoryName, isDirectory: true)
    }

    nonisolated static func relativeResourcePath(modelFingerprint: String) -> String {
        "\(resourceDirectoryName)/\(modelFingerprint).pt"
    }

    nonisolated static func relativeResourcePath(for url: URL) -> String {
        "\(resourceDirectoryName)/\(url.lastPathComponent)"
    }
}
