import Foundation
import WebKit

// MARK: - WebAddressGuard

/// SSRF guard for the *anonymous* web paths — the single home for this policy,
/// applied by both the browser `fetch` tool and the agent's `web_fetch` (which
/// routes through ``EphemeralPageReader``). The authenticated **Agent Browser**
/// deliberately omits it — it is the user's own browser and may reach localhost
/// — but the anonymous readers take an untrusted URL as a convenience and must
/// not be turned into a private-network probe.
nonisolated enum WebAddressGuard {

    enum GuardError: LocalizedError, Sendable {
        case invalidURL
        case privateAddress
        var errorDescription: String? {
            switch self {
            case .invalidURL: "Invalid URL — must be an http:// or https:// address"
            case .privateAddress: "Refusing to fetch a private/internal network address"
            }
        }
    }

    static func validate(_ url: URL) throws {
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
            url.user == nil, url.password == nil
        else { throw GuardError.invalidURL }
        guard let host = url.host?.lowercased() else { throw GuardError.invalidURL }
        guard !isPrivate(host: host) else { throw GuardError.privateAddress }
    }

    static func isPrivate(host: String) -> Bool {
        if ["localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"].contains(host) { return true }
        if host.hasPrefix("10.") || host.hasPrefix("192.168.") || host.hasPrefix("169.254.") {
            return true
        }
        if host.hasPrefix("172.") {
            let parts = host.split(separator: ".")
            if parts.count >= 2, let second = Int(parts[1]), (16...31).contains(second) {
                return true
            }
        }
        if host.hasPrefix("fc") || host.hasPrefix("fd") || host.hasPrefix("[fc")
            || host.hasPrefix("[fd")
        {
            return true
        }
        if host.hasSuffix(".local") || host.hasSuffix(".internal") { return true }
        return false
    }
}

// MARK: - EphemeralPageReader

/// The **Ephemeral Page**: a cookieless, profile-less render used by anonymous
/// `fetch`. A throwaway non-persistent `WebPage` executes the page's JavaScript
/// (so JS-heavy sites read correctly), the live DOM is distilled to Markdown,
/// and the page is discarded — nothing touches the Agent Profile and no window
/// is shown.
@MainActor
enum EphemeralPageReader {

    static func read(url: URL) async throws -> WebContentExtractor.ExtractedContent {
        try WebAddressGuard.validate(url)

        var config = WebPage.Configuration()
        config.websiteDataStore = .nonPersistent()

        // Reuse a BrowserTab purely as a render harness — no presenter, so no
        // window; the ephemeral data store keeps it outside the profile.
        let tab = BrowserTab(configuration: config)
        try await tab.navigate(to: url)
        return try await tab.pageContent()
    }
}
