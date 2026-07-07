import Foundation

// MARK: - BrowserURL

/// Shared address normalization for every browser surface: the MCP
/// `navigate`/`fetch` tools and the user login window's address bar all accept a
/// bare host ("example.com") or a full URL and resolve it identically, so the
/// one rule lives here rather than being re-derived per call site.
nonisolated enum BrowserURL {

    /// Turn user- or agent-supplied text into a URL, defaulting a scheme-less
    /// address to `https://`. Returns nil for empty input.
    static func normalized(from raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.contains("://") { return URL(string: trimmed) }
        return URL(string: "https://\(trimmed)")
    }
}
