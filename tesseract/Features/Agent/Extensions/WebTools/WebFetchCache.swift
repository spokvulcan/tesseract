import Foundation

// MARK: - WebFetchCache

/// Simple TTL cache for web fetch results.
/// Prevents redundant fetches when the agent re-reads a page within a session.
actor WebFetchCache {
    static let shared = WebFetchCache()

    private struct Entry {
        let content: WebContentExtractor.ExtractedContent
        let fetchedAt: ContinuousClock.Instant
    }

    private var entries: [String: Entry] = [:]
    private let ttl: Duration = .seconds(15 * 60)  // 15 minutes
    private let maxEntries = 50

    func get(url: String) -> WebContentExtractor.ExtractedContent? {
        guard let entry = entries[url] else { return nil }
        if ContinuousClock.now - entry.fetchedAt > ttl {
            entries.removeValue(forKey: url)
            return nil
        }
        return entry.content
    }

    func set(url: String, content: WebContentExtractor.ExtractedContent) {
        if entries.count >= maxEntries {
            // Evict oldest entry
            if let oldest = entries.min(by: { $0.value.fetchedAt < $1.value.fetchedAt }) {
                entries.removeValue(forKey: oldest.key)
            }
        }
        entries[url] = Entry(content: content, fetchedAt: .now)
    }
}
