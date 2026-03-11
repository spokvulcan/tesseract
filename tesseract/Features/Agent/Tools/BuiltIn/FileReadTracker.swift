import Foundation

/// Tracks which files have been read during an agent session.
/// Write and edit tools check this before modifying files to prevent blind overwrites.
nonisolated final class FileReadTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var paths: Set<String> = []

    func record(_ absolutePath: String) {
        lock.lock()
        paths.insert(absolutePath)
        lock.unlock()
    }

    func hasRead(_ absolutePath: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return paths.contains(absolutePath)
    }
}
