//
//  BrowserMCPScreenshotStore.swift
//  tesseract
//
//  Sidecar artifact store for Browser MCP telemetry (ADR-0031): the
//  actual image bytes a tool call returned (screenshots), saved as
//  ordinary files so the owner can open exactly what the model was
//  shown — the Playwright-trace model of "event stream + attached
//  artifacts". One directory per day under
//  `<telemetry dir>/artifacts/<yyyy-MM-dd>/`; the JSONL event records
//  each image's relative path.
//
//  Bounded two ways: day directories older than the newest
//  `retainedDayDirectories` are pruned at day-roll (mirroring the JSONL
//  retention), and a per-day byte budget stops saving (path becomes
//  nil; dimensions are still recorded) so a screenshot-heavy day cannot
//  fill the disk. The byte budget is tracked in memory and re-seeded
//  from the directory on day-roll — an app restart mid-day can
//  overshoot by at most one restart's worth, acceptable for a backstop.
//
//  MainActor-isolated (state), file I/O dispatched to a utility queue
//  so the server's request path never blocks on disk.
//

import Foundation

@MainActor
final class BrowserMCPScreenshotStore {

    static let retainedDayDirectories = 30
    /// Per-day byte budget across all artifacts (~2,800 screenshots at
    /// the ~90 KB observed size).
    static let maxBytesPerDay = 256 * 1024 * 1024

    private let root: URL
    private let queue = DispatchQueue(
        label: "app.tesseract.agent.browser-mcp-artifacts", qos: .utility)
    private var currentDay: String?
    private var currentDayBytes = 0

    private let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    /// - Parameter root: the artifacts root, conventionally
    ///   `<telemetry dir>/artifacts`.
    init(root: URL) {
        self.root = root
    }

    /// Persist one image and return its path **relative to the
    /// telemetry directory** (`artifacts/<day>/<name>.<ext>`), or nil
    /// when the day's byte budget is exhausted. The write itself is
    /// asynchronous; the returned path is deterministic so the caller
    /// can record it immediately.
    func save(_ data: Data, mimeType: String, timestamp: Date, name: String) -> String? {
        let day = dayFormatter.string(from: timestamp)
        if day != currentDay {
            currentDay = day
            currentDayBytes = seededBytes(day: day)
            prune()
        }
        guard currentDayBytes + data.count <= Self.maxBytesPerDay else { return nil }
        currentDayBytes += data.count

        let filename = "\(name).\(Self.fileExtension(for: mimeType))"
        let directory = root.appendingPathComponent(day, isDirectory: true)
        let url = directory.appendingPathComponent(filename, isDirectory: false)
        queue.async {
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try? data.write(to: url)
        }
        return "artifacts/\(day)/\(filename)"
    }

    /// Barrier for tests: returns after every previously enqueued
    /// artifact is on disk.
    func flushForTesting() {
        queue.sync {}
    }

    // MARK: - Bounds

    /// Bytes already on disk for `day` — re-seeds the in-memory budget
    /// after an app restart so the cap holds across launches.
    private func seededBytes(day: String) -> Int {
        let directory = root.appendingPathComponent(day, isDirectory: true)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.reduce(0) { total, name in
            let path = directory.appendingPathComponent(name).path
            let size = (try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int
            return total + (size ?? 0)
        }
    }

    /// Delete day directories older than the newest
    /// `retainedDayDirectories` (name order is day order). Runs on the
    /// queue, once per day-roll.
    private func prune() {
        let root = self.root
        queue.async {
            let manager = FileManager.default
            let names = (try? manager.contentsOfDirectory(atPath: root.path)) ?? []
            let days = names.sorted(by: >)
            for name in days.dropFirst(Self.retainedDayDirectories) {
                try? manager.removeItem(at: root.appendingPathComponent(name))
            }
        }
    }

    nonisolated private static func fileExtension(for mimeType: String) -> String {
        switch mimeType.lowercased() {
        case "image/png": return "png"
        case "image/jpeg": return "jpg"
        case "image/webp": return "webp"
        default: return "img"
        }
    }
}
