//
//  RotatingJSONLWriter.swift
//  tesseract
//
//  Queue-confined rotating daily-JSONL appender — the file machinery
//  shared by `PromptCacheDiagnosticsFileSink` and `CompletionTraceLog`.
//  One file per day (`<prefix><yyyy-MM-dd>.jsonl`); when the current
//  day file would exceed `maxFileBytes` it is rotated once to `.old`
//  (replacing any previous rotation) and writing continues on a fresh
//  file.
//
//  All mutable state lives on the private serial queue; `append` is
//  callable from any thread and runs the caller's encode thunk on the
//  queue, so emitters never pay serialization on their own thread. A
//  failing disk drops the handle rather than taking emission down; the
//  next day-roll retries cleanly.
//
//  Files open with `O_APPEND` and each line lands as one `write(2)`
//  (issue #159): a second writer on the same file — another process, or
//  a parallel test runner — appends at EOF instead of overwriting at
//  this instance's stale offset, which is exactly how the 2026-07-05
//  trace file got its torn mid-record line.
//

import Foundation

nonisolated final class RotatingJSONLWriter: @unchecked Sendable {
    private let queue: DispatchQueue
    private let directory: URL
    private let filenamePrefix: String
    private let maxFileBytes: Int
    private let retainedDayFiles: Int?
    private let freshFilePreamble: (@Sendable () -> Data?)?
    private var handle: FileHandle?
    private var currentDay: String?
    private var currentFileBytes: Int = 0

    // Per-instance (not static): `DateFormatter` is not thread-safe and
    // each instance must stay confined to its own queue.
    private let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    /// - Parameters:
    ///   - retainedDayFiles: when non-nil, every day-roll prunes the
    ///     directory down to the newest N day files (each with its
    ///     `.old` rotation sibling). `nil` keeps everything.
    ///   - freshFilePreamble: encoded first line for a fresh (or
    ///     previously empty) day file — e.g. a schema header. Runs on
    ///     the queue; the trailing newline is appended here.
    init(
        directory: URL,
        queueLabel: String,
        filenamePrefix: String = "",
        maxFileBytes: Int,
        retainedDayFiles: Int? = nil,
        freshFilePreamble: (@Sendable () -> Data?)? = nil
    ) {
        self.queue = DispatchQueue(label: queueLabel, qos: .utility)
        self.directory = directory
        self.filenamePrefix = filenamePrefix
        self.maxFileBytes = maxFileBytes
        self.retainedDayFiles = retainedDayFiles
        self.freshFilePreamble = freshFilePreamble
    }

    deinit {
        try? handle?.close()
    }

    /// Append one JSON line. Callable from any thread; ordering follows
    /// enqueue order on the serial queue. `timestamp` picks the day
    /// file; `encodeLine` runs on the queue and may return `nil` to
    /// drop the line. The trailing newline is appended here.
    func append(timestamp: Date = Date(), encodeLine: @escaping @Sendable () -> Data?) {
        queue.async { [weak self] in
            self?.write(timestamp: timestamp, encodeLine: encodeLine)
        }
    }

    /// Barrier for tests: returns after every previously appended line
    /// is on disk.
    func flushForTesting() {
        queue.sync {}
    }

    // MARK: - Queue-confined

    private func write(timestamp: Date, encodeLine: () -> Data?) {
        guard let data = encodeLine() else { return }
        var encoded = data
        encoded.append(0x0A)

        let day = dayFormatter.string(from: timestamp)
        if day != currentDay {
            try? handle?.close()
            handle = nil
            currentDay = day
            pruneOldDayFiles()
            openCurrentFile(day: day)
        }
        if currentFileBytes + encoded.count > maxFileBytes {
            rotateCurrentFile(day: day)
        }
        guard let handle else { return }

        do {
            try handle.write(contentsOf: encoded)
            currentFileBytes += encoded.count
        } catch {
            try? handle.close()
            self.handle = nil
        }
    }

    private func fileURL(day: String) -> URL {
        directory.appendingPathComponent("\(filenamePrefix)\(day).jsonl", isDirectory: false)
    }

    private func rotateCurrentFile(day: String) {
        try? handle?.close()
        handle = nil
        let url = fileURL(day: day)
        let rotated = url.appendingPathExtension("old")
        try? FileManager.default.removeItem(at: rotated)
        try? FileManager.default.moveItem(at: url, to: rotated)
        openCurrentFile(day: day)
    }

    private func openCurrentFile(day: String) {
        let manager = FileManager.default
        do {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            return
        }
        let url = fileURL(day: day)
        // O_APPEND: the kernel moves every write atomically to EOF, so a
        // concurrent writer on the same file (another process) can no longer
        // tear a record by landing inside this instance's stale offset.
        let fd = url.path.withCString {
            open($0, O_WRONLY | O_APPEND | O_CREAT, 0o644)
        }
        guard fd >= 0 else { return }
        let opened = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        let end = (try? opened.seekToEnd()) ?? 0
        handle = opened
        currentFileBytes = Int(end)
        if currentFileBytes == 0, let preamble = freshFilePreamble?() {
            var encoded = preamble
            encoded.append(0x0A)
            if (try? opened.write(contentsOf: encoded)) != nil {
                currentFileBytes += encoded.count
            }
        }
    }

    /// Keep the directory bounded: delete every day file (and its
    /// `.old` rotation sibling) older than the newest `retainedDayFiles`
    /// days. Name order is day order (`yyyy-MM-dd`). Runs once per
    /// day-roll — one directory listing per process in steady state.
    private func pruneOldDayFiles() {
        guard let retainedDayFiles else { return }
        let manager = FileManager.default
        let names = (try? manager.contentsOfDirectory(atPath: directory.path)) ?? []
        func dayFileName(_ name: String) -> String? {
            guard name.hasPrefix(filenamePrefix) else { return nil }
            if name.hasSuffix(".jsonl") { return name }
            if name.hasSuffix(".jsonl.old") { return String(name.dropLast(4)) }
            return nil
        }
        let dayFiles = Set(names.compactMap(dayFileName))
        let keep = Set(dayFiles.sorted(by: >).prefix(retainedDayFiles))
        for name in names {
            guard let dayFile = dayFileName(name), !keep.contains(dayFile) else { continue }
            try? manager.removeItem(at: directory.appendingPathComponent(name))
        }
    }
}
