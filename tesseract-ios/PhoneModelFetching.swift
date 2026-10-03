//
//  PhoneModelFetching.swift
//  tesseract-ios
//
//  The phone's Model Fetching adapter (#515): Hugging Face files over a
//  background URLSession, so the voice goes on downloading with the screen
//  locked, a dropped connection resumes mid-file, and a relaunched app
//  rejoins the transfer in flight. Wi-Fi only unless the owner allows
//  cellular. The download manager above the port is the Mac's.
//

import CryptoKit
import Foundation
import Observation

/// The bytes moving for the file that is downloading.
@Observable @MainActor
final class DownloadActivity {
    var path: String?
    var received: Int64 = 0
    var expected: Int64 = 0
}

@MainActor
final class PhoneModelFetching: RangedModelFetching {
    static let sessionIdentifier = "app.tesseract.agent.downloads"

    let activity: DownloadActivity
    private let transfers: BackgroundTransfers
    private let allowsCellular: @MainActor () -> Bool

    init(allowsCellular: @escaping @MainActor () -> Bool) {
        self.allowsCellular = allowsCellular
        let activity = DownloadActivity()
        self.activity = activity
        transfers = BackgroundTransfers(identifier: Self.sessionIdentifier) { received, expected in
            Task { @MainActor in
                activity.received = received
                if expected > 0 { activity.expected = expected }
            }
        }
    }

    /// The system's handler for a relaunch that delivered download events;
    /// called once the session has handed them all over.
    func handleEvents(completion: @escaping @Sendable () -> Void) {
        transfers.setEventsFinished(completion)
    }

    // MARK: - Model Fetching

    func listFiles(in repo: String, recursive: Bool) async throws -> [RemoteModelFile] {
        struct Entry: Decodable {
            let type: String
            let path: String
            let size: Int?
        }
        var components = URLComponents(
            string: "https://huggingface.co/api/models/\(repo)/tree/main")
        if recursive { components?.queryItems = [URLQueryItem(name: "recursive", value: "true")] }
        guard let url = components?.url else { throw ModelFetchingError.invalidRepository(repo) }
        let (data, response) = try await URLSession.shared.data(for: request(url))
        try Self.check(response, url: url)
        return try JSONDecoder().decode([Entry].self, from: data)
            .filter { $0.type == "file" }
            .map { RemoteModelFile(path: $0.path, size: $0.size) }
    }

    func fetchFile(at path: String, from repo: String, to destination: URL) async throws {
        try await fetch(path, from: repo, to: destination, length: nil)
    }

    func fetchFile(at path: String, from repo: String, to destination: URL, length: Int)
        async throws
    {
        try await fetch(path, from: repo, to: destination, length: length)
    }

    func fetchPrefix(of path: String, from repo: String, length: Int) async throws -> Data {
        let url = try Self.fileURL(path, in: repo)
        var request = request(url)
        request.setValue("bytes=0-\(length - 1)", forHTTPHeaderField: "Range")
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.check(response, url: url)
        guard data.count >= length else { throw URLError(.cannotDecodeContentData) }
        return data.prefix(length)
    }

    // MARK: - Transfers

    private func fetch(_ path: String, from repo: String, to destination: URL, length: Int?)
        async throws
    {
        activity.path = path
        activity.received = 0
        activity.expected = Int64(length ?? 0)
        defer { activity.path = nil }
        let url = try Self.fileURL(path, in: repo)
        var request = request(url)
        if let length { request.setValue("bytes=0-\(length - 1)", forHTTPHeaderField: "Range") }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try await transfers.download(request, to: destination)
        // A resumed ranged transfer can run on to the file's end.
        if let length, let size = try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize,
            size > length
        {
            let handle = try FileHandle(forWritingTo: destination)
            try handle.truncate(atOffset: UInt64(length))
            try handle.close()
        }
    }

    private func request(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        request.allowsCellularAccess = allowsCellular()
        return request
    }

    private static func fileURL(_ path: String, in repo: String) throws -> URL {
        let escaped = path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path
        guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(escaped)") else {
            throw ModelFetchingError.invalidRepository(repo)
        }
        return url
    }

    private static func check(_ response: URLResponse, url: URL) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw NSError(
                domain: "PhoneModelFetching", code: http.statusCode,
                userInfo: [
                    NSLocalizedDescriptionKey: "\(url.lastPathComponent): HTTP \(http.statusCode)"
                ])
        }
    }
}

/// The background URLSession and its delegate. A transfer's destination
/// rides in its task description, so a file finished while the app was
/// gone still lands where it belongs; a failure's resume data is kept on
/// disk, so the next try continues mid-file.
nonisolated final class BackgroundTransfers: NSObject, URLSessionDownloadDelegate,
    @unchecked Sendable
{
    private let lock = NSLock()
    private var session: URLSession!
    private var waiting: [Int: CheckedContinuation<Void, Error>] = [:]
    /// A finished file's move failure or bad HTTP status, until the task
    /// completes.
    private var failures: [Int: Error] = [:]
    private var retries: [String: Int] = [:]
    private var eventsFinished: (@Sendable () -> Void)?
    private let onProgress: @Sendable (Int64, Int64) -> Void
    private let resumeDirectory: URL

    /// Resumes in a row before a transfer gives up.
    static let retryLimit = 8

    init(identifier: String, onProgress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.onProgress = onProgress
        resumeDirectory = URL.cachesDirectory.appendingPathComponent(
            "download-resume", isDirectory: true)
        super.init()
        try? FileManager.default.createDirectory(
            at: resumeDirectory, withIntermediateDirectories: true)
        let configuration = URLSessionConfiguration.background(withIdentifier: identifier)
        configuration.sessionSendsLaunchEvents = true
        configuration.isDiscretionary = false
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    func setEventsFinished(_ completion: @escaping @Sendable () -> Void) {
        lock.withLock { eventsFinished = completion }
    }

    /// Downloads `request` into `destination`: joining a transfer of the same
    /// bytes already running, or continuing from resume data a failed one
    /// left.
    func download(_ request: URLRequest, to destination: URL) async throws {
        let key = Self.key(request)
        let running = await session.allTasks.first {
            $0.state == .running && $0.originalRequest.map(Self.key) == key
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let task: URLSessionTask
                if let running {
                    task = running
                } else if let data = resumeData(for: key) {
                    task = session.downloadTask(withResumeData: data)
                } else {
                    task = session.downloadTask(with: request)
                }
                task.taskDescription = destination.path
                lock.withLock { waiting[task.taskIdentifier] = continuation }
                task.resume()
            }
        } onCancel: { [weak self, session] in
            session?.getAllTasks { tasks in
                for task in tasks where task.originalRequest.map(Self.key) == key {
                    (task as? URLSessionDownloadTask)?.cancel { [weak self] data in
                        if let data { self?.storeResumeData(data, for: key) }
                    }
                }
            }
        }
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let id = downloadTask.taskIdentifier
        if let http = downloadTask.response as? HTTPURLResponse,
            !(200..<300).contains(http.statusCode)
        {
            lock.withLock { failures[id] = URLError(.badServerResponse) }
            return
        }
        guard let path = downloadTask.taskDescription else { return }
        let destination = URL(fileURLWithPath: path)
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: location, to: destination)
        } catch {
            lock.withLock { failures[id] = error }
        }
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?)
    {
        let id = task.taskIdentifier
        let key = task.originalRequest.map(Self.key) ?? ""
        let (continuation, failure) = lock.withLock {
            (waiting.removeValue(forKey: id), failures.removeValue(forKey: id))
        }
        guard let error = error ?? failure else {
            lock.withLock { retries[key] = nil }
            try? FileManager.default.removeItem(at: resumeURL(for: key))
            continuation?.resume()
            return
        }
        let resume = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        if let resume { storeResumeData(resume, for: key) }
        // A dropped connection: go on from where it stopped. The background
        // session waits for the network before the new task moves.
        let cancelled = (error as? URLError)?.code == .cancelled
        let attempt = lock.withLock { () -> Int in
            retries[key, default: 0] += 1
            return retries[key]!
        }
        if let continuation, let resume, !cancelled, attempt <= Self.retryLimit {
            let next = session.downloadTask(withResumeData: resume)
            next.taskDescription = task.taskDescription
            lock.withLock { waiting[next.taskIdentifier] = continuation }
            next.resume()
            return
        }
        continuation?.resume(throwing: cancelled ? CancellationError() : error)
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        let completion = lock.withLock { () -> (@Sendable () -> Void)? in
            defer { eventsFinished = nil }
            return eventsFinished
        }
        DispatchQueue.main.async { completion?() }
    }

    // MARK: - Resume data

    /// A transfer's identity: its URL and the bytes it asks for.
    private static func key(_ request: URLRequest) -> String {
        "\(request.url?.absoluteString ?? "") \(request.value(forHTTPHeaderField: "Range") ?? "")"
    }

    private func resumeURL(for key: String) -> URL {
        let digest = SHA256.hash(data: Data(key.utf8)).prefix(16).map { String(format: "%02x", $0) }
        return resumeDirectory.appendingPathComponent(digest.joined() + ".resume")
    }

    private func resumeData(for key: String) -> Data? {
        try? Data(contentsOf: resumeURL(for: key))
    }

    private func storeResumeData(_ data: Data, for key: String) {
        try? data.write(to: resumeURL(for: key))
    }
}
