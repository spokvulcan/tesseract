import Foundation
import Network
import Observation

// MARK: - HTTP Types

enum HTTPMethod: String, Sendable {
    case GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
}

struct HTTPRequest: Sendable {
    let method: HTTPMethod
    let path: String
    let headers: [(name: String, value: String)]
    let body: Data?

    func header(_ name: String) -> String? {
        let lower = name.lowercased()
        return headers.first(where: { $0.name.lowercased() == lower })?.value
    }
}

struct HTTPResponse: Sendable {
    let statusCode: Int
    let statusText: String
    let headers: [(name: String, value: String)]
    let body: Data?

    func serialized() -> Data {
        var head = "HTTP/1.1 \(statusCode) \(statusText)\r\n"
        for (name, value) in headers {
            head += "\(name): \(value)\r\n"
        }
        if let body, !headers.contains(where: { $0.name.lowercased() == "content-length" }) {
            head += "Content-Length: \(body.count)\r\n"
        }
        if !headers.contains(where: { $0.name.lowercased() == "connection" }) {
            head += "Connection: close\r\n"
        }
        head += "\r\n"
        var data = Data(head.utf8)
        if let body { data.append(body) }
        return data
    }

    // MARK: Factories

    static func json(_ value: some Encodable, status: Int = 200) -> HTTPResponse {
        let encoder = JSONEncoder()
        let body = (try? encoder.encode(value)) ?? Data("{}".utf8)
        return HTTPResponse(
            statusCode: status,
            statusText: statusText(for: status),
            headers: [("Content-Type", "application/json")],
            body: body
        )
    }

    /// JSON response from pre-encoded Data. Use when encoding must happen in an
    /// isolated context (e.g. MainActor) due to Swift 6.2 conformance inference.
    static func jsonBody(_ body: Data, status: Int = 200) -> HTTPResponse {
        HTTPResponse(
            statusCode: status,
            statusText: statusText(for: status),
            headers: [("Content-Type", "application/json")],
            body: body
        )
    }

    static func error(status: Int, message: String) -> HTTPResponse {
        let errorBody = OpenAIError(
            error: .init(message: message, type: errorType(for: status), code: status)
        )
        return json(errorBody, status: status)
    }

    static func badRequest(_ message: String) -> HTTPResponse { error(status: 400, message: message) }
    static func notFound() -> HTTPResponse { error(status: 404, message: "Not found") }
    static func internalError(_ message: String) -> HTTPResponse { error(status: 500, message: message) }
    static func serviceUnavailable(_ message: String) -> HTTPResponse { error(status: 503, message: message) }

    static func methodNotAllowed(allowed: [HTTPMethod]) -> HTTPResponse {
        let allow = allowed.map(\.rawValue).joined(separator: ", ")
        let base = error(status: 405, message: "Method not allowed")
        return HTTPResponse(
            statusCode: base.statusCode,
            statusText: base.statusText,
            headers: base.headers + [("Allow", allow)],
            body: base.body
        )
    }

    // MARK: OpenAI-strict error envelope
    //
    // The legacy `error(status:message:)` factory above encodes `code` as an
    // integer HTTP status — a pre-existing wire format bug; OpenAI's real
    // API uses a string `code` (e.g. `"model_not_found"`) and an optional
    // `param`. Fixing the legacy shape in place would change every existing
    // 4xx/5xx body and break the regression test at CompletionHandlerTests
    // that asserts `error.code as? Int == 404`. Instead we keep the legacy
    // factory untouched and provide this strict-shape factory for new code
    // paths that need OpenAI compatibility (currently: `model_not_found`).

    /// Build an OpenAI-compatible error response with a string `code` and
    /// optional `param`. Prefer this factory for new error responses.
    static func openAIError(
        status: Int,
        type: String,
        code: String,
        message: String,
        param: String? = nil
    ) -> HTTPResponse {
        let body = OpenAIErrorStrict(
            error: .init(message: message, type: type, param: param, code: code)
        )
        return json(body, status: status)
    }

    /// Reason for a `model_not_found` response, controlling the human-readable
    /// message while keeping the same HTTP status + code string.
    enum ModelNotFoundReason: Sendable {
        case unknownID
        case notDownloaded

        fileprivate func message(for id: String) -> String {
            switch self {
            case .unknownID:
                return "The model `\(id)` does not exist or you do not have access to it."
            case .notDownloaded:
                return "The model `\(id)` is not downloaded. "
                    + "Download it from Settings → Models before use."
            }
        }
    }

    /// HTTP 404 `model_not_found`, matching OpenAI's response shape. Used when
    /// the client's `model` field refers to something the server cannot route
    /// to — either absent from the catalog or present but not on disk.
    static func modelNotFound(
        modelID: String,
        reason: ModelNotFoundReason
    ) -> HTTPResponse {
        openAIError(
            status: 404,
            type: "invalid_request_error",
            code: "model_not_found",
            message: reason.message(for: modelID),
            param: "model"
        )
    }

    fileprivate static func statusText(for code: Int) -> String {
        switch code {
        case 200: "OK"
        case 400: "Bad Request"
        case 404: "Not Found"
        case 405: "Method Not Allowed"
        case 500: "Internal Server Error"
        case 503: "Service Unavailable"
        default: "Unknown"
        }
    }

    private static func errorType(for code: Int) -> String {
        switch code {
        case 400: "invalid_request_error"
        case 404: "not_found_error"
        case 405: "method_not_allowed"
        case 503: "service_unavailable"
        default: "internal_error"
        }
    }
}

/// OpenAI-compatible error envelope (legacy, `code: Int`).
///
/// The numeric `code` is a pre-existing wire bug — OpenAI's real API uses a
/// string code. Leaving this type in place to preserve backwards compatibility
/// for existing 4xx/5xx callers. New paths should prefer ``OpenAIErrorStrict``
/// via ``HTTPResponse/openAIError(status:type:code:message:param:)``.
private struct OpenAIError: Encodable {
    let error: Detail
    struct Detail: Encodable {
        let message: String
        let type: String
        let code: Int
    }
}

/// Strict OpenAI-compatible error envelope: string `code`, optional `param`.
/// Matches what `platform.openai.com/v1/chat/completions` returns for
/// `model_not_found` and similar validation failures.
///
/// Note the explicit `encode(to:)`: by default `JSONEncoder` omits keys for
/// nil optionals, but OpenAI's real API always writes `"param":null` and
/// `"code":null` when those values don't apply. Some SDK clients depend on
/// the keys being present; match the wire format exactly.
private struct OpenAIErrorStrict: Encodable {
    let error: Detail

    struct Detail: Encodable {
        let message: String
        let type: String
        let param: String?
        let code: String?

        enum CodingKeys: String, CodingKey {
            case message, type, param, code
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(message, forKey: .message)
            try container.encode(type, forKey: .type)
            // Explicit nil-writing: keeps the keys in the output body even
            // when the values aren't applicable, matching OpenAI exactly.
            if let param {
                try container.encode(param, forKey: .param)
            } else {
                try container.encodeNil(forKey: .param)
            }
            if let code {
                try container.encode(code, forKey: .code)
            } else {
                try container.encodeNil(forKey: .code)
            }
        }
    }
}

// MARK: - HTTP Response Writer

/// Wraps an `NWConnection` to provide async single-shot and streaming response writing.
final class HTTPResponseWriter: @unchecked Sendable {
    private let connection: NWConnection
    private let lifecycle: HTTPConnectionLifecycle
    private var responseSent = false
    private var streaming = false

    nonisolated init(connection: NWConnection, lifecycle: HTTPConnectionLifecycle) {
        self.connection = connection
        self.lifecycle = lifecycle
    }

    /// Send a complete single-shot response. No-op if a response was already
    /// sent or a stream is open. If streaming, terminates the stream instead.
    func send(_ response: HTTPResponse) async throws {
        if streaming {
            try? await finish()
            return
        }
        guard !responseSent else { return }
        responseSent = true
        try await writeAll(response.serialized())
    }

    /// Begin a chunked streaming response (sends status line + headers).
    func beginStreaming(
        statusCode: Int = 200,
        headers: [(name: String, value: String)] = []
    ) async throws {
        guard !responseSent, !streaming else { return }
        responseSent = true
        streaming = true

        var head = "HTTP/1.1 \(statusCode) \(HTTPResponse.statusText(for: statusCode))\r\n"
        for (name, value) in headers {
            head += "\(name): \(value)\r\n"
        }
        head += "Transfer-Encoding: chunked\r\n"
        if !headers.contains(where: { $0.name.lowercased() == "connection" }) {
            head += "Connection: close\r\n"
        }
        head += "\r\n"
        try await writeAll(Data(head.utf8))
    }

    /// Write a body chunk using HTTP chunked transfer encoding.
    func writeChunk(_ data: Data) async throws {
        let hex = String(data.count, radix: 16)
        var chunk = Data("\(hex)\r\n".utf8)
        chunk.append(data)
        chunk.append(Data("\r\n".utf8))
        try await writeAll(chunk)
    }

    /// Send the terminal chunk to end a chunked response. No-op if not streaming.
    func finish() async throws {
        guard streaming else { return }
        streaming = false
        try await writeAll(Data("0\r\n\r\n".utf8))
    }

    func isDisconnected() async -> Bool {
        await lifecycle.isDisconnected()
    }

    func waitForDisconnect() async {
        await lifecycle.waitForDisconnect()
    }

    private func writeAll(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error {
                    Task { await self.lifecycle.markDisconnected() }
                    cont.resume(throwing: error)
                } else {
                    cont.resume()
                }
            })
        }
    }
}

actor HTTPConnectionLifecycle {
    private var disconnected = false
    private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]

    func markDisconnected() {
        guard !disconnected else { return }
        disconnected = true
        let currentWaiters = waiters.values
        waiters.removeAll()
        for waiter in currentWaiters {
            waiter.resume()
        }
    }

    func isDisconnected() -> Bool {
        disconnected
    }

    func waitForDisconnect() async {
        if disconnected { return }
        let waiterID = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if disconnected || Task.isCancelled {
                    continuation.resume()
                } else {
                    waiters[waiterID] = continuation
                }
            }
        } onCancel: {
            Task { await self.resumeCancelledWaiter(waiterID) }
        }
    }

    private func resumeCancelledWaiter(_ waiterID: UUID) {
        guard let waiter = waiters.removeValue(forKey: waiterID) else { return }
        waiter.resume()
    }
}

// MARK: - SSE Writer

/// Formats and sends Server-Sent Events over an `HTTPResponseWriter`.
///
/// Usage:
/// ```
/// let sse = SSEWriter(writer)
/// try await sse.open()
/// await sse.send(someEncodable)
/// await sse.keepalive("prefill 1024/4096")
/// await sse.done()
/// ```
actor SSEWriter {
    private let writer: HTTPResponseWriter
    private let clock = ContinuousClock()
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = .sortedKeys
        return e
    }()
    private var closed = false
    private var lastWriteAt: ContinuousClock.Instant

    init(_ writer: HTTPResponseWriter) {
        self.writer = writer
        self.lastWriteAt = clock.now
    }

    /// Send SSE headers and begin the chunked stream.
    func open() async throws {
        try await writer.beginStreaming(headers: [
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive"),
        ])
        lastWriteAt = clock.now
    }

    /// Send a JSON-encoded SSE data line: `data: {json}\n\n`
    @discardableResult
    func send(_ value: some Encodable) async -> Bool {
        var line = Data("data: ".utf8)
        do { line.append(try encoder.encode(value)) }
        catch { return false }
        line.append(Data("\n\n".utf8))
        return await write(line)
    }

    /// Send a raw SSE data line: `data: {string}\n\n`
    @discardableResult
    func sendRaw(_ string: String) async -> Bool {
        await write(Data("data: \(string)\n\n".utf8))
    }

    /// Send an SSE comment (ignored by clients): `: {text}\n\n`
    @discardableResult
    func keepalive(_ text: String = "keepalive") async -> Bool {
        await write(Data(": \(text)\n\n".utf8))
    }

    /// Send the `data: [DONE]` sentinel and close the stream.
    @discardableResult
    func done() async -> Bool {
        guard !closed else { return false }
        closed = true
        do {
            try await writer.writeChunk(Data("data: [DONE]\n\n".utf8))
            try await writer.finish()
            return true
        } catch {
            return false
        }
    }

    /// Whether a write has failed (client disconnected).
    var isDisconnected: Bool { closed }

    func idleFor(atLeast duration: Duration) -> Bool {
        (clock.now - lastWriteAt) >= duration
    }

    // MARK: - Private

    private func write(_ data: Data) async -> Bool {
        guard !closed else { return false }
        do {
            try await writer.writeChunk(data)
            lastWriteAt = clock.now
            return true
        } catch {
            closed = true
            return false
        }
    }
}

// MARK: - HTTP Request Parser

private nonisolated enum HTTPRequestParser {

    struct HeaderResult {
        let method: HTTPMethod
        let path: String
        let headers: [(name: String, value: String)]
        let contentLength: Int
        let bodyOffset: Int
    }

    /// Parse the request line and headers. Returns nil if the header terminator hasn't arrived yet.
    static func parseHeaders(_ data: Data) throws -> HeaderResult? {
        let separator = Data("\r\n\r\n".utf8)
        guard let separatorRange = data.range(of: separator) else {
            return nil
        }

        let headerData = data[data.startIndex..<separatorRange.lowerBound]
        guard let headerString = String(data: headerData, encoding: .utf8) else {
            throw HTTPParseError.malformedRequestLine
        }

        var lines = headerString.split(separator: "\r\n", omittingEmptySubsequences: false)
        guard !lines.isEmpty else { throw HTTPParseError.malformedRequestLine }

        let requestLine = lines.removeFirst()
        let parts = requestLine.split(separator: " ", maxSplits: 2)
        guard parts.count >= 2 else { throw HTTPParseError.malformedRequestLine }

        let methodString = String(parts[0])
        guard let method = HTTPMethod(rawValue: methodString) else {
            throw HTTPParseError.invalidMethod(methodString)
        }

        let path = parsePath(String(parts[1]))

        var headers: [(name: String, value: String)] = []
        for line in lines {
            guard let colonIndex = line.firstIndex(of: ":") else { continue }
            let name = String(line[line.startIndex..<colonIndex]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: colonIndex)...]).trimmingCharacters(in: .whitespaces)
            headers.append((name, value))
        }

        let contentLengthStr = headers.first(where: { $0.name.lowercased() == "content-length" })?.value
        let contentLength = contentLengthStr.flatMap(Int.init) ?? 0
        let bodyOffset = data.distance(from: data.startIndex, to: separatorRange.upperBound)

        return HeaderResult(
            method: method, path: path, headers: headers,
            contentLength: contentLength, bodyOffset: bodyOffset
        )
    }

    private static func parsePath(_ raw: String) -> String {
        if let q = raw.firstIndex(of: "?") {
            return String(raw[raw.startIndex..<q])
        }
        return raw
    }
}

private enum HTTPParseError: LocalizedError, Sendable {
    case connectionClosed
    case requestTooLarge
    case malformedRequestLine
    case invalidMethod(String)

    var errorDescription: String? {
        switch self {
        case .connectionClosed: "Connection closed before request completed"
        case .requestTooLarge: "Request exceeds maximum allowed size"
        case .malformedRequestLine: "Malformed HTTP request line"
        case .invalidMethod(let m): "Unsupported HTTP method: \(m)"
        }
    }
}

// MARK: - HTTP Server

/// Lightweight HTTP/1.1 server built on Network.framework.
///
/// Binds to `127.0.0.1` only. Supports route registration, single-shot and
/// streaming (chunked) responses. No external dependencies.
@Observable @MainActor
final class HTTPServer {

    // MARK: - Observable State

    private(set) var isRunning = false
    private(set) var activeConnections = 0
    private(set) var totalRequestsServed = 0
    /// Non-nil when the most recent enable attempt failed to bind or the listener
    /// transitioned to `.failed`. Cleared on successful start or user-initiated
    /// stop. Used by the Dashboard to distinguish "starting" from "failed".
    private(set) var lastStartError: String?

    // MARK: - Configuration

    @ObservationIgnored private var port: UInt16

    // MARK: - Private

    @ObservationIgnored private var listener: NWListener?
    @ObservationIgnored private var routes: [Route] = []
    @ObservationIgnored private var trackedConnections: [UUID: TrackedConnection] = [:]
    @ObservationIgnored private var isStopping = false

    fileprivate struct Route: Sendable {
        let method: HTTPMethod
        let path: String
        let handler: @Sendable (HTTPRequest, HTTPResponseWriter) async throws -> Void
    }

    private struct TrackedConnection: Sendable {
        let task: Task<Void, Never>
        let cancelTransport: @Sendable () -> Void
    }

    // MARK: - Init

    init(port: UInt16 = 8321) {
        self.port = port
    }

    @discardableResult
    func registerConnectionTaskForTesting(
        _ task: Task<Void, Never>,
        cancelTransport: @escaping @Sendable () -> Void = {}
    ) -> Bool {
        trackConnection(task: task, cancelTransport: cancelTransport)
    }

    // MARK: - Route Registration

    func route(
        _ method: HTTPMethod,
        _ path: String,
        handler: @escaping @Sendable (HTTPRequest, HTTPResponseWriter) async throws -> Void
    ) {
        routes.append(Route(method: method, path: path, handler: handler))
    }

    // MARK: - Lifecycle

    func start() async {
        guard !isRunning else { return }
        isStopping = false

        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            Log.server.error("Invalid port: \(self.port)")
            lastStartError = "Invalid port \(port)"
            return
        }

        let params = NWParameters.tcp
        params.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: .ipv4(.loopback),
            port: nwPort
        )

        do {
            let newListener = try NWListener(using: params)
            self.listener = newListener

            newListener.stateUpdateHandler = { [weak self] state in
                Task { @MainActor [weak self] in
                    self?.handleListenerState(state)
                }
            }

            newListener.newConnectionHandler = { [weak self] connection in
                Task { @MainActor [weak self] in
                    self?.handleNewConnection(connection)
                }
            }

            newListener.start(queue: .global(qos: .userInitiated))
            isRunning = true
            lastStartError = nil
            Log.server.info("Server starting on 127.0.0.1:\(self.port)")
        } catch {
            Log.server.error("Failed to create listener: \(error)")
            lastStartError = "Bind failed: \(error.localizedDescription)"
        }
    }

    func stop() {
        isStopping = true
        let connections = Array(trackedConnections.values)
        for connection in connections {
            connection.cancelTransport()
            connection.task.cancel()
        }
        trackedConnections.removeAll()
        listener?.cancel()
        listener = nil
        isRunning = false
        lastStartError = nil
        Log.server.info("Server stopped")
    }

    func stopAndDrain() async {
        isStopping = true
        listener?.cancel()
        listener = nil
        isRunning = false
        lastStartError = nil
        let connections = Array(trackedConnections.values)
        Log.server.info("Server stopping — draining \(connections.count) connection task(s)")

        for connection in connections {
            connection.cancelTransport()
            connection.task.cancel()
        }

        for connection in connections {
            _ = await connection.task.result
        }

        trackedConnections.removeAll()
        Log.server.info("Server stopped")
    }

    func updatePort(_ newPort: UInt16) async {
        guard newPort != port else { return }
        let wasRunning = isRunning
        if wasRunning { stop() }
        port = newPort
        if wasRunning { await start() }
    }

    // MARK: - Private — Listener

    private func handleListenerState(_ state: NWListener.State) {
        switch state {
        case .ready:
            Log.server.info("Server ready on 127.0.0.1:\(self.port)")
            lastStartError = nil
        case .failed(let error):
            Log.server.error("Server listener failed: \(error)")
            isRunning = false
            lastStartError = "Listener failed: \(error.localizedDescription)"
        case .cancelled:
            isRunning = false
        default:
            break
        }
    }

    // MARK: - Private — Connection

    private func handleNewConnection(_ connection: NWConnection) {
        guard !isStopping else {
            connection.cancel()
            return
        }
        activeConnections += 1

        let routes = self.routes
        let task = Task.detached { [weak self] in
            let lifecycle = HTTPConnectionLifecycle()
            defer {
                connection.cancel()
                Task { @MainActor [weak self] in
                    self?.activeConnections -= 1
                }
            }

            connection.stateUpdateHandler = { state in
                switch state {
                case .failed, .cancelled:
                    Task { await lifecycle.markDisconnected() }
                default:
                    break
                }
            }
            connection.start(queue: .global(qos: .userInitiated))

            do {
                let request = try await HTTPServer.readRequest(from: connection)
                let writer = HTTPResponseWriter(connection: connection, lifecycle: lifecycle)
                let disconnectMonitor = Task {
                    await HTTPServer.monitorPeerDisconnect(
                        on: connection,
                        lifecycle: lifecycle
                    )
                }
                defer { disconnectMonitor.cancel() }
                await dispatchRoute(request, writer: writer, routes: routes)
                Task { @MainActor [weak self] in self?.totalRequestsServed += 1 }
            } catch is CancellationError {
                // Server shutting down
            } catch {
                Log.server.error("Connection error: \(error)")
                let writer = HTTPResponseWriter(connection: connection, lifecycle: lifecycle)
                try? await writer.send(.badRequest(error.localizedDescription))
            }
        }

        _ = trackConnection(task: task, cancelTransport: { connection.cancel() })
    }

    @discardableResult
    private func trackConnection(
        task: Task<Void, Never>,
        cancelTransport: @escaping @Sendable () -> Void
    ) -> Bool {
        guard !isStopping else {
            cancelTransport()
            task.cancel()
            return false
        }

        let id = UUID()
        trackedConnections[id] = TrackedConnection(
            task: task,
            cancelTransport: cancelTransport
        )

        Task { @MainActor [weak self] in
            _ = await task.result
            self?.trackedConnections.removeValue(forKey: id)
        }

        return true
    }

    nonisolated private static func monitorPeerDisconnect(
        on connection: NWConnection,
        lifecycle: HTTPConnectionLifecycle
    ) async {
        while !Task.isCancelled {
            let state = await withCheckedContinuation { continuation in
                connection.receive(minimumIncompleteLength: 0, maximumLength: 1) {
                    content, _, isComplete, error in
                    continuation.resume(returning: (
                        contentBytes: content?.count ?? 0,
                        isComplete: isComplete,
                        hasError: error != nil
                    ))
                }
            }

            if state.isComplete || state.hasError {
                await lifecycle.markDisconnected()
                return
            }

            if state.contentBytes == 0 {
                do {
                    try await Task.sleep(nanoseconds: 50_000_000)
                } catch {
                    return
                }
            }
        }
    }

    // MARK: - Private — Read Request

    nonisolated private static func readRequest(from connection: NWConnection) async throws -> HTTPRequest {
        var buffer = Data()
        let maxSize = 50 * 1024 * 1024 // 50MB (base64 images)

        // Phase 1: Accumulate until headers are complete
        var headerResult: HTTPRequestParser.HeaderResult?
        while headerResult == nil {
            try Task.checkCancellation()
            buffer.append(try await receiveChunk(from: connection))
            guard buffer.count <= maxSize else { throw HTTPParseError.requestTooLarge }
            headerResult = try HTTPRequestParser.parseHeaders(buffer)
        }

        let parsed = headerResult!

        // Phase 2: If Content-Length, accumulate body bytes (no re-parsing)
        if parsed.contentLength > 0 {
            while buffer.count < parsed.bodyOffset + parsed.contentLength {
                try Task.checkCancellation()
                buffer.append(try await receiveChunk(from: connection))
                guard buffer.count <= maxSize else { throw HTTPParseError.requestTooLarge }
            }
            let bodyEnd = buffer.index(buffer.startIndex, offsetBy: parsed.bodyOffset + parsed.contentLength)
            let body = Data(buffer[buffer.index(buffer.startIndex, offsetBy: parsed.bodyOffset)..<bodyEnd])
            return HTTPRequest(method: parsed.method, path: parsed.path, headers: parsed.headers, body: body)
        }

        return HTTPRequest(method: parsed.method, path: parsed.path, headers: parsed.headers, body: nil)
    }

    nonisolated private static func receiveChunk(from connection: NWConnection) async throws -> Data {
        try await withCheckedThrowingContinuation { cont in
            connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) {
                content, _, _, error in
                if let error {
                    cont.resume(throwing: error)
                } else if let content, !content.isEmpty {
                    cont.resume(returning: content)
                } else {
                    cont.resume(throwing: HTTPParseError.connectionClosed)
                }
            }
        }
    }

}

// MARK: - Route Dispatch (nonisolated — runs on caller's context, not MainActor)

private func dispatchRoute(
    _ request: HTTPRequest,
    writer: HTTPResponseWriter,
    routes: [HTTPServer.Route]
) async {
    if request.path == "/v1/chat/completions" {
        Log.server.info(
            "HTTP request — method=\(request.method.rawValue) path=\(request.path) bodyBytes=\(request.body?.count ?? 0)"
        )
    } else {
        Log.server.debug("\(request.method.rawValue) \(request.path)")
    }

    let pathMatches = routes.filter { $0.path == request.path }

    if pathMatches.isEmpty {
        try? await writer.send(.notFound())
        return
    }

    if let route = pathMatches.first(where: { $0.method == request.method }) {
        do {
            try await route.handler(request, writer)
        } catch {
            Log.server.error("Handler error \(request.method.rawValue) \(request.path): \(error)")
            try? await writer.send(.internalError(error.localizedDescription))
        }
    } else {
        let allowed = pathMatches.map(\.method)
        try? await writer.send(.methodNotAllowed(allowed: allowed))
    }
}
