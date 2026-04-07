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

/// OpenAI-compatible error envelope.
private struct OpenAIError: Encodable {
    let error: Detail
    struct Detail: Encodable {
        let message: String
        let type: String
        let code: Int
    }
}

// MARK: - HTTP Response Writer

/// Wraps an `NWConnection` to provide async single-shot and streaming response writing.
final class HTTPResponseWriter: @unchecked Sendable {
    private let connection: NWConnection
    private var responseSent = false
    private var streaming = false

    nonisolated init(connection: NWConnection) {
        self.connection = connection
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

    private func writeAll(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) }
                else { cont.resume() }
            })
        }
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
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = .sortedKeys
        return e
    }()
    private var closed = false

    init(_ writer: HTTPResponseWriter) {
        self.writer = writer
    }

    /// Send SSE headers and begin the chunked stream.
    func open() async throws {
        try await writer.beginStreaming(headers: [
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive"),
        ])
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
    func done() async {
        guard !closed else { return }
        closed = true
        try? await writer.writeChunk(Data("data: [DONE]\n\n".utf8))
        try? await writer.finish()
    }

    /// Whether a write has failed (client disconnected).
    var isDisconnected: Bool { closed }

    // MARK: - Private

    private func write(_ data: Data) async -> Bool {
        guard !closed else { return false }
        do {
            try await writer.writeChunk(data)
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

    // MARK: - Configuration

    @ObservationIgnored private var port: UInt16

    // MARK: - Private

    @ObservationIgnored private var listener: NWListener?
    @ObservationIgnored private var routes: [Route] = []
    @ObservationIgnored private var connectionTasks: Set<Task<Void, Never>> = []

    fileprivate struct Route: Sendable {
        let method: HTTPMethod
        let path: String
        let handler: @Sendable (HTTPRequest, HTTPResponseWriter) async throws -> Void
    }

    // MARK: - Init

    init(port: UInt16 = 8321) {
        self.port = port
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

        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            Log.server.error("Invalid port: \(self.port)")
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
            Log.server.info("Server starting on 127.0.0.1:\(self.port)")
        } catch {
            Log.server.error("Failed to create listener: \(error)")
        }
    }

    func stop() {
        for task in connectionTasks { task.cancel() }
        connectionTasks.removeAll()
        listener?.cancel()
        listener = nil
        isRunning = false
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
        case .failed(let error):
            Log.server.error("Server listener failed: \(error)")
            isRunning = false
        case .cancelled:
            isRunning = false
        default:
            break
        }
    }

    // MARK: - Private — Connection

    private func handleNewConnection(_ connection: NWConnection) {
        activeConnections += 1

        let routes = self.routes
        let task = Task.detached { [weak self] in
            defer {
                connection.cancel()
                Task { @MainActor [weak self] in
                    self?.activeConnections -= 1
                }
            }

            connection.start(queue: .global(qos: .userInitiated))

            do {
                let request = try await HTTPServer.readRequest(from: connection)
                let writer = HTTPResponseWriter(connection: connection)
                await dispatchRoute(request, writer: writer, routes: routes)
                Task { @MainActor [weak self] in self?.totalRequestsServed += 1 }
            } catch is CancellationError {
                // Server shutting down
            } catch {
                Log.server.error("Connection error: \(error)")
                let writer = HTTPResponseWriter(connection: connection)
                try? await writer.send(.badRequest(error.localizedDescription))
            }
        }

        connectionTasks.insert(task)
        Task {
            _ = await task.result
            connectionTasks.remove(task)
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
    Log.server.debug("\(request.method.rawValue) \(request.path)")

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
