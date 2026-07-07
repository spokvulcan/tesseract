import Foundation
import MLXLMCommon

// MARK: - MCPTransportResponse

/// The transport-level outcome of one POST: the JSON-RPC response value (nil for
/// an accepted notification, HTTP 202) and the session id the server echoed.
nonisolated struct MCPTransportResponse: Sendable {
    let json: JSONValue?
    let sessionID: String?

    /// Map a completed response into a transport response, or throw
    /// ``MCPClientError/http`` for a non-2xx status carrying no JSON-RPC error
    /// envelope — a transport-level failure (e.g. 503 from a disabled server).
    /// Shared by both transports so that rule lives in one place. `errorBody` is
    /// an autoclosure: the (possibly large) body string is built only on the
    /// throw path.
    static func mapping(
        status: Int, json: JSONValue?, sessionID: String?,
        errorBody: @autoclosure () -> String
    ) throws -> MCPTransportResponse {
        if status >= 400, json?.asObject?["error"] == nil {
            throw MCPClientError.http(status: status, body: errorBody())
        }
        return MCPTransportResponse(json: json, sessionID: sessionID)
    }
}

// MARK: - MCPTransport

/// The wire under an ``MCPClient``: POST one JSON-RPC message, return the
/// server's response, forwarding any interim `notifications/*` (progress,
/// `tools/list_changed`) that arrive — inline in a Streamable-HTTP SSE stream —
/// to `onNotification`.
///
/// Two implementations back the two consumers the PRD names: ``HTTPMCPTransport``
/// (URLSession → arbitrary user-configured servers) and
/// ``InProcessMCPTransport`` (→ the in-app Browser MCP server, no socket, so
/// browser-use in chat never depends on the inference HTTP listener running).
nonisolated protocol MCPTransport: Sendable {
    func post(
        _ body: Data,
        sessionID: String?,
        onNotification: @escaping @Sendable (JSONValue) -> Void
    ) async throws -> MCPTransportResponse
}

// MARK: - HTTPMCPTransport

/// Streamable-HTTP transport over `URLSession`. A single `/mcp` endpoint that
/// answers each POST with either an `application/json` body (our own server, and
/// simple servers) or a `text/event-stream` (servers that stream progress before
/// the final result). Custom headers (e.g. an API key, US #14) ride every
/// request.
nonisolated final class HTTPMCPTransport: MCPTransport {
    private let endpoint: URL?
    private let headers: [String: String]
    private let session: URLSession

    init(endpoint: URL?, headers: [String: String] = [:], session: URLSession = .shared) {
        self.endpoint = endpoint
        self.headers = headers
        self.session = session
    }

    func post(
        _ body: Data,
        sessionID: String?,
        onNotification: @escaping @Sendable (JSONValue) -> Void
    ) async throws -> MCPTransportResponse {
        guard let endpoint else {
            throw MCPClientError.transport("invalid server URL")
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        if let sessionID { request.setValue(sessionID, forHTTPHeaderField: "Mcp-Session-Id") }
        // Custom headers last so a server config can override defaults if needed.
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        request.httpBody = body

        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw MCPClientError.transport("no HTTP response")
        }
        let sessionID = http.value(forHTTPHeaderField: "Mcp-Session-Id")
        let contentType = (http.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()

        // 202 Accepted: notification acknowledged, no body to read.
        if http.statusCode == 202 {
            for try await _ in bytes {}  // drain
            return MCPTransportResponse(json: nil, sessionID: sessionID)
        }

        if contentType.contains("text/event-stream") {
            let value = try await Self.consumeEventStream(bytes, onNotification: onNotification)
            return MCPTransportResponse(json: value, sessionID: sessionID)
        }

        var data = Data()
        if http.expectedContentLength > 0 {
            data.reserveCapacity(Int(http.expectedContentLength))  // e.g. a base64 image
        }
        for try await byte in bytes { data.append(byte) }
        return try MCPTransportResponse.mapping(
            status: http.statusCode, json: MCPWire.decode(data), sessionID: sessionID,
            errorBody: String(data: data, encoding: .utf8) ?? "")
    }

    /// Parse an SSE stream: forward every server-initiated notification to
    /// `onNotification` and return the first JSON-RPC response (the reply to our
    /// request), after which the server closes the stream. Each MCP SSE event
    /// carries one single-line JSON message in its `data:` field, so every such
    /// line is decoded as a complete message — no dependence on how blank-line
    /// event boundaries survive line iteration.
    private static func consumeEventStream(
        _ bytes: URLSession.AsyncBytes,
        onNotification: @escaping @Sendable (JSONValue) -> Void
    ) async throws -> JSONValue? {
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }  // ignore event:/id:/comments
            var payload = Substring(line.dropFirst("data:".count))
            if payload.first == " " { payload = payload.dropFirst() }
            guard let value = MCPWire.decode(Data(String(payload).utf8)) else { continue }
            if MCPWire.isResponse(value) {
                return value  // reply seen — server closes the stream after this
            }
            onNotification(value)
        }
        return nil
    }
}

// MARK: - InProcessMCPTransport

/// Transport that speaks the full MCP wire to an in-app server *without* a
/// socket, by handing the request straight to a handler — the in-app Browser MCP
/// server's `handle(request:)`. This is the ADR-0027 dogfooded path: the agent
/// consumes the browser server through the real MCP client and the server's real
/// request handler, but browser-use in chat no longer depends on the inference
/// HTTP listener being enabled (it only starts with `isServerEnabled`).
nonisolated final class InProcessMCPTransport: MCPTransport {
    private let handle: @Sendable (HTTPRequest) async -> HTTPResponse

    init(handle: @escaping @Sendable (HTTPRequest) async -> HTTPResponse) {
        self.handle = handle
    }

    func post(
        _ body: Data,
        sessionID: String?,
        onNotification: @escaping @Sendable (JSONValue) -> Void
    ) async throws -> MCPTransportResponse {
        var headers: [(name: String, value: String)] = [("Content-Type", "application/json")]
        if let sessionID { headers.append(("Mcp-Session-Id", sessionID)) }
        let request = HTTPRequest(method: .POST, path: "/mcp", headers: headers, body: body)
        let response = await handle(request)
        let sessionID = response.headers.first { $0.name.lowercased() == "mcp-session-id" }?.value

        if response.statusCode == 202 {
            return MCPTransportResponse(json: nil, sessionID: sessionID)
        }
        return try MCPTransportResponse.mapping(
            status: response.statusCode, json: response.body.flatMap { MCPWire.decode($0) },
            sessionID: sessionID,
            errorBody: response.body.flatMap { String(data: $0, encoding: .utf8) } ?? "")
    }
}
