import Foundation
import MLXLMCommon

// MARK: - MCPBrowserServer

/// The **Browser MCP Server**: the MCP (Model Context Protocol) endpoint the
/// running app serves over its existing loopback `HTTPServer`, so external
/// agents — and, later, Tesseract's own agent (PRD #190) — can drive the
/// **Agent Browser**. Implements the minimal, spec-compliant *Streamable HTTP*
/// transport: JSON request in, JSON response out, with a per-connection session
/// id mapping each client to its private **Browser Session**.
///
/// v1 answers each request with a single JSON body (no server-initiated SSE
/// stream); that is a valid Streamable HTTP server and is what HTTP MCP clients
/// like Claude Code connect to via `claude mcp add --transport http`.
@MainActor
final class MCPBrowserServer {

    private let browser: AgentBrowser
    private let executor: BrowserToolExecutor
    private let isEnabled: @MainActor @Sendable () -> Bool
    private let telemetry: BrowserMCPTelemetryRecorder

    init(
        browser: AgentBrowser,
        executor: BrowserToolExecutor,
        isEnabled: @escaping @MainActor @Sendable () -> Bool,
        telemetry: BrowserMCPTelemetryRecorder = BrowserMCPTelemetryRecorder()
    ) {
        self.browser = browser
        self.executor = executor
        self.isEnabled = isEnabled
        self.telemetry = telemetry
    }

    /// Register the `/mcp` routes on the shared HTTP server. Requests go through
    /// ``handleOverHTTP(request:)``, which applies the loopback-only gates; the
    /// in-process transport instead calls ``handle(request:)`` directly.
    func attach(to httpServer: HTTPServer, path: String = "/mcp") {
        for method in [HTTPMethod.POST, .DELETE] {
            httpServer.route(method, path) { [weak self] request, writer in
                guard let self else {
                    try await writer.send(.serviceUnavailable("Browser MCP server unavailable"))
                    return
                }
                let response = await self.handleOverHTTP(request: request)
                try await writer.send(response)
            }
        }
    }

    /// The loopback-listener entry point: applies the **HTTP-exposure** gate
    /// (`isEnabled`) and the DNS-rebinding origin guard, then dispatches to
    /// ``handle(request:)``. These gates live here, *not* in `handle`, so the
    /// in-process transport (the in-app agent) reaches `handle` directly with no
    /// port open and regardless of this switch — the two enablement switches are
    /// thereby separable (ADR-0028).
    func handleOverHTTP(request: HTTPRequest) async -> HTTPResponse {
        guard isEnabled() else {
            return plain(503, "Browser MCP server is disabled in Tesseract settings.")
        }
        if let rejection = originRejection(request) {
            return rejection
        }
        return await handle(request: request, origin: .http)
    }

    /// Close every client session (server stop / app termination).
    func closeAllSessions() {
        telemetry.recordServerShutdown()
        browser.closeAll()
    }

    // MARK: - Request handling

    /// Handle one MCP request. Pure protocol dispatch — the HTTP-exposure gate and
    /// the origin guard are applied by ``attach(to:path:)`` on the loopback
    /// listener, *not* here, so the in-process transport (the in-app agent) reaches
    /// this directly (ADR-0027/0028). `origin` names the entry path for telemetry:
    /// `.http` from the loopback listener, `.inProcess` from the in-app transport.
    func handle(request: HTTPRequest, origin: MCPClientOrigin) async -> HTTPResponse {
        switch request.method {
        case .DELETE:
            if let sessionID = request.header("Mcp-Session-Id") {
                telemetry.recordSessionEnd(sessionID: sessionID, origin: origin)
                browser.closeSession(id: sessionID)
            }
            return empty(200)
        case .POST:
            return await handlePost(request: request, origin: origin)
        default:
            return plain(405, "Method not allowed")
        }
    }

    private func handlePost(request: HTTPRequest, origin: MCPClientOrigin) async -> HTTPResponse {
        guard let body = request.body, !body.isEmpty,
            let message = try? JSONDecoder().decode(JSONValue.self, from: body)
        else {
            telemetry.recordProtocolError(
                method: nil, code: MCPProtocol.ErrorCode.parseError,
                message: "Invalid or empty JSON body",
                sessionID: request.header("Mcp-Session-Id"), origin: origin)
            return json(
                MCPEncoding.error(
                    id: .null, code: MCPProtocol.ErrorCode.parseError,
                    message: "Invalid or empty JSON body"),
                status: 400)
        }

        // Batch: process each, return the responses that have an id.
        if case .array(let items) = message {
            var responses: [JSONValue] = []
            for item in items {
                if let response = await dispatch(item, request: request, origin: origin) {
                    responses.append(response)
                }
            }
            return responses.isEmpty ? empty(202) : json(.array(responses))
        }

        // `initialize` is special: it mints the session id returned in the header.
        if let object = message.asObject, object["method"]?.asString == "initialize" {
            return initialize(object, origin: origin)
        }

        if let response = await dispatch(message, request: request, origin: origin) {
            return json(response)
        }
        // Notification (no id) — acknowledge with 202 and no body.
        return empty(202)
    }

    private func initialize(_ object: [String: JSONValue], origin: MCPClientOrigin)
        -> HTTPResponse
    {
        let id = object["id"] ?? .null
        let requested = object["params"]?.asObject?["protocolVersion"]?.asString
        let version = requested.flatMap { $0.isEmpty ? nil : $0 } ?? MCPProtocol.version

        let sessionID = UUID().uuidString
        _ = browser.session(id: sessionID)  // create the Browser Session now
        telemetry.recordSessionStart(
            sessionID: sessionID, origin: origin, params: object["params"]?.asObject)

        let result = MCPEncoding.result(
            id: id, MCPEncoding.initializeResult(protocolVersion: version))
        return json(result, sessionID: sessionID)
    }

    /// Handle one JSON-RPC message. Returns the response value, or nil for
    /// notifications (which get a bodyless 202 at the transport layer).
    private func dispatch(
        _ message: JSONValue, request: HTTPRequest, origin: MCPClientOrigin
    ) async -> JSONValue? {
        guard let object = message.asObject, let method = object["method"]?.asString else {
            telemetry.recordProtocolError(
                method: nil, code: MCPProtocol.ErrorCode.invalidRequest,
                message: "Not a JSON-RPC request",
                sessionID: request.header("Mcp-Session-Id"), origin: origin)
            return MCPEncoding.error(
                id: message.asObject?["id"] ?? .null,
                code: MCPProtocol.ErrorCode.invalidRequest,
                message: "Not a JSON-RPC request")
        }

        // Notifications carry no id and expect no response.
        guard let id = object["id"] else { return nil }

        switch method {
        case "ping":
            return MCPEncoding.result(id: id, .object([:]))
        case "tools/list":
            telemetry.recordToolsList(
                sessionID: request.header("Mcp-Session-Id"), origin: origin)
            return MCPEncoding.result(id: id, MCPEncoding.toolsListResult(BrowserToolCatalog.all))
        case "tools/call":
            return await handleToolCall(id: id, object: object, request: request, origin: origin)
        default:
            telemetry.recordProtocolError(
                method: method, code: MCPProtocol.ErrorCode.methodNotFound,
                message: "Unknown method: \(method)",
                sessionID: request.header("Mcp-Session-Id"), origin: origin)
            return MCPEncoding.error(
                id: id, code: MCPProtocol.ErrorCode.methodNotFound,
                message: "Unknown method: \(method)")
        }
    }

    private func handleToolCall(
        id: JSONValue, object: [String: JSONValue], request: HTTPRequest,
        origin: MCPClientOrigin
    ) async -> JSONValue {
        guard let sessionID = request.header("Mcp-Session-Id"),
            let session = browser.existingSession(id: sessionID)
        else {
            telemetry.recordProtocolError(
                method: "tools/call", code: MCPProtocol.ErrorCode.noSession,
                message: "No active session",
                sessionID: request.header("Mcp-Session-Id"), origin: origin)
            return MCPEncoding.error(
                id: id, code: MCPProtocol.ErrorCode.noSession,
                message: "No active session — call initialize first (Mcp-Session-Id required).")
        }
        guard let params = object["params"]?.asObject, let name = params["name"]?.asString else {
            telemetry.recordProtocolError(
                method: "tools/call", code: MCPProtocol.ErrorCode.invalidParams,
                message: "tools/call requires params.name",
                sessionID: sessionID, origin: origin)
            return MCPEncoding.error(
                id: id, code: MCPProtocol.ErrorCode.invalidParams,
                message: "tools/call requires params.name")
        }
        let arguments = params["arguments"]?.asObject ?? [:]
        let normalized = ToolArgumentNormalizer.normalize(arguments)
        let start = ContinuousClock.now
        let result = await executor.call(name, session: session, arguments: normalized)
        telemetry.recordToolCall(
            sessionID: sessionID, origin: origin, tool: name, arguments: normalized,
            result: result, duration: ContinuousClock.now - start)
        return MCPEncoding.result(id: id, MCPEncoding.toolCallResult(result))
    }

    // MARK: - Origin guard (DNS-rebinding protection)

    /// Reject requests whose `Origin` is a real remote site — the DNS-rebinding
    /// vector where a malicious page tries to reach the loopback server. Native
    /// clients (Claude Code) send no `Origin` and pass; a browser attack always
    /// carries its site's origin, which is not loopback.
    private func originRejection(_ request: HTTPRequest) -> HTTPResponse? {
        guard let origin = request.header("Origin") else { return nil }
        // A loopback Origin is the browser dev-server case (localhost:5173, …) —
        // allow it. Everything else is rejected fail-closed: a real remote site
        // (classic DNS-rebinding) *and* an opaque `null` origin (sandboxed iframe
        // / file://), which the rebinding vector can elicit and from which no
        // legitimate client browses.
        if let url = URL(string: origin), let host = url.host?.lowercased(),
            ["localhost", "127.0.0.1", "::1", "[::1]"].contains(host)
        {
            return nil
        }
        return plain(403, "Origin not allowed")
    }

    // MARK: - Response builders

    private func json(_ value: JSONValue, status: Int = 200, sessionID: String? = nil)
        -> HTTPResponse
    {
        let body = (try? JSONEncoder().encode(value)) ?? Data("{}".utf8)
        var headers: [(name: String, value: String)] = [("Content-Type", "application/json")]
        if let sessionID { headers.append(("Mcp-Session-Id", sessionID)) }
        return HTTPResponse(
            statusCode: status, statusText: HTTPResponse.statusText(for: status), headers: headers,
            body: body)
    }

    private func empty(_ status: Int) -> HTTPResponse {
        HTTPResponse(
            statusCode: status, statusText: HTTPResponse.statusText(for: status), headers: [],
            body: nil)
    }

    private func plain(_ status: Int, _ message: String) -> HTTPResponse {
        HTTPResponse(
            statusCode: status, statusText: HTTPResponse.statusText(for: status),
            headers: [("Content-Type", "text/plain; charset=utf-8")],
            body: Data(message.utf8))
    }
}
