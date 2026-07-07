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

    init(
        browser: AgentBrowser,
        executor: BrowserToolExecutor,
        isEnabled: @escaping @MainActor @Sendable () -> Bool
    ) {
        self.browser = browser
        self.executor = executor
        self.isEnabled = isEnabled
    }

    /// Register the `/mcp` routes on the shared HTTP server.
    func attach(to httpServer: HTTPServer, path: String = "/mcp") {
        for method in [HTTPMethod.POST, .DELETE] {
            httpServer.route(method, path) { [weak self] request, writer in
                guard let self else {
                    try await writer.send(.serviceUnavailable("Browser MCP server unavailable"))
                    return
                }
                let response = await self.handle(request: request)
                try await writer.send(response)
            }
        }
    }

    /// Close every client session (server stop / app termination).
    func closeAllSessions() {
        browser.closeAll()
    }

    // MARK: - Request handling

    func handle(request: HTTPRequest) async -> HTTPResponse {
        guard isEnabled() else {
            return plain(503, "Browser MCP server is disabled in Tesseract settings.")
        }
        if let rejection = originRejection(request) {
            return rejection
        }

        switch request.method {
        case .DELETE:
            if let sessionID = request.header("Mcp-Session-Id") {
                browser.closeSession(id: sessionID)
            }
            return empty(200)
        case .POST:
            return await handlePost(request: request)
        default:
            return plain(405, "Method not allowed")
        }
    }

    private func handlePost(request: HTTPRequest) async -> HTTPResponse {
        guard let body = request.body, !body.isEmpty,
            let message = try? JSONDecoder().decode(JSONValue.self, from: body)
        else {
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
                if let response = await dispatch(item, request: request) {
                    responses.append(response)
                }
            }
            return responses.isEmpty ? empty(202) : json(.array(responses))
        }

        // `initialize` is special: it mints the session id returned in the header.
        if let object = message.asObject, object["method"]?.asString == "initialize" {
            return initialize(object)
        }

        if let response = await dispatch(message, request: request) {
            return json(response)
        }
        // Notification (no id) — acknowledge with 202 and no body.
        return empty(202)
    }

    private func initialize(_ object: [String: JSONValue]) -> HTTPResponse {
        let id = object["id"] ?? .null
        let requested = object["params"]?.asObject?["protocolVersion"]?.asString
        let version = (requested?.isEmpty == false) ? requested! : MCPProtocol.version

        let sessionID = UUID().uuidString
        _ = browser.session(id: sessionID)  // create the Browser Session now

        let result = MCPEncoding.result(
            id: id, MCPEncoding.initializeResult(protocolVersion: version))
        return json(result, sessionID: sessionID)
    }

    /// Handle one JSON-RPC message. Returns the response value, or nil for
    /// notifications (which get a bodyless 202 at the transport layer).
    private func dispatch(_ message: JSONValue, request: HTTPRequest) async -> JSONValue? {
        guard let object = message.asObject, let method = object["method"]?.asString else {
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
            return MCPEncoding.result(id: id, MCPEncoding.toolsListResult(BrowserToolCatalog.all))
        case "tools/call":
            return await handleToolCall(id: id, object: object, request: request)
        default:
            return MCPEncoding.error(
                id: id, code: MCPProtocol.ErrorCode.methodNotFound,
                message: "Unknown method: \(method)")
        }
    }

    private func handleToolCall(
        id: JSONValue, object: [String: JSONValue], request: HTTPRequest
    ) async -> JSONValue {
        guard let sessionID = request.header("Mcp-Session-Id"),
            let session = browser.existingSession(id: sessionID)
        else {
            return MCPEncoding.error(
                id: id, code: MCPProtocol.ErrorCode.noSession,
                message: "No active session — call initialize first (Mcp-Session-Id required).")
        }
        guard let params = object["params"]?.asObject, let name = params["name"]?.asString else {
            return MCPEncoding.error(
                id: id, code: MCPProtocol.ErrorCode.invalidParams,
                message: "tools/call requires params.name")
        }
        let arguments = params["arguments"]?.asObject ?? [:]
        let normalized = ToolArgumentNormalizer.normalize(arguments)
        let result = await executor.call(name, session: session, arguments: normalized)
        return MCPEncoding.result(id: id, MCPEncoding.toolCallResult(result))
    }

    // MARK: - Origin guard (DNS-rebinding protection)

    /// Reject requests whose `Origin` is a real remote site — the DNS-rebinding
    /// vector where a malicious page tries to reach the loopback server. Native
    /// clients (Claude Code) send no `Origin` and pass; a browser attack always
    /// carries its site's origin, which is not loopback.
    private func originRejection(_ request: HTTPRequest) -> HTTPResponse? {
        guard let origin = request.header("Origin") else { return nil }
        if origin == "null" { return nil }
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
            statusCode: status, statusText: Self.statusText(status), headers: headers, body: body)
    }

    private func empty(_ status: Int) -> HTTPResponse {
        HTTPResponse(
            statusCode: status, statusText: Self.statusText(status), headers: [], body: nil)
    }

    private func plain(_ status: Int, _ message: String) -> HTTPResponse {
        HTTPResponse(
            statusCode: status, statusText: Self.statusText(status),
            headers: [("Content-Type", "text/plain; charset=utf-8")],
            body: Data(message.utf8))
    }

    private static func statusText(_ status: Int) -> String {
        switch status {
        case 200: "OK"
        case 202: "Accepted"
        case 400: "Bad Request"
        case 403: "Forbidden"
        case 404: "Not Found"
        case 405: "Method Not Allowed"
        case 503: "Service Unavailable"
        default: "OK"
        }
    }
}
