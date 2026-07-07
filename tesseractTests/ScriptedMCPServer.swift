import Foundation
import MLXLMCommon

@testable import Tesseract_Agent

// MARK: - ScriptedMCPServer

/// A canned, in-process HTTP MCP **server fixture** — the PRD's named test seam.
/// Binds an ephemeral loopback port (parallel test processes never collide) and
/// answers `initialize` / `tools/list` / `tools/call` from a script, with
/// controllable delays, streamed progress, RPC errors, and hard failures.
///
/// Drive the real ``HTTPMCPTransport`` against `.endpoint` to exercise the whole
/// client stack over an actual socket, exactly as a user-configured server would
/// be reached.
@MainActor
final class ScriptedMCPServer {

    /// One tool the fixture advertises in `tools/list`.
    struct ToolSpec: Sendable {
        let name: String
        let description: String
        let inputSchema: JSONValue

        init(name: String, description: String = "", inputSchema: JSONValue? = nil) {
            self.name = name
            self.description = description
            self.inputSchema =
                inputSchema ?? .object(["type": .string("object"), "properties": .object([:])])
        }
    }

    /// What a scripted `tools/call` returns.
    struct CallOutcome: Sendable {
        var content: [ContentBlock] = [.text("ok")]
        var isError = false
        var rpcError: RPCError?
        /// Non-empty → stream these as `notifications/progress` over SSE before
        /// the final result.
        var progress: [String] = []
        var delayMillis: Int = 0

        struct RPCError: Sendable {
            let code: Int
            let message: String
        }

        static let ok = CallOutcome()
    }

    let server: HTTPServer
    private(set) var port: UInt16 = 0

    private let tools: [ToolSpec]
    private let onCall: @Sendable (_ name: String, _ arguments: [String: JSONValue]) -> CallOutcome
    private let initializeStatus: Int

    /// `initializeStatus` other than 200 makes `initialize` fail (e.g. 503 for a
    /// disabled/dead server).
    init(
        tools: [ToolSpec] = [],
        initializeStatus: Int = 200,
        onCall:
            @escaping @Sendable (_ name: String, _ arguments: [String: JSONValue]) ->
            CallOutcome = { _, _ in .ok }
    ) {
        self.tools = tools
        self.onCall = onCall
        self.initializeStatus = initializeStatus
        self.server = HTTPServer(port: 0)
        registerRoute()
    }

    func start() async {
        await server.start()
        port = await Self.waitForPort(server)
    }

    func stop() {
        server.stop()
    }

    var endpoint: URL { URL(string: "http://127.0.0.1:\(port)/mcp")! }

    // MARK: - Routing

    private func registerRoute() {
        let tools = self.tools
        let onCall = self.onCall
        let initializeStatus = self.initializeStatus

        // The nonisolated route closure only forwards Sendable values to the
        // MainActor dispatch — it never touches the request's isolated
        // properties itself (the pattern the real MCPBrowserServer route uses).
        server.route(.POST, "/mcp") { request, writer in
            try await Self.dispatch(
                request: request, writer: writer,
                tools: tools, onCall: onCall, initializeStatus: initializeStatus)
        }
    }

    private static func dispatch(
        request: HTTPRequest,
        writer: HTTPResponseWriter,
        tools: [ToolSpec],
        onCall: @Sendable (_ name: String, _ arguments: [String: JSONValue]) -> CallOutcome,
        initializeStatus: Int
    ) async throws {
        guard let body = request.body,
            let message = MCPWire.decode(body),
            let object = message.asObject,
            let method = object["method"]?.asString
        else {
            try await writer.send(
                Self.json(
                    MCPEncoding.error(
                        id: .null, code: MCPProtocol.ErrorCode.parseError,
                        message: "bad body"), status: 400))
            return
        }
        let id = object["id"] ?? .null

        switch method {
        case "initialize":
            guard initializeStatus == 200 else {
                try await writer.send(
                    HTTPResponse(
                        statusCode: initializeStatus,
                        statusText: HTTPResponse.statusText(for: initializeStatus),
                        headers: [("Content-Type", "text/plain")],
                        body: Data("unavailable".utf8)))
                return
            }
            let result = MCPEncoding.result(
                id: id, MCPEncoding.initializeResult(protocolVersion: MCPProtocol.version))
            try await writer.send(Self.json(result, sessionID: UUID().uuidString))

        case "notifications/initialized":
            try await writer.send(Self.empty(202))

        case "tools/list":
            let toolsJSON = JSONValue.object([
                "tools": .array(
                    tools.map { tool in
                        .object([
                            "name": .string(tool.name),
                            "description": .string(tool.description),
                            "inputSchema": tool.inputSchema,
                        ])
                    })
            ])
            try await writer.send(Self.json(MCPEncoding.result(id: id, toolsJSON)))

        case "tools/call":
            let params = object["params"]?.asObject
            let name = params?["name"]?.asString ?? ""
            let arguments = params?["arguments"]?.asObject ?? [:]
            let token = params?["_meta"]?.asObject?["progressToken"]?.asString
            let outcome = onCall(name, arguments)
            try await Self.respondToCall(
                id: id, token: token, outcome: outcome, writer: writer)

        default:
            try await writer.send(
                Self.json(
                    MCPEncoding.error(
                        id: id, code: MCPProtocol.ErrorCode.methodNotFound,
                        message: "unknown method \(method)")))
        }
    }

    private static func respondToCall(
        id: JSONValue,
        token: String?,
        outcome: CallOutcome,
        writer: HTTPResponseWriter
    ) async throws {
        if outcome.delayMillis > 0 {
            try? await Task.sleep(for: .milliseconds(outcome.delayMillis))
        }

        if let rpcError = outcome.rpcError {
            try await writer.send(
                json(MCPEncoding.error(id: id, code: rpcError.code, message: rpcError.message)))
            return
        }

        let result = JSONValue.object([
            "content": .array(outcome.content.map(MCPEncoding.content)),
            "isError": .bool(outcome.isError),
        ])

        guard !outcome.progress.isEmpty else {
            try await writer.send(json(MCPEncoding.result(id: id, result)))
            return
        }

        // Streamable-HTTP SSE: progress notifications, then the final response.
        try await writer.beginStreaming(headers: [("Content-Type", "text/event-stream")])
        for (index, text) in outcome.progress.enumerated() {
            var params: [String: JSONValue] = [
                "progress": .double(Double(index + 1)),
                "total": .double(Double(outcome.progress.count)),
                "message": .string(text),
            ]
            if let token { params["progressToken"] = .string(token) }
            let note = MCPWire.notification(
                method: "notifications/progress", params: .object(params))
            try await writer.writeChunk(sseData(note))
        }
        try await writer.writeChunk(sseData(MCPEncoding.result(id: id, result)))
        try await writer.finish()
    }

    // MARK: - Response builders

    private static func json(_ value: JSONValue, status: Int = 200, sessionID: String? = nil)
        -> HTTPResponse
    {
        var headers: [(name: String, value: String)] = [("Content-Type", "application/json")]
        if let sessionID { headers.append(("Mcp-Session-Id", sessionID)) }
        return HTTPResponse(
            statusCode: status, statusText: HTTPResponse.statusText(for: status),
            headers: headers, body: MCPWire.encode(value))
    }

    private static func empty(_ status: Int) -> HTTPResponse {
        HTTPResponse(
            statusCode: status, statusText: HTTPResponse.statusText(for: status), headers: [],
            body: nil)
    }

    private static func sseData(_ value: JSONValue) -> Data {
        var data = Data("data: ".utf8)
        data.append(MCPWire.encode(value))
        data.append(Data("\n\n".utf8))
        return data
    }

    static func waitForPort(_ server: HTTPServer) async -> UInt16 {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(3))
        while clock.now < deadline {
            if let port = server.boundPort { return port }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return server.boundPort ?? 0
    }
}
