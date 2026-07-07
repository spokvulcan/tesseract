import Foundation
import MLXLMCommon

// MARK: - MCPClient

/// A transport-agnostic MCP **client** for one server: the handshake
/// (`initialize` → session id → `notifications/initialized`), `tools/list`, and
/// `tools/call` with progress streaming and cancellation. Speaks JSON-RPC over
/// an injected ``MCPTransport`` so the same logic drives an HTTP server and the
/// in-app Browser MCP server alike.
///
/// MainActor-isolated: it owns per-session mutable state (session id, request-id
/// counter) and is only ever driven from the agent's MainActor context. The
/// transport does its I/O off the MainActor.
@MainActor
final class MCPClient {

    private let transport: any MCPTransport
    private let clientName: String
    private let clientVersion: String

    private(set) var sessionID: String?
    private(set) var protocolVersion: String?
    private var nextRequestID = 1

    /// Fired when the server sends `notifications/tools/list_changed` while a
    /// call's stream is open (US #13). The connection re-lists in response.
    var onToolsListChanged: (@MainActor @Sendable () -> Void)?

    init(
        transport: any MCPTransport,
        clientName: String = "tesseract-agent",
        clientVersion: String = "1.0.0"
    ) {
        self.transport = transport
        self.clientName = clientName
        self.clientVersion = clientVersion
    }

    // MARK: - Handshake

    /// Negotiate the connection: `initialize`, capture the session id and agreed
    /// protocol version, then send the required `notifications/initialized`.
    func initialize() async throws {
        let params: JSONValue = .object([
            "protocolVersion": .string(MCPProtocol.version),
            "capabilities": .object([:]),
            "clientInfo": .object([
                "name": .string(clientName),
                "version": .string(clientVersion),
            ]),
        ])
        let response = try await send(
            MCPWire.request(id: allocateID(), method: "initialize", params: params))
        if let sessionID = response.sessionID { self.sessionID = sessionID }
        guard let result = try result(from: response) else {
            throw MCPClientError.malformedResponse("initialize returned no result")
        }
        protocolVersion = result.asObject?["protocolVersion"]?.asString
        try await notify(MCPWire.notification(method: "notifications/initialized", params: nil))
    }

    // MARK: - Tools

    func listTools() async throws -> [MCPToolDescriptor] {
        let response = try await send(
            MCPWire.request(id: allocateID(), method: "tools/list", params: nil))
        guard let result = try result(from: response) else { return [] }
        return MCPWire.parseToolsList(result)
    }

    /// Call a tool. Progress notifications stream to `onProgress`; a flip of
    /// `signal` cancels the in-flight request. MCP `isError` results are returned
    /// (not thrown) so the agent recovers in-conversation (US #10/#11).
    func callTool(
        name: String,
        arguments: [String: JSONValue],
        signal: CancellationToken? = nil,
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> MCPCallResult {
        let requestID = allocateID()
        var params: [String: JSONValue] = [
            "name": .string(name),
            "arguments": .object(arguments),
        ]
        if onProgress != nil {
            // Opt in to progress: give the server a token to tag notifications.
            params["_meta"] = .object(["progressToken": .string("call-\(requestID)")])
        }
        let request = MCPWire.request(id: requestID, method: "tools/call", params: .object(params))
        let response = try await sendCancellable(request, signal: signal, onProgress: onProgress)
        guard let result = try result(from: response) else {
            throw MCPClientError.malformedResponse("tools/call returned no result")
        }
        return MCPWire.parseCallResult(result)
    }

    // MARK: - Send

    private func allocateID() -> Int {
        defer { nextRequestID += 1 }
        return nextRequestID
    }

    /// Post a request and await its response, ignoring interim notifications.
    private func send(_ request: JSONValue) async throws -> MCPTransportResponse {
        try await transport.post(
            MCPWire.encode(request), sessionID: sessionID, onNotification: { _ in })
    }

    /// Post a notification (no response expected).
    private func notify(_ notification: JSONValue) async throws {
        _ = try await transport.post(
            MCPWire.encode(notification), sessionID: sessionID, onNotification: { _ in })
    }

    /// Post a request, routing interim notifications to `onProgress` /
    /// `onToolsListChanged`, and cancelling the in-flight request when `signal`
    /// flips. The `CancellationToken` is bridged to task cancellation by racing
    /// the request against a poll of the token.
    private func sendCancellable(
        _ request: JSONValue,
        signal: CancellationToken?,
        onProgress: (@Sendable (String) -> Void)?
    ) async throws -> MCPTransportResponse {
        if signal?.isCancelled == true { throw MCPClientError.cancelled }

        let listChanged = onToolsListChanged
        let onNotification: @Sendable (JSONValue) -> Void = { value in
            guard let object = value.asObject, let method = object["method"]?.asString else {
                return
            }
            switch method {
            case "notifications/progress":
                guard let onProgress else { return }
                let params = object["params"]?.asObject
                if let message = params?["message"]?.asString {
                    onProgress(message)
                } else if let progress = params?["progress"]?.asDouble {
                    let total = params?["total"]?.asDouble
                    onProgress(total.map { "progress \(progress)/\($0)" } ?? "progress \(progress)")
                }
            case "notifications/tools/list_changed":
                if let listChanged { Task { @MainActor in listChanged() } }
            default:
                break
            }
        }

        let body = MCPWire.encode(request)
        let sessionID = self.sessionID
        let transport = self.transport

        return try await withThrowingTaskGroup(of: MCPTransportResponse.self) { group in
            group.addTask {
                try await transport.post(
                    body, sessionID: sessionID, onNotification: onNotification)
            }
            if let signal {
                group.addTask {
                    while !signal.isCancelled {
                        try await Task.sleep(for: .milliseconds(50))
                    }
                    throw MCPClientError.cancelled
                }
            }
            defer { group.cancelAll() }
            guard let first = try await group.next() else {
                throw MCPClientError.malformedResponse("no response")
            }
            return first
        }
    }

    /// Extract the `result` from a response, converting a JSON-RPC `error`
    /// envelope into a thrown ``MCPClientError``.
    private func result(from response: MCPTransportResponse) throws -> JSONValue? {
        guard let json = response.json else { return nil }
        if let error = json.asObject?["error"]?.asObject {
            throw MCPClientError.rpc(
                code: error["code"]?.asInt ?? MCPProtocol.ErrorCode.internalError,
                message: error["message"]?.asString ?? "unknown error")
        }
        return json.asObject?["result"]
    }
}
