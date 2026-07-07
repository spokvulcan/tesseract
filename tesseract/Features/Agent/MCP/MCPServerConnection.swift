import Foundation
import MLXLMCommon

// MARK: - MCPServerConnection

/// One live connection to a configured MCP server: owns its ``MCPClient``,
/// tracks connection state and the discovered tool list (both observable for the
/// settings UI), and projects those tools as namespaced ``AgentToolDefinition``s.
///
/// Failure is graceful (US #6): a server that will not connect lands in
/// `.failed` with no tools, and everything else keeps working.
@MainActor
@Observable
final class MCPServerConnection {

    let config: MCPServerConfig
    private let client: MCPClient
    private let onChange: @MainActor () -> Void

    private(set) var state: MCPConnectionState = .idle
    private(set) var tools: [MCPToolDescriptor] = []

    init(
        config: MCPServerConfig,
        transport: any MCPTransport,
        onChange: @escaping @MainActor () -> Void
    ) {
        self.config = config
        self.client = MCPClient(transport: transport)
        self.onChange = onChange
        // React to a server's tools-changed push so our surface stays current
        // without a restart (US #13).
        client.onToolsListChanged = { [weak self] in
            Task { await self?.refreshTools() }
        }
    }

    /// Negotiate the connection and fetch the tool list. Never throws — a failure
    /// is recorded as `.failed` so one dead server can't brick the agent.
    func connect() async {
        state = .connecting
        onChange()
        do {
            try await client.initialize()
            tools = try await client.listTools()
            state = .connected
        } catch {
            tools = []
            state = .failed(Self.describe(error))
        }
        onChange()
    }

    /// Re-list tools (after a `tools/list_changed` notification).
    func refreshTools() async {
        guard state == .connected else { return }
        if let refreshed = try? await client.listTools() {
            tools = refreshed
            onChange()
        }
    }

    /// The server's tools as agent tools, namespaced by server (US #7/#8).
    var toolDefinitions: [AgentToolDefinition] {
        let invoke = makeInvoke()
        let namespace = config.namespace
        return tools.map { descriptor in
            MCPToolAdapter.toolDefinition(
                descriptor: descriptor, namespace: namespace, invoke: invoke)
        }
    }

    // MARK: - Private

    /// The Sendable closure the adapter uses to invoke a tool over this
    /// connection's client. Capturing the MainActor client is safe (it is
    /// Sendable); the call hops back to the MainActor.
    private func makeInvoke() -> MCPToolAdapter.Invoke {
        let client = self.client
        return { name, arguments, signal, onProgress in
            try await client.callTool(
                name: name, arguments: arguments, signal: signal, onProgress: onProgress)
        }
    }

    private static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}
