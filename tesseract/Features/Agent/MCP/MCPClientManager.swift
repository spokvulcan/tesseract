import Foundation
import Observation

// MARK: - MCPClientManager

/// Owns the agent's live MCP connections and keeps them reconciled with the
/// configured server list. Aggregates every connected server's tools and pushes
/// them into its ``MCPToolsExtension`` so they appear in the agent's registry
/// alongside built-ins.
///
/// The built-in Browser server is just the first configured server (ADR-0027
/// dogfooding); the manager treats it like any other.
@MainActor
@Observable
final class MCPClientManager {

    private(set) var connections: [MCPServerConnection] = []

    /// The extension registered with the `ExtensionHost`; its tool set is this
    /// manager's aggregated MCP tools.
    let toolsExtension: MCPToolsExtension

    private let configsProvider: @MainActor () -> [MCPServerConfig]
    private let makeTransport: @MainActor (MCPServerConfig) -> any MCPTransport
    private let refreshRegistry: @MainActor () -> Void

    @ObservationIgnored private var observationTask: Task<Void, Never>?

    init(
        configsProvider: @escaping @MainActor () -> [MCPServerConfig],
        makeTransport: @escaping @MainActor (MCPServerConfig) -> any MCPTransport,
        refreshRegistry: @escaping @MainActor () -> Void = {}
    ) {
        self.configsProvider = configsProvider
        self.makeTransport = makeTransport
        self.refreshRegistry = refreshRegistry
        self.toolsExtension = MCPToolsExtension()
    }

    // MARK: - Lifecycle

    /// Reconcile to the current config list, then watch for changes and
    /// re-reconcile live (enable/disable and edits take effect without a
    /// restart, US #4).
    func start() {
        resync()
        observationTask?.cancel()
        observationTask = Task { [weak self] in
            guard let self else { return }
            for await _ in Observations({ self.configsProvider() }) {
                self.resync()
            }
        }
    }

    func stop() {
        observationTask?.cancel()
        observationTask = nil
        connections.removeAll()
        handleToolsChanged()
    }

    /// Re-read the configured servers and reconcile.
    func resync() {
        sync(configs: configsProvider())
    }

    // MARK: - Reconciliation

    /// Bring live connections in line with `configs`: drop connections that are
    /// gone, disabled, or materially changed; create + connect any that are new.
    func sync(configs: [MCPServerConfig]) {
        let desired = configs.filter { $0.enabled }
        var desiredByID: [String: MCPServerConfig] = [:]
        for config in desired { desiredByID[config.id] = config }

        connections.removeAll { connection in
            guard let config = desiredByID[connection.config.id] else { return true }
            // A materially-changed config (url, headers, name, …) reconnects fresh.
            return config != connection.config
        }

        for config in desired where !connections.contains(where: { $0.config.id == config.id }) {
            let connection = MCPServerConnection(
                config: config,
                transport: makeTransport(config),
                onChange: { [weak self] in self?.handleToolsChanged() })
            connections.append(connection)
            Task { await connection.connect() }
        }

        handleToolsChanged()
    }

    // MARK: - Tools

    /// Every connected server's tools, namespaced. Built-ins keep precedence at
    /// the `ToolRegistry` layer; among servers, namespacing prevents collisions.
    var aggregatedToolDefinitions: [AgentToolDefinition] {
        connections.flatMap { $0.toolDefinitions }
    }

    /// Look up a connection by config id (settings UI, tests).
    func connection(id: String) -> MCPServerConnection? {
        connections.first { $0.config.id == id }
    }

    private func handleToolsChanged() {
        toolsExtension.update(aggregatedToolDefinitions)
        refreshRegistry()
    }
}
