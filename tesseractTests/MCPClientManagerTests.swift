import Foundation
import Testing
import MLXLMCommon
import WebKit

@testable import Tesseract_Agent

/// The PRD's headline seam: the **agent's tool surface**. Drives the
/// ``MCPClientManager`` against scripted HTTP servers (and the real in-app
/// Browser MCP server over the in-process transport) and asserts that tools
/// materialize as agent tools, calls round-trip, dead servers degrade cleanly,
/// namespacing prevents collisions, and enable/disable takes effect live.
@MainActor
struct MCPClientManagerTests {

    // MARK: - Helpers

    /// HTTP transport factory for configs whose `url` points at a fixture.
    private func httpTransportFactory() -> @MainActor (MCPServerConfig) -> any MCPTransport {
        { config in
            HTTPMCPTransport(
                endpoint: URL(string: config.url)!,
                session: URLSession(configuration: .ephemeral))
        }
    }

    private func makeManager(
        configs: [MCPServerConfig],
        makeTransport: @escaping @MainActor (MCPServerConfig) -> any MCPTransport,
        onRefresh: @escaping @MainActor () -> Void = {}
    ) -> MCPClientManager {
        MCPClientManager(
            configsProvider: { configs },
            makeTransport: makeTransport,
            refreshRegistry: onRefresh)
    }

    private func waitUntil(
        timeout: Duration = .seconds(3), _ predicate: @MainActor () -> Bool
    ) async {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if predicate() { return }
            try? await Task.sleep(for: .milliseconds(20))
        }
    }

    // MARK: - Materialization

    @Test
    func toolsMaterializeAsNamespacedAgentTools() async {
        let fixture = ScriptedMCPServer(tools: [
            .init(name: "search"), .init(name: "fetch"),
        ])
        await fixture.start()
        defer { fixture.stop() }

        let config = MCPServerConfig(
            name: "My Tools", url: fixture.endpoint.absoluteString, enabled: true)
        let manager = makeManager(configs: [config], makeTransport: httpTransportFactory())
        manager.sync(configs: [config])

        await waitUntil { manager.aggregatedToolDefinitions.count == 2 }

        let names = Set(manager.aggregatedToolDefinitions.map(\.name))
        #expect(names == ["my_tools.search", "my_tools.fetch"])
        // The same tools appear through the extension the registry aggregates.
        #expect(Set(manager.toolsExtension.tools.keys) == names)
        // Display label keeps the un-namespaced name (US #7).
        #expect(manager.aggregatedToolDefinitions.first?.label != nil)
    }

    @Test
    func materializedToolCallRoundTrips() async throws {
        let fixture = ScriptedMCPServer(
            tools: [.init(name: "echo")],
            onCall: { _, args in
                var outcome = ScriptedMCPServer.CallOutcome()
                outcome.content = [.text("you said \(args.string(for: "text") ?? "")")]
                return outcome
            })
        await fixture.start()
        defer { fixture.stop() }

        let config = MCPServerConfig(
            name: "svc", url: fixture.endpoint.absoluteString, enabled: true)
        let manager = makeManager(configs: [config], makeTransport: httpTransportFactory())
        manager.sync(configs: [config])
        await waitUntil { !manager.aggregatedToolDefinitions.isEmpty }

        let tool = try #require(
            manager.aggregatedToolDefinitions.first { $0.name == "svc.echo" })
        let result = try await tool.execute("call-1", ["text": .string("hi")], nil, nil)
        #expect(result.content.textContent == "you said hi")
    }

    // MARK: - Graceful degradation

    @Test
    func deadServerDegradesButOthersSurvive() async {
        let dead = ScriptedMCPServer(initializeStatus: 503)
        let healthy = ScriptedMCPServer(tools: [.init(name: "ok")])
        await dead.start()
        await healthy.start()
        defer { dead.stop(); healthy.stop() }

        let deadConfig = MCPServerConfig(
            name: "Dead", url: dead.endpoint.absoluteString, enabled: true)
        let healthyConfig = MCPServerConfig(
            name: "Healthy", url: healthy.endpoint.absoluteString, enabled: true)
        let manager = makeManager(
            configs: [deadConfig, healthyConfig], makeTransport: httpTransportFactory())
        manager.sync(configs: [deadConfig, healthyConfig])

        // The healthy server's tool shows up; the dead one contributes none.
        await waitUntil { manager.aggregatedToolDefinitions.contains { $0.name == "healthy.ok" } }

        #expect(manager.aggregatedToolDefinitions.map(\.name) == ["healthy.ok"])
        let deadState = manager.connection(id: deadConfig.id)?.state
        if case .failed = deadState {
            // expected — dead server degraded, kept its slot, contributes no tools
        } else {
            Issue.record("dead server should be .failed, was \(String(describing: deadState))")
        }
    }

    // MARK: - Namespacing

    @Test
    func sameToolNameFromTwoServersDoesNotCollide() async {
        let a = ScriptedMCPServer(tools: [.init(name: "run")])
        let b = ScriptedMCPServer(tools: [.init(name: "run")])
        await a.start()
        await b.start()
        defer { a.stop(); b.stop() }

        let configA = MCPServerConfig(name: "Alpha", url: a.endpoint.absoluteString, enabled: true)
        let configB = MCPServerConfig(name: "Beta", url: b.endpoint.absoluteString, enabled: true)
        let manager = makeManager(
            configs: [configA, configB], makeTransport: httpTransportFactory())
        manager.sync(configs: [configA, configB])
        await waitUntil { manager.aggregatedToolDefinitions.count == 2 }

        #expect(Set(manager.aggregatedToolDefinitions.map(\.name)) == ["alpha.run", "beta.run"])
    }

    // MARK: - Enable / disable

    @Test
    func disablingAServerRemovesItsToolsLive() async {
        let fixture = ScriptedMCPServer(tools: [.init(name: "ok")])
        await fixture.start()
        defer { fixture.stop() }

        var config = MCPServerConfig(
            name: "svc", url: fixture.endpoint.absoluteString, enabled: true)
        let manager = makeManager(configs: [config], makeTransport: httpTransportFactory())
        manager.sync(configs: [config])
        await waitUntil { !manager.aggregatedToolDefinitions.isEmpty }
        #expect(manager.aggregatedToolDefinitions.count == 1)

        config.enabled = false
        manager.sync(configs: [config])
        #expect(manager.aggregatedToolDefinitions.isEmpty)
        #expect(manager.toolsExtension.tools.isEmpty)
    }

    @Test
    func refreshHookFiresWhenToolsChange() async {
        let fixture = ScriptedMCPServer(tools: [.init(name: "ok")])
        await fixture.start()
        defer { fixture.stop() }

        let counter = Counter()
        let config = MCPServerConfig(
            name: "svc", url: fixture.endpoint.absoluteString, enabled: true)
        let manager = makeManager(
            configs: [config], makeTransport: httpTransportFactory(),
            onRefresh: { counter.bump() })
        manager.sync(configs: [config])
        await waitUntil { !manager.aggregatedToolDefinitions.isEmpty }
        #expect(counter.value >= 1)
    }

    // MARK: - In-process Browser server (ADR-0027 dogfood path)

    @Test
    func builtInBrowserServerMaterializesAllTwelveToolsInProcess() async {
        let browser = AgentBrowser(
            profile: AgentProfile(dataStore: .nonPersistent()),
            presenter: NoOpBrowserPresenter())
        let executor = BrowserToolExecutor(browser: browser)
        let server = MCPBrowserServer(browser: browser, executor: executor, isEnabled: { true })

        let manager = MCPClientManager(
            configsProvider: { [MCPServerConfig.builtInBrowser(enabled: true)] },
            makeTransport: { _ in
                InProcessMCPTransport(handle: { request in
                    await server.handle(request: request, origin: .inProcess)
                })
            })
        manager.sync(configs: [MCPServerConfig.builtInBrowser(enabled: true)])

        await waitUntil { manager.aggregatedToolDefinitions.count == 12 }

        let names = Set(manager.aggregatedToolDefinitions.map(\.name))
        #expect(names.count == 12)
        #expect(names.isSuperset(of: ["browser.navigate", "browser.read_page", "browser.click"]))
        // The live tool names must equal the static set the web-access switch
        // gates (US #16) — pins live namespacing to the gated set so a rename or
        // namespacing change can't silently break gating.
        #expect(names == MCPServerConfig.browserToolNames)
    }

    /// Thread-safe hit counter for the refresh hook.
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var _value = 0
        func bump() {
            lock.lock()
            _value += 1
            lock.unlock()
        }
        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return _value
        }
    }
}
