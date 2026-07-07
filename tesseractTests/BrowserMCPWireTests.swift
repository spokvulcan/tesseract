import Foundation
import Testing
import WebKit

@testable import Tesseract_Agent

// MARK: - JSON-RPC client helper

/// A parsed JSON-RPC/MCP HTTP response. MainActor-isolated (holds `Any`); it
/// never leaves the MainActor test context.
@MainActor
private struct RPCResponse {
    let status: Int
    let sessionID: String?
    let json: [String: Any]?

    var result: [String: Any]? { json?["result"] as? [String: Any] }
    var error: [String: Any]? { json?["error"] as? [String: Any] }

    /// Joined text of every `text` content block in a `tools/call` result.
    var contentText: String {
        let content = (result?["content"] as? [[String: Any]]) ?? []
        return content.compactMap { $0["text"] as? String }.joined(separator: "\n")
    }

    var isError: Bool { (result?["isError"] as? Bool) ?? false }
}

/// POST one JSON-RPC message to the MCP endpoint and parse the reply.
@MainActor
private func rpc(
    port: UInt16,
    method: String,
    id: Int? = 1,
    params: [String: Any]? = nil,
    sessionID: String? = nil,
    origin: String? = nil
) async -> RPCResponse {
    var body: [String: Any] = ["jsonrpc": "2.0", "method": method]
    if let id { body["id"] = id }
    if let params { body["params"] = params }

    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/mcp")!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
    if let sessionID { request.setValue(sessionID, forHTTPHeaderField: "Mcp-Session-Id") }
    if let origin { request.setValue(origin, forHTTPHeaderField: "Origin") }
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)

    let session = URLSession(configuration: .ephemeral)
    guard let (data, response) = try? await session.data(for: request),
        let http = response as? HTTPURLResponse
    else {
        return RPCResponse(status: -1, sessionID: nil, json: nil)
    }
    let parsed = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    return RPCResponse(
        status: http.statusCode,
        sessionID: http.value(forHTTPHeaderField: "Mcp-Session-Id"),
        json: parsed)
}

@MainActor
private func waitForPort(_ server: HTTPServer) async -> UInt16 {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: .seconds(3))
    while clock.now < deadline {
        if let port = server.boundPort { return port }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return server.boundPort ?? 0
}

// MARK: - Fixtures

private enum Fixture {
    static let hello = """
        <!DOCTYPE html><html><head><title>Hello Fixture</title></head>
        <body><main>
        <h1>Welcome</h1>
        <p>This is the hello fixture page with some readable content about widgets and gadgets.</p>
        <a href="/page2">Go to page two</a>
        </main></body></html>
        """

    static let page2 = """
        <!DOCTYPE html><html><head><title>Page Two</title></head>
        <body><main><h1>Second</h1><p>You reached page two successfully.</p></main></body></html>
        """
}

/// Primary browser test seam: drive the **Browser MCP Server** exactly as an
/// external agent would — real HTTP, JSON-RPC, session id — with the Agent
/// Browser loading local fixture pages over loopback (no live internet) and a
/// no-op window presenter (no windows appear).
@MainActor
struct BrowserMCPWireTests {

    /// A running MCP endpoint plus the fixture "web" it browses.
    private struct Harness {
        let mcpPort: UInt16
        let fixturePort: UInt16
        let mcp: HTTPServer
        let fixtures: HTTPServer
        let browser: AgentBrowser
        /// Retained: the HTTP route holds the server weakly (as in production,
        /// where DependencyContainer owns it), so the harness must keep it alive.
        let mcpServer: MCPBrowserServer
    }

    private func makeHarness(enabled: Bool = true) async -> Harness {
        // Fixture "web" server.
        let fixtures = HTTPServer(port: 0)
        func serve(_ path: String, _ html: String) {
            fixtures.route(.GET, path) { _, writer in
                try await writer.send(
                    HTTPResponse(
                        statusCode: 200, statusText: "OK",
                        headers: [("Content-Type", "text/html; charset=utf-8")],
                        body: Data(html.utf8)))
            }
        }
        serve("/hello", Fixture.hello)
        serve("/page2", Fixture.page2)
        await fixtures.start()
        let fixturePort = await waitForPort(fixtures)

        // MCP server under test. Ephemeral profile so parallel test processes
        // never contend on the real persistent Agent Profile store.
        let browser = AgentBrowser(
            profile: AgentProfile(dataStore: .nonPersistent()),
            presenter: NoOpBrowserPresenter())
        let executor = BrowserToolExecutor(browser: browser)
        let mcpServer = MCPBrowserServer(
            browser: browser, executor: executor, isEnabled: { enabled })
        let mcp = HTTPServer(port: 0)
        mcpServer.attach(to: mcp)
        await mcp.start()
        let mcpPort = await waitForPort(mcp)

        return Harness(
            mcpPort: mcpPort, fixturePort: fixturePort, mcp: mcp, fixtures: fixtures,
            browser: browser, mcpServer: mcpServer)
    }

    private func initialize(_ port: UInt16) async -> String {
        let response = await rpc(
            port: port, method: "initialize",
            params: [
                "protocolVersion": "2025-06-18",
                "capabilities": [:],
                "clientInfo": ["name": "test", "version": "1"],
            ])
        return response.sessionID ?? ""
    }

    // MARK: - Handshake

    @Test
    func initializeReturnsSessionAndServerInfo() async {
        let h = await makeHarness()
        defer { h.mcp.stop(); h.fixtures.stop() }

        let response = await rpc(
            port: h.mcpPort, method: "initialize",
            params: ["protocolVersion": "2025-06-18", "capabilities": [:]])
        #expect(response.status == 200)
        #expect(!(response.sessionID ?? "").isEmpty)
        let serverInfo = response.result?["serverInfo"] as? [String: Any]
        #expect(serverInfo?["name"] as? String == "tesseract-agent-browser")
    }

    @Test
    func toolsListAdvertisesTheFullSurfaceWithSchemas() async {
        let h = await makeHarness()
        defer { h.mcp.stop(); h.fixtures.stop() }
        let session = await initialize(h.mcpPort)

        let response = await rpc(
            port: h.mcpPort, method: "tools/list", id: 2, sessionID: session)
        let tools = (response.result?["tools"] as? [[String: Any]]) ?? []
        let names = Set(tools.compactMap { $0["name"] as? String })

        #expect(tools.count == 12)
        #expect(names.isSuperset(of: ["navigate", "read_page", "page_map", "click", "type"]))
        // Every tool advertises an object inputSchema.
        for tool in tools {
            let schema = tool["inputSchema"] as? [String: Any]
            #expect(schema?["type"] as? String == "object")
        }
    }

    // MARK: - Read path

    @Test
    func navigateThenReadPageReturnsDistilledContent() async {
        let h = await makeHarness()
        defer { h.mcp.stop(); h.fixtures.stop() }
        let session = await initialize(h.mcpPort)

        let nav = await rpc(
            port: h.mcpPort, method: "tools/call", id: 2,
            params: [
                "name": "navigate",
                "arguments": ["url": "http://127.0.0.1:\(h.fixturePort)/hello"],
            ],
            sessionID: session)
        #expect(!nav.isError)
        #expect(nav.contentText.contains("Hello Fixture"))

        let read = await rpc(
            port: h.mcpPort, method: "tools/call", id: 3,
            params: ["name": "read_page", "arguments": [:]],
            sessionID: session)
        #expect(!read.isError)
        #expect(read.contentText.contains("readable content about widgets"))
    }

    // MARK: - Interaction path

    @Test
    func pageMapThenClickFollowsTheLink() async {
        let h = await makeHarness()
        defer { h.mcp.stop(); h.fixtures.stop() }
        let session = await initialize(h.mcpPort)

        _ = await rpc(
            port: h.mcpPort, method: "tools/call", id: 2,
            params: [
                "name": "navigate",
                "arguments": ["url": "http://127.0.0.1:\(h.fixturePort)/hello"],
            ],
            sessionID: session)

        let map = await rpc(
            port: h.mcpPort, method: "tools/call", id: 3,
            params: ["name": "page_map", "arguments": [:]],
            sessionID: session)
        #expect(!map.isError)
        #expect(map.contentText.contains("[1]"))
        #expect(map.contentText.contains("Go to page two"))

        let click = await rpc(
            port: h.mcpPort, method: "tools/call", id: 4,
            params: ["name": "click", "arguments": ["ref": 1]],
            sessionID: session)
        #expect(!click.isError)
        #expect(click.contentText.contains("/page2"))
    }

    // MARK: - Session isolation

    @Test
    func sessionsDoNotShareTabs() async {
        let h = await makeHarness()
        defer { h.mcp.stop(); h.fixtures.stop() }

        let sessionA = await initialize(h.mcpPort)
        let sessionB = await initialize(h.mcpPort)
        #expect(sessionA != sessionB)

        // A navigates.
        _ = await rpc(
            port: h.mcpPort, method: "tools/call", id: 2,
            params: [
                "name": "navigate",
                "arguments": ["url": "http://127.0.0.1:\(h.fixturePort)/hello"],
            ],
            sessionID: sessionA)

        // B has no page — reading fails, proving B doesn't see A's tab.
        let readB = await rpc(
            port: h.mcpPort, method: "tools/call", id: 3,
            params: ["name": "read_page", "arguments": [:]],
            sessionID: sessionB)
        #expect(readB.isError)
        #expect(readB.contentText.contains("No page open"))

        // A still reads its own page fine.
        let readA = await rpc(
            port: h.mcpPort, method: "tools/call", id: 4,
            params: ["name": "read_page", "arguments": [:]],
            sessionID: sessionA)
        #expect(!readA.isError)
        #expect(readA.contentText.contains("widgets"))
    }

    // MARK: - Guards

    @Test
    func toolCallWithoutSessionIsRejected() async {
        let h = await makeHarness()
        defer { h.mcp.stop(); h.fixtures.stop() }

        let response = await rpc(
            port: h.mcpPort, method: "tools/call", id: 2,
            params: ["name": "page_map", "arguments": [:]])
        #expect(response.error?["code"] as? Int == MCPProtocol.ErrorCode.noSession)
    }

    @Test
    func remoteOriginIsRejected() async {
        let h = await makeHarness()
        defer { h.mcp.stop(); h.fixtures.stop() }

        let response = await rpc(
            port: h.mcpPort, method: "initialize",
            params: ["protocolVersion": "2025-06-18"],
            origin: "https://evil.example.com")
        #expect(response.status == 403)
    }

    @Test
    func localhostOriginIsAllowed() async {
        let h = await makeHarness()
        defer { h.mcp.stop(); h.fixtures.stop() }

        let response = await rpc(
            port: h.mcpPort, method: "initialize",
            params: ["protocolVersion": "2025-06-18"],
            origin: "http://localhost:5173")
        #expect(response.status == 200)
    }

    @Test
    func disabledServerRefusesRequests() async {
        let h = await makeHarness(enabled: false)
        defer { h.mcp.stop(); h.fixtures.stop() }

        let response = await rpc(
            port: h.mcpPort, method: "initialize",
            params: ["protocolVersion": "2025-06-18"])
        #expect(response.status == 503)
    }
}
