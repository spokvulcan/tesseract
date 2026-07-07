import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

/// Drives the real ``MCPClient`` + ``HTTPMCPTransport`` against a scripted HTTP
/// MCP server over an actual loopback socket — the handshake, tool listing, tool
/// calls, streamed progress, RPC errors, and cancellation an arbitrary
/// user-configured server would exercise.
@MainActor
struct MCPClientTests {

    private func makeClient(for fixture: ScriptedMCPServer) -> MCPClient {
        MCPClient(
            transport: HTTPMCPTransport(
                endpoint: fixture.endpoint,
                session: URLSession(configuration: .ephemeral)))
    }

    // MARK: - Handshake

    @Test
    func initializeCapturesSessionAndProtocol() async throws {
        let fixture = ScriptedMCPServer()
        await fixture.start()
        defer { fixture.stop() }

        let client = makeClient(for: fixture)
        try await client.initialize()

        #expect(client.sessionID?.isEmpty == false)
        #expect(client.protocolVersion == MCPProtocol.version)
    }

    @Test
    func initializeThrowsWhenServerRefuses() async {
        let fixture = ScriptedMCPServer(initializeStatus: 503)
        await fixture.start()
        defer { fixture.stop() }

        let client = makeClient(for: fixture)
        await #expect(throws: (any Error).self) {
            try await client.initialize()
        }
    }

    // MARK: - tools/list

    @Test
    func listToolsParsesNamesDescriptionsAndSchema() async throws {
        let fixture = ScriptedMCPServer(tools: [
            .init(
                name: "echo", description: "Echo text back",
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "text": .object([
                            "type": .string("string"), "description": .string("what to echo"),
                        ])
                    ]),
                    "required": .array([.string("text")]),
                ]))
        ])
        await fixture.start()
        defer { fixture.stop() }

        let client = makeClient(for: fixture)
        try await client.initialize()
        let tools = try await client.listTools()

        #expect(tools.count == 1)
        #expect(tools.first?.name == "echo")
        #expect(tools.first?.description == "Echo text back")

        // Schema decodes faithfully — required is load-bearing for loop validation.
        let schema = MCPSchemaDecoder.decode(tools[0].inputSchema)
        #expect(schema.required == ["text"])
        #expect(schema.properties["text"]?.type == "string")
    }

    // MARK: - tools/call

    @Test
    func callToolRoundTripsTextContent() async throws {
        let fixture = ScriptedMCPServer(
            tools: [.init(name: "echo")],
            onCall: { name, args in
                var outcome = ScriptedMCPServer.CallOutcome()
                outcome.content = [.text("echoed: \(args.string(for: "text") ?? "")")]
                return outcome
            })
        await fixture.start()
        defer { fixture.stop() }

        let client = makeClient(for: fixture)
        try await client.initialize()
        let result = try await client.callTool(name: "echo", arguments: ["text": .string("hi")])

        #expect(result.isError == false)
        #expect(result.content.textContent == "echoed: hi")
    }

    @Test
    func callToolDeliversImageBlocks() async throws {
        let pngBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x01, 0x02, 0x03])
        let fixture = ScriptedMCPServer(
            tools: [.init(name: "shoot")],
            onCall: { _, _ in
                var outcome = ScriptedMCPServer.CallOutcome()
                outcome.content = [.image(data: pngBytes, mimeType: "image/png"), .text("shot")]
                return outcome
            })
        await fixture.start()
        defer { fixture.stop() }

        let client = makeClient(for: fixture)
        try await client.initialize()
        let result = try await client.callTool(name: "shoot", arguments: [:])

        let images = result.content.compactMap { block -> Data? in
            if case .image(let data, _) = block { return data }
            return nil
        }
        #expect(images == [pngBytes])
        #expect(result.content.textContent == "shot")
    }

    @Test
    func callToolSurfacesIsErrorAsContentNotThrow() async throws {
        let fixture = ScriptedMCPServer(
            tools: [.init(name: "boom")],
            onCall: { _, _ in
                var outcome = ScriptedMCPServer.CallOutcome()
                outcome.content = [.text("No page open.")]
                outcome.isError = true
                return outcome
            })
        await fixture.start()
        defer { fixture.stop() }

        let client = makeClient(for: fixture)
        try await client.initialize()
        // isError comes back as a normal (non-throwing) result the agent can read.
        let result = try await client.callTool(name: "boom", arguments: [:])
        #expect(result.isError == true)
        #expect(result.content.textContent == "No page open.")
    }

    @Test
    func callToolThrowsOnRPCError() async throws {
        let fixture = ScriptedMCPServer(
            tools: [.init(name: "bad")],
            onCall: { _, _ in
                var outcome = ScriptedMCPServer.CallOutcome()
                outcome.rpcError = .init(code: -32602, message: "invalid params")
                return outcome
            })
        await fixture.start()
        defer { fixture.stop() }

        let client = makeClient(for: fixture)
        try await client.initialize()
        await #expect(throws: (any Error).self) {
            _ = try await client.callTool(name: "bad", arguments: [:])
        }
    }

    // MARK: - Progress streaming

    @Test
    func callToolStreamsProgress() async throws {
        let fixture = ScriptedMCPServer(
            tools: [.init(name: "work")],
            onCall: { _, _ in
                var outcome = ScriptedMCPServer.CallOutcome()
                outcome.progress = ["step 1", "step 2"]
                outcome.content = [.text("done")]
                return outcome
            })
        await fixture.start()
        defer { fixture.stop() }

        let collected = ProgressCollector()
        let client = makeClient(for: fixture)
        try await client.initialize()
        let result = try await client.callTool(
            name: "work", arguments: [:], onProgress: { collected.append($0) })

        #expect(result.content.textContent == "done")
        #expect(collected.messages == ["step 1", "step 2"])
    }

    // MARK: - Cancellation

    @Test
    func callToolCancellationPropagates() async throws {
        let fixture = ScriptedMCPServer(
            tools: [.init(name: "slow")],
            onCall: { _, _ in
                var outcome = ScriptedMCPServer.CallOutcome()
                outcome.delayMillis = 5_000
                return outcome
            })
        await fixture.start()
        defer { fixture.stop() }

        let client = makeClient(for: fixture)
        try await client.initialize()

        let signal = CancellationToken()
        let clock = ContinuousClock()
        let start = clock.now
        Task {
            try? await Task.sleep(for: .milliseconds(100))
            signal.cancel()
        }
        await #expect(throws: (any Error).self) {
            _ = try await client.callTool(name: "slow", arguments: [:], signal: signal)
        }
        // Returns promptly on cancel, not after the 5s server delay.
        #expect(clock.now - start < .seconds(2))
    }

    /// Thread-safe progress sink (the callback fires off the MainActor).
    private final class ProgressCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var _messages: [String] = []
        func append(_ message: String) {
            lock.lock()
            _messages.append(message)
            lock.unlock()
        }
        var messages: [String] {
            lock.lock()
            defer { lock.unlock() }
            return _messages
        }
    }
}
