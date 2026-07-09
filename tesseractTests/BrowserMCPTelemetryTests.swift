import AppKit
import Foundation
import MLXLMCommon
import Testing
import WebKit

@testable import Tesseract_Agent

// MARK: - BrowserMCPTelemetryTests

/// The Browser MCP telemetry corpus (ADR-0031): every MCP interaction —
/// from either entry path — lands as a decodable JSONL event with the
/// fields tool-improvement analysis needs (origin, client identity,
/// arguments, latency, outcome, result shape, screenshot dimensions).
@MainActor
struct BrowserMCPTelemetryTests {

    /// A recorder writing into a private temp directory, plus the reader
    /// side pointed at the same place.
    private func makeRecorder(
        enabled: Bool = true
    ) -> (recorder: BrowserMCPTelemetryRecorder, directory: URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("BrowserMCPTelemetryTests-\(UUID().uuidString)")
        let recorder = BrowserMCPTelemetryRecorder(
            log: BrowserMCPTelemetryLog(directory: directory),
            isEnabled: { enabled })
        return (recorder, directory)
    }

    private func events(in directory: URL, from recorder: BrowserMCPTelemetryRecorder)
        -> [BrowserMCPTelemetryEvent]
    {
        recorder.flushForTesting()
        return BrowserMCPTelemetryLog.readEvents(
            at: BrowserMCPTelemetryLog.telemetryFiles(in: directory))
    }

    /// A minimal valid PNG of the given pixel size.
    private func pngFixture(width: Int, height: Int) -> Data {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        return rep.representation(using: .png, properties: [:])!
    }

    // MARK: - Recorder events

    @Test
    func sessionStartCapturesClientIdentityAndOrigin() {
        let (recorder, directory) = makeRecorder()
        recorder.recordSessionStart(
            sessionID: "session-1", origin: .http,
            params: [
                "protocolVersion": .string("2025-06-18"),
                "clientInfo": .object([
                    "name": .string("opencode"), "version": .string("0.3.1"),
                ]),
            ])

        let events = events(in: directory, from: recorder)
        #expect(events.count == 1)
        #expect(events.first?.kind == .sessionStart)
        #expect(events.first?.origin == .http)
        #expect(events.first?.sessionID == "session-1")
        #expect(events.first?.clientName == "opencode")
        #expect(events.first?.clientVersion == "0.3.1")
    }

    @Test
    func toolCallCapturesArgumentsLatencyOutcomeAndSequence() {
        let (recorder, directory) = makeRecorder()
        recorder.recordSessionStart(
            sessionID: "session-1", origin: .inProcess,
            params: ["clientInfo": .object(["name": .string("tesseract-agent")])])

        recorder.recordToolCall(
            sessionID: "session-1", origin: .inProcess, tool: "navigate",
            arguments: ["url": .string("https://example.com")],
            result: .text("Loaded https://example.com"),
            duration: .milliseconds(42))
        recorder.recordToolCall(
            sessionID: "session-1", origin: .inProcess, tool: "read_page",
            arguments: [:],
            result: .error("No page open."),
            duration: .milliseconds(3))

        let all = events(in: directory, from: recorder)
        let calls = all.filter { $0.kind == .toolCall }
        #expect(calls.count == 2)

        let navigate = calls[0]
        #expect(navigate.tool == "navigate")
        #expect(navigate.seq == 1)
        #expect(navigate.origin == .inProcess)
        #expect(navigate.clientName == "tesseract-agent")
        #expect(navigate.arguments == .object(["url": .string("https://example.com")]))
        #expect(navigate.outcome == .ok)
        #expect(navigate.errorMessage == nil)
        #expect(navigate.durationMS.map { abs($0 - 42) < 1 } == true)
        #expect(navigate.resultTextChars == "Loaded https://example.com".count)
        #expect(navigate.resultTextPreview == "Loaded https://example.com")

        let read = calls[1]
        #expect(read.seq == 2)
        #expect(read.outcome == .error)
        #expect(read.errorMessage == "No page open.")
    }

    @Test
    func toolCallRecordsImageDimensionsAndBytes() {
        let (recorder, directory) = makeRecorder()
        recorder.recordSessionStart(sessionID: "s", origin: .http, params: nil)
        let png = pngFixture(width: 320, height: 200)

        recorder.recordToolCall(
            sessionID: "s", origin: .http, tool: "screenshot", arguments: [:],
            result: .blocks([
                .image(data: png, mimeType: "image/png"),
                .text("Screenshot of https://example.com"),
            ]),
            duration: .milliseconds(120))

        let call = events(in: directory, from: recorder).first { $0.kind == .toolCall }
        let image = call?.resultImages?.first
        #expect(call?.resultImages?.count == 1)
        #expect(image?.width == 320)
        #expect(image?.height == 200)
        #expect(image?.bytes == png.count)
        #expect(image?.mimeType == "image/png")
        // Total payload = image bytes + text bytes.
        #expect(call?.resultBytes == png.count + "Screenshot of https://example.com".utf8.count)
    }

    @Test
    func longStringsAreCappedButOriginalSizeIsPreserved() {
        let (recorder, directory) = makeRecorder()
        recorder.recordSessionStart(sessionID: "s", origin: .http, params: nil)
        let script = String(repeating: "x", count: 10_000)

        recorder.recordToolCall(
            sessionID: "s", origin: .http, tool: "evaluate",
            arguments: ["script": .string(script)],
            result: .text(String(repeating: "y", count: 30_000)),
            duration: .milliseconds(5))

        let call = events(in: directory, from: recorder).first { $0.kind == .toolCall }
        #expect(call?.argumentsTruncated == true)
        // Original encoded size is preserved even though the value is capped.
        #expect((call?.argumentsBytes ?? 0) > 10_000)
        if case .object(let object)? = call?.arguments, case .string(let capped)? = object["script"]
        {
            #expect(capped.count <= BrowserMCPTelemetryRecorder.maxArgumentChars + 12)
            #expect(capped.hasSuffix("…[truncated]"))
        } else {
            Issue.record("expected capped script argument")
        }
        #expect(call?.resultTextChars == 30_000)
        #expect(call?.resultTextTruncated == true)
        #expect(
            call?.resultTextPreview?.count == BrowserMCPTelemetryRecorder.maxPreviewChars)
    }

    @Test
    func protocolErrorsAndSessionEndAreRecorded() {
        let (recorder, directory) = makeRecorder()
        recorder.recordSessionStart(sessionID: "s", origin: .http, params: nil)
        recorder.recordProtocolError(
            method: "tools/call", code: MCPProtocol.ErrorCode.noSession,
            message: "No active session", sessionID: nil, origin: .http)
        recorder.recordSessionEnd(sessionID: "s", origin: .http)

        let all = events(in: directory, from: recorder)
        let error = all.first { $0.kind == .protocolError }
        #expect(error?.errorCode == MCPProtocol.ErrorCode.noSession)
        #expect(error?.method == "tools/call")
        #expect(all.last?.kind == .sessionEnd)
    }

    @Test
    func disabledRecorderWritesNothing() {
        let (recorder, directory) = makeRecorder(enabled: false)
        recorder.recordSessionStart(sessionID: "s", origin: .http, params: nil)
        recorder.recordToolCall(
            sessionID: "s", origin: .http, tool: "navigate", arguments: [:],
            result: .text("ok"), duration: .milliseconds(1))
        recorder.flushForTesting()

        #expect(BrowserMCPTelemetryLog.telemetryFiles(in: directory).isEmpty)
    }

    @Test
    func fileOpensWithSchemaHeader() {
        let (recorder, directory) = makeRecorder()
        recorder.recordSessionStart(sessionID: "s", origin: .http, params: nil)
        recorder.flushForTesting()

        let files = BrowserMCPTelemetryLog.telemetryFiles(in: directory)
        #expect(files.count == 1)
        let lines = BrowserMCPTelemetryLog.readLines(at: files[0])
        guard case .header(let header)? = lines.first else {
            Issue.record("expected header line")
            return
        }
        #expect(header.schemaVersion == BrowserMCPTelemetryEvent.currentSchemaVersion)
        #expect(header.serverName == MCPProtocol.serverName)
    }

    // MARK: - Through the server (both entry paths)

    /// Build one JSON-RPC POST the way the in-process transport does.
    private func post(_ body: [String: Any], sessionID: String? = nil) -> HTTPRequest {
        var headers: [(name: String, value: String)] = [("Content-Type", "application/json")]
        if let sessionID { headers.append(("Mcp-Session-Id", sessionID)) }
        return HTTPRequest(
            method: .POST, path: "/mcp", headers: headers,
            body: try? JSONSerialization.data(withJSONObject: body))
    }

    private func sessionID(from response: HTTPResponse) -> String? {
        response.headers.first { $0.name.lowercased() == "mcp-session-id" }?.value
    }

    /// Drive the server exactly as the in-app agent does (in-process, no
    /// socket): the telemetry must attribute every event to `in_process`
    /// and stamp the client identity from `initialize`.
    @Test
    func inProcessCallsAreAttributedToTheInProcessOrigin() async {
        let (recorder, directory) = makeRecorder()
        let browser = AgentBrowser(
            profile: AgentProfile(dataStore: .nonPersistent()),
            presenter: NoOpBrowserPresenter())
        let server = MCPBrowserServer(
            browser: browser, executor: BrowserToolExecutor(browser: browser),
            isEnabled: { true }, telemetry: recorder)

        let initialize = await server.handle(
            request: post([
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": [
                    "protocolVersion": "2025-06-18",
                    "clientInfo": ["name": "tesseract-agent", "version": "1.0.0"],
                ],
            ]),
            origin: .inProcess)
        let session = sessionID(from: initialize)
        #expect(session != nil)

        // A tool call that fails fast (no page open) still lands in telemetry.
        _ = await server.handle(
            request: post(
                [
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": ["name": "read_page", "arguments": [:]],
                ],
                sessionID: session),
            origin: .inProcess)

        let all = events(in: directory, from: recorder)
        let start = all.first { $0.kind == .sessionStart }
        #expect(start?.origin == .inProcess)
        #expect(start?.clientName == "tesseract-agent")
        let call = all.first { $0.kind == .toolCall }
        #expect(call?.origin == .inProcess)
        #expect(call?.tool == "read_page")
        #expect(call?.outcome == .error)
        #expect((call?.durationMS ?? -1) >= 0)
    }

    /// The HTTP path (external clients: OpenCode, Claude Code) attributes
    /// events to `http`, and a session-less call records a protocol error.
    @Test
    func httpCallsAreAttributedToTheHTTPOrigin() async {
        let (recorder, directory) = makeRecorder()
        let browser = AgentBrowser(
            profile: AgentProfile(dataStore: .nonPersistent()),
            presenter: NoOpBrowserPresenter())
        let server = MCPBrowserServer(
            browser: browser, executor: BrowserToolExecutor(browser: browser),
            isEnabled: { true }, telemetry: recorder)

        let initialize = await server.handleOverHTTP(
            request: post([
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": [
                    "protocolVersion": "2025-06-18",
                    "clientInfo": ["name": "opencode", "version": "0.3.1"],
                ],
            ]))
        #expect(sessionID(from: initialize) != nil)

        // tools/call without a session: protocol error, attributed to http.
        _ = await server.handleOverHTTP(
            request: post([
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": ["name": "read_page", "arguments": [:]],
            ]))

        let all = events(in: directory, from: recorder)
        let start = all.first { $0.kind == .sessionStart }
        #expect(start?.origin == .http)
        #expect(start?.clientName == "opencode")
        let error = all.first { $0.kind == .protocolError }
        #expect(error?.origin == .http)
        #expect(error?.errorCode == MCPProtocol.ErrorCode.noSession)
    }

    /// `closeAllSessions` (server stop / app termination) records a
    /// shutdown event per open session with its final call count.
    @Test
    func serverShutdownRecordsEveryOpenSession() async {
        let (recorder, directory) = makeRecorder()
        let browser = AgentBrowser(
            profile: AgentProfile(dataStore: .nonPersistent()),
            presenter: NoOpBrowserPresenter())
        let server = MCPBrowserServer(
            browser: browser, executor: BrowserToolExecutor(browser: browser),
            isEnabled: { true }, telemetry: recorder)

        _ = await server.handle(
            request: post([
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": ["protocolVersion": "2025-06-18"],
            ]),
            origin: .inProcess)
        server.closeAllSessions()

        let all = events(in: directory, from: recorder)
        #expect(all.contains { $0.kind == .serverShutdown })
    }
}
