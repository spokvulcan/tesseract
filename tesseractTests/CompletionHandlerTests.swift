import Foundation
import Testing
@testable import Tesseract_Agent

struct CompletionHandlerTests {

    // MARK: - LeaseAcquiredSignal

    @Test func signalStartsFalse() {
        let signal = LeaseAcquiredSignal()
        #expect(!signal.isSet)
    }

    @Test func signalBecomesTrue() {
        let signal = LeaseAcquiredSignal()
        signal.set()
        #expect(signal.isSet)
    }

    @Test func signalSetIsIdempotent() {
        let signal = LeaseAcquiredSignal()
        signal.set()
        signal.set()
        #expect(signal.isSet)
    }

    // MARK: - withAcquisitionTimeout

    @Test func timeoutThrowsWhenBodyNeverSignals() async {
        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 50_000_000
            ) { _ in
                try await Task.sleep(nanoseconds: 5_000_000_000)
            }
            Issue.record("Expected LeaseTimeoutError")
        } catch is LeaseTimeoutError {
            // Expected
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test func longBodyNotCancelledAfterSignal() async throws {
        let completed = LeaseAcquiredSignal()

        try await CompletionHandler.withAcquisitionTimeout(
            timeoutNanoseconds: 100_000_000
        ) { signal in
            signal.set()
            try await Task.sleep(nanoseconds: 300_000_000)
            completed.set()
        }

        #expect(completed.isSet)
    }

    @Test func fastBodyCompletesBeforeTimeout() async throws {
        let completed = LeaseAcquiredSignal()

        try await CompletionHandler.withAcquisitionTimeout(
            timeoutNanoseconds: 1_000_000_000
        ) { signal in
            signal.set()
            completed.set()
        }

        #expect(completed.isSet)
    }

    @Test func bodyErrorPropagatesNotTimeout() async {
        struct BodyError: Error {}

        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 1_000_000_000
            ) { signal in
                signal.set()
                throw BodyError()
            }
            Issue.record("Expected BodyError")
        } catch is BodyError {
            // Expected
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test func bodyErrorBeforeSignalPropagates() async {
        struct EarlyError: Error {}

        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 1_000_000_000
            ) { _ in
                throw EarlyError()
            }
            Issue.record("Expected EarlyError")
        } catch is EarlyError {
            // Expected
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
}

// MARK: - HTTPServer Integration Tests

@MainActor
struct HTTPServerIntegrationTests {

    @Test func healthEndpointReturnsOK() async throws {
        let server = HTTPServer(port: 0)
        server.route(.GET, "/health") { _, writer in
            try await writer.send(.json(["status": "ok"] as [String: String]))
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 200)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: String]
        #expect(json["status"] == "ok")
    }

    @Test func notFoundForUnknownPath() async throws {
        let server = HTTPServer(port: 0)
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/nonexistent")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 404)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let error = json["error"] as? [String: Any]
        #expect(error?["code"] as? Int == 404)
    }

    @Test func methodNotAllowedReturnsAllowHeader() async throws {
        let server = HTTPServer(port: 0)
        server.route(.POST, "/only-post") { _, writer in
            try await writer.send(.json(["ok": true] as [String: Bool]))
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/only-post")!)
        request.httpMethod = "GET"
        let (_, response) = try await URLSession.shared.data(for: request)
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 405)
        #expect(http.value(forHTTPHeaderField: "Allow")?.contains("POST") == true)
    }

    @Test func sseStreamDeliversChunksAndDone() async throws {
        let server = HTTPServer(port: 0)
        server.route(.GET, "/test-sse") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()
            for i in 1...3 {
                await sse.send(["n": "\(i)"] as [String: String])
            }
            await sse.done()
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (bytes, response) = try await URLSession.shared.bytes(
            from: URL(string: "http://127.0.0.1:\(port)/test-sse")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 200)

        var lines: [String] = []
        for try await line in bytes.lines {
            lines.append(line)
            if line == "data: [DONE]" { break }
        }

        let dataLines = lines.filter { $0.hasPrefix("data: {") }
        #expect(dataLines.count == 3)
        #expect(lines.last == "data: [DONE]")
    }

    @Test func sseStreamSupportsReasoningChunks() async throws {
        let server = HTTPServer(port: 0)
        server.route(.GET, "/test-reasoning-sse") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()

            await sse.sendRaw(
                #"{"choices":[{"delta":{"reasoning_content":"Thinking...","role":"assistant"},"index":0}],"created":1712345678,"id":"chatcmpl-reason","model":"qwen3.5-4b-paro","object":"chat.completion.chunk"}"#
            )
            await sse.sendRaw(
                #"{"choices":[{"delta":{"content":"Hello"},"index":0}],"created":1712345678,"id":"chatcmpl-reason","model":"qwen3.5-4b-paro","object":"chat.completion.chunk"}"#
            )
            await sse.done()
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (bytes, response) = try await URLSession.shared.bytes(
            from: URL(string: "http://127.0.0.1:\(port)/test-reasoning-sse")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 200)

        var payloads: [OpenAI.ChatCompletionChunk] = []
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            if line == "data: [DONE]" { break }

            let json = Data(line.dropFirst(6).utf8)
            payloads.append(try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: json))
        }

        #expect(payloads.count == 2)
        #expect(payloads[0].choices[0].delta.reasoning_content == "Thinking...")
        #expect(payloads[0].choices[0].delta.content == nil)
        #expect(payloads[1].choices[0].delta.content == "Hello")
        #expect(payloads.allSatisfy { $0.choices[0].delta.content != "<think>" })
    }

    @Test func sseWriterDetectsDisconnect() async throws {
        // Verify that SSEWriter.send returns false when the connection fails,
        // and that the handler does not run all 200 iterations.
        let chunksSent = LeaseAcquiredSignal() // reuse as "at least some sent" flag
        let handlerDone = LeaseAcquiredSignal()

        let server = HTTPServer(port: 0)
        server.route(.GET, "/slow-sse") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()
            var sent = 0
            for i in 1...200 {
                let ok = await sse.send(["n": "\(i)"] as [String: String])
                if !ok { break }
                sent += 1
                if sent == 2 { chunksSent.set() }
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
            handlerDone.set()
        }
        let port = try await startOnRandomPort(server)

        // Connect, read 1 chunk to confirm stream works, then stop server
        let readTask = Task {
            let (bytes, _) = try await URLSession.shared.bytes(
                from: URL(string: "http://127.0.0.1:\(port)/slow-sse")!
            )
            for try await line in bytes.lines {
                if line.hasPrefix("data: {") { break }
            }
        }

        try? await readTask.value
        // Server stop cancels connection tasks, causing writes to fail
        server.stop()

        try await Task.sleep(nanoseconds: 500_000_000)
        // Handler must have exited (not still running all 200 iterations)
        #expect(handlerDone.isSet)
    }

    // MARK: - Prefill Disconnect

    @Test func prefillDisconnectCancelsGeneration() async throws {
        // Exercises the exact task-group pattern from runStreamingCompletion:
        // keepalive detects disconnect → throws → cancels generation child task.
        let generationCancelled = LeaseAcquiredSignal()
        let handlerExited = LeaseAcquiredSignal()

        let server = HTTPServer(port: 0)
        server.route(.GET, "/prefill-disconnect") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()

            // Send initial role chunk so the client can connect
            await sse.send(["role": "assistant"] as [String: String])

            // Simulate the CompletionHandler task group pattern:
            // keepalive monitors for disconnect, generation blocks on prefill.
            struct Disconnected: Error {}

            do {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    // Keepalive: check connection every 100ms (fast for testing)
                    group.addTask {
                        while true {
                            try await Task.sleep(nanoseconds: 100_000_000)
                            try Task.checkCancellation()
                            guard await sse.keepalive("keepalive") else {
                                throw Disconnected()
                            }
                        }
                    }

                    // Fake "generation" that blocks for 30s (simulating prefill)
                    group.addTask {
                        do {
                            try await Task.sleep(nanoseconds: 30_000_000_000)
                        } catch is CancellationError {
                            generationCancelled.set()
                        }
                    }

                    try await group.next()
                    group.cancelAll()
                }
            } catch is Disconnected {
                // Expected: keepalive detected client gone
            } catch {}

            handlerExited.set()
        }
        let port = try await startOnRandomPort(server)

        // Connect and read the initial chunk, then disconnect by stopping server
        let readTask = Task {
            let (bytes, _) = try await URLSession.shared.bytes(
                from: URL(string: "http://127.0.0.1:\(port)/prefill-disconnect")!
            )
            for try await line in bytes.lines {
                if line.hasPrefix("data: {") { break }
            }
        }

        try? await readTask.value
        server.stop()

        // The keepalive should detect disconnect within ~200ms, cancel generation
        try await Task.sleep(nanoseconds: 500_000_000)
        #expect(generationCancelled.isSet)
        #expect(handlerExited.isSet)
    }

    @Test func midStreamDisconnectBreaksGenerationLoop() async throws {
        // Verify that failed sse.send() breaks the labeled generation loop,
        // not just the switch statement.
        let loopExited = LeaseAcquiredSignal()

        let server = HTTPServer(port: 0)
        server.route(.GET, "/midstream-disconnect") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()

            generation: for i in 1...500 {
                guard await sse.send(["n": "\(i)"] as [String: String]) else {
                    break generation
                }
                try? await Task.sleep(nanoseconds: 10_000_000)
                if Task.isCancelled { break generation }
            }
            loopExited.set()
        }
        let port = try await startOnRandomPort(server)

        // Read a few chunks then disconnect
        let readTask = Task {
            let (bytes, _) = try await URLSession.shared.bytes(
                from: URL(string: "http://127.0.0.1:\(port)/midstream-disconnect")!
            )
            var count = 0
            for try await line in bytes.lines {
                if line.hasPrefix("data: {") { count += 1 }
                if count >= 3 { break }
            }
        }

        try? await readTask.value
        server.stop()

        try await Task.sleep(nanoseconds: 500_000_000)
        // Loop must have exited — not still running 500 iterations
        #expect(loopExited.isSet)
    }

    // MARK: - Helpers

    /// Start server on a random available port, return the actual port.
    private func startOnRandomPort(_ server: HTTPServer) async throws -> UInt16 {
        // Port 0 isn't supported by NWListener, so find a free port
        let port = try findFreePort()
        await server.updatePort(port)
        await server.start()
        // Brief pause for listener to become ready
        try await Task.sleep(nanoseconds: 100_000_000)
        return port
    }

    private func findFreePort() throws -> UInt16 {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw PortError() }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = INADDR_LOOPBACK.bigEndian

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else { throw PortError() }

        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &len)
            }
        }
        guard nameResult == 0 else { throw PortError() }

        return UInt16(bigEndian: addr.sin_port)
    }
}

private struct PortError: Error {}
