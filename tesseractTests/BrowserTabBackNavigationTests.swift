import Foundation
import Testing
import WebKit

@testable import Tesseract_Agent

// MARK: - BrowserTabBackNavigationTests

/// `back` returns to the previous history entry promptly. Complements
/// ``BrowserTabNavigationTimeoutTests`` (which covers the *stall* path): a
/// well-behaved page must still emit `.finished` and land back on the prior
/// URL, so the timeout fix never regresses the common case into a 30s wait.
@MainActor
struct BrowserTabBackNavigationTests {

    private func serveTwoPages() async -> (HTTPServer, UInt16) {
        let server = HTTPServer(port: 0)
        func page(_ path: String, _ title: String) {
            server.route(.GET, path) { _, writer in
                try await writer.send(
                    HTTPResponse(
                        statusCode: 200, statusText: "OK",
                        headers: [("Content-Type", "text/html; charset=utf-8")],
                        body: Data(
                            "<!DOCTYPE html><html><head><title>\(title)</title></head>"
                                .appending("<body><h1>\(title)</h1></body></html>").utf8)))
            }
        }
        page("/a", "Page A")
        page("/b", "Page B")
        await server.start()
        let port = await ScriptedMCPServer.waitForPort(server)
        return (server, port)
    }

    @Test func backReturnsToPreviousPagePromptly() async throws {
        let (server, port) = await serveTwoPages()
        defer { server.stop() }

        let tab = BrowserTab(configuration: WebPage.Configuration())
        try await tab.navigate(to: URL(string: "http://127.0.0.1:\(port)/a")!)
        try await tab.navigate(to: URL(string: "http://127.0.0.1:\(port)/b")!)

        let start = ContinuousClock.now
        let status = try await tab.goBack()
        let elapsed = ContinuousClock.now - start

        #expect(status.url.hasSuffix("/a"))
        // Well under the 30s navigation budget — proves the fix left the happy
        // path fast (observed ~a few ms + hydration settle) rather than making
        // every back wait for the timeout.
        #expect(elapsed < .seconds(5))
    }
}
