import Foundation
import Testing

@testable import Tesseract_Agent

/// Wire contract of the OpenCode Integration routes, tested at the response
/// factory — status, content type, warning header, body — without a listener,
/// the same altitude as `CompletionHandlerTests`.
@MainActor
struct OpenCodeIntegrationEndpointTests {

    @Test func setupScriptIsServedAsShellScript() {
        let response = OpenCodeIntegrationEndpoint.setupScriptResponse(snapshot: snapshot())

        #expect(response.statusCode == 200)
        #expect(header(response, "Content-Type") == "text/x-shellscript; charset=utf-8")
        // Lossy UTF-8 decode is intentional here.
        // swiftlint:disable:next optional_data_string_conversion
        let body = String(decoding: response.body ?? Data(), as: UTF8.self)
        #expect(body.hasPrefix("#!/bin/sh"))
    }

    @Test func mergeReturnsJSONWithoutWarningForValidInput() throws {
        let existing = Data(#"{ "model": "omlx/some-model" }"#.utf8)

        let response = OpenCodeIntegrationEndpoint.mergeResponse(
            existingConfig: existing,
            snapshot: snapshot()
        )

        #expect(response.statusCode == 200)
        #expect(header(response, "Content-Type") == "application/json")
        #expect(header(response, "X-Tesseract-Config-Warning") == nil)
        let root = try #require(
            JSONSerialization.jsonObject(with: response.body ?? Data()) as? [String: Any]
        )
        #expect((root["provider"] as? [String: Any])?["tesseract"] != nil)
    }

    @Test func mergeFlagsCorruptInputViaWarningHeader() {
        let response = OpenCodeIntegrationEndpoint.mergeResponse(
            existingConfig: Data("{ not json".utf8),
            snapshot: snapshot()
        )

        #expect(response.statusCode == 200)
        #expect(header(response, "X-Tesseract-Config-Warning") == "existing-config-unparseable")
    }

    @Test func mergeTreatsAbsentBodyAsFreshConfig() throws {
        let response = OpenCodeIntegrationEndpoint.mergeResponse(
            existingConfig: nil,
            snapshot: snapshot()
        )

        #expect(header(response, "X-Tesseract-Config-Warning") == nil)
        let root = try #require(
            JSONSerialization.jsonObject(with: response.body ?? Data()) as? [String: Any]
        )
        #expect(root["$schema"] as? String == "https://opencode.ai/config.json")
    }

    // MARK: - Fixtures

    private func snapshot() -> IntegrationSnapshot {
        IntegrationSnapshot(
            port: 8321,
            models: [
                IntegrationSnapshot.Model(
                    id: "qwen3.5-4b-paro",
                    displayName: "Qwen3.5-4B PARO",
                    visionCapable: true,
                    contextLength: 262_144
                )
            ],
            defaultModelID: "qwen3.5-4b-paro"
        )
    }

    private func header(_ response: HTTPResponse, _ name: String) -> String? {
        response.headers.first { $0.name.lowercased() == name.lowercased() }?.value
    }
}
