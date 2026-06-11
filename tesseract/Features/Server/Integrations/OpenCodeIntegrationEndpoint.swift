//
//  OpenCodeIntegrationEndpoint.swift
//  tesseract
//

import Foundation

/// HTTP response factories for the OpenCode Integration routes — pure
/// (snapshot in, `HTTPResponse` out) so the route closures in the composition
/// root stay one-line glue and the wire contract is testable without a
/// listener.
@MainActor
enum OpenCodeIntegrationEndpoint {

    /// `GET /integrations/opencode/setup.sh`
    static func setupScriptResponse(snapshot: IntegrationSnapshot) -> HTTPResponse {
        HTTPResponse(
            statusCode: 200,
            statusText: "OK",
            headers: [("Content-Type", "text/x-shellscript; charset=utf-8")],
            body: Data(OpenCodeSetupScript.render(snapshot: snapshot).utf8)
        )
    }

    /// `POST /integrations/opencode/merge` — body: the client's existing
    /// config bytes (or empty for none); response: the complete new file
    /// content. Pure of server-side effects; the script owns backup + write.
    static func mergeResponse(
        existingConfig: Data?,
        snapshot: IntegrationSnapshot
    ) -> HTTPResponse {
        let output = OpenCodeConfigMerge.merge(
            existingConfig: existingConfig,
            snapshot: snapshot
        )
        var headers = [("Content-Type", "application/json")]
        if output.replacedCorruptInput {
            headers.append((OpenCodeSetupScript.configWarningHeader, "existing-config-unparseable"))
        }
        return HTTPResponse(
            statusCode: 200,
            statusText: "OK",
            headers: headers,
            body: output.configData
        )
    }
}
