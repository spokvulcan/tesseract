//
//  BrowserMCPTelemetryRecorder.swift
//  tesseract
//
//  Builds and persists Browser MCP telemetry events (ADR-0031). Sits at
//  the server's protocol choke point, so it sees BOTH entry paths — the
//  in-app agent (in-process transport) and external HTTP clients
//  (OpenCode, Claude Code) — with the origin and `clientInfo` identity
//  that distinguish them. Owns the per-session registry (who connected,
//  how many calls) and the result-shape measurement, including
//  screenshot pixel dimensions.
//
//  Persistence is `BrowserMCPTelemetryLog` (durable JSONL, Application
//  Support); each event also emits one `Log.browser` line so live
//  behavior is visible in `log stream` without opening the file.
//

import Foundation
import ImageIO
import MLXLMCommon

@MainActor
final class BrowserMCPTelemetryRecorder {

    /// Per-string cap inside recorded arguments — long `evaluate`
    /// scripts and `type` text are truncated, never dropped; the
    /// original encoded size is preserved in `argumentsBytes`.
    nonisolated static let maxArgumentChars = 2_000
    /// Cap on the recorded result-text preview; full length is
    /// preserved in `resultTextChars`.
    nonisolated static let maxPreviewChars = 2_000
    nonisolated static let maxErrorChars = 500

    private struct SessionInfo {
        let origin: MCPClientOrigin
        var clientName: String?
        var clientVersion: String?
        var toolCalls: Int = 0
    }

    private let log: BrowserMCPTelemetryLog
    private let isEnabled: @MainActor @Sendable () -> Bool
    private var sessions: [String: SessionInfo] = [:]

    init(
        log: BrowserMCPTelemetryLog = BrowserMCPTelemetryLog(),
        isEnabled: @escaping @MainActor @Sendable () -> Bool = { true }
    ) {
        self.log = log
        self.isEnabled = isEnabled
    }

    /// Where the JSONL corpus lives — surfaced so UI can offer
    /// "Open Telemetry Folder" next to the request-log affordance.
    var directoryURL: URL { log.directory }

    // MARK: - Session lifecycle

    /// Record an `initialize` and register the session so later tool
    /// calls inherit the client's identity. `params` is the request's
    /// `params` object; `clientInfo.name/version` are lifted per spec.
    func recordSessionStart(
        sessionID: String, origin: MCPClientOrigin, params: [String: JSONValue]?
    ) {
        let clientInfo = params?["clientInfo"]?.asObject
        let name = clientInfo?["name"]?.asString
        let version = clientInfo?["version"]?.asString
        sessions[sessionID] = SessionInfo(
            origin: origin, clientName: name, clientVersion: version)
        guard isEnabled() else { return }
        append(
            BrowserMCPTelemetryEvent(
                kind: .sessionStart,
                timestamp: now(),
                sessionID: sessionID,
                origin: origin,
                clientName: name,
                clientVersion: version
            ))
        Log.browser.info(
            "mcp session start \(short(sessionID)) origin=\(origin.rawValue) "
                + "client=\(name ?? "?")/\(version ?? "?")")
    }

    func recordSessionEnd(sessionID: String, origin: MCPClientOrigin) {
        let info = sessions.removeValue(forKey: sessionID)
        guard isEnabled() else { return }
        append(
            BrowserMCPTelemetryEvent(
                kind: .sessionEnd,
                timestamp: now(),
                sessionID: sessionID,
                origin: info?.origin ?? origin,
                clientName: info?.clientName,
                clientVersion: info?.clientVersion,
                seq: info?.toolCalls
            ))
        Log.browser.info("mcp session end \(short(sessionID)) calls=\(info?.toolCalls ?? 0)")
    }

    /// All sessions closed at once (server stop / app termination).
    func recordServerShutdown() {
        let open = sessions
        sessions.removeAll()
        guard isEnabled() else { return }
        for (sessionID, info) in open {
            append(
                BrowserMCPTelemetryEvent(
                    kind: .serverShutdown,
                    timestamp: now(),
                    sessionID: sessionID,
                    origin: info.origin,
                    clientName: info.clientName,
                    clientVersion: info.clientVersion,
                    seq: info.toolCalls
                ))
        }
    }

    // MARK: - Methods

    func recordToolsList(sessionID: String?, origin: MCPClientOrigin) {
        guard isEnabled() else { return }
        let info = sessionID.flatMap { sessions[$0] }
        append(
            BrowserMCPTelemetryEvent(
                kind: .toolsList,
                timestamp: now(),
                sessionID: sessionID,
                origin: info?.origin ?? origin,
                clientName: info?.clientName,
                clientVersion: info?.clientVersion,
                method: "tools/list"
            ))
    }

    /// Record one executed `tools/call` — the analysis workhorse.
    func recordToolCall(
        sessionID: String,
        origin: MCPClientOrigin,
        tool: String,
        arguments: [String: JSONValue],
        result: BrowserToolResult,
        duration: Duration
    ) {
        var seq: Int?
        if var info = sessions[sessionID] {
            info.toolCalls += 1
            sessions[sessionID] = info
            seq = info.toolCalls
        }
        guard isEnabled() else { return }
        let info = sessions[sessionID]

        let originalBytes = encodedSize(.object(arguments))
        let (capped, truncated) = Self.capped(.object(arguments))
        let shape = Self.measure(result)
        let durationMS = Self.milliseconds(duration)

        append(
            BrowserMCPTelemetryEvent(
                kind: .toolCall,
                timestamp: now(),
                sessionID: sessionID,
                origin: info?.origin ?? origin,
                clientName: info?.clientName,
                clientVersion: info?.clientVersion,
                seq: seq,
                tool: tool,
                arguments: capped,
                argumentsBytes: originalBytes,
                argumentsTruncated: truncated ? true : nil,
                durationMS: durationMS,
                outcome: result.isError ? .error : .ok,
                errorMessage: result.isError ? shape.errorMessage : nil,
                resultTextChars: shape.textChars,
                resultTextPreview: shape.preview,
                resultTextTruncated: shape.previewTruncated ? true : nil,
                resultImages: shape.images.isEmpty ? nil : shape.images,
                resultBytes: shape.totalBytes
            ))

        let images = shape.images.map { "\($0.width ?? 0)x\($0.height ?? 0)" }
            .joined(separator: ",")
        Log.browser.info(
            "mcp tool=\(tool) origin=\((info?.origin ?? origin).rawValue) "
                + "session=\(short(sessionID)) \(Int(durationMS))ms "
                + "\(result.isError ? "error" : "ok") text=\(shape.textChars)ch"
                + (images.isEmpty ? "" : " images=\(images)"))
    }

    func recordProtocolError(
        method: String?, code: Int, message: String,
        sessionID: String?, origin: MCPClientOrigin
    ) {
        guard isEnabled() else { return }
        let info = sessionID.flatMap { sessions[$0] }
        append(
            BrowserMCPTelemetryEvent(
                kind: .protocolError,
                timestamp: now(),
                sessionID: sessionID,
                origin: info?.origin ?? origin,
                clientName: info?.clientName,
                clientVersion: info?.clientVersion,
                errorMessage: String(message.prefix(Self.maxErrorChars)),
                method: method,
                errorCode: code
            ))
        Log.browser.warning(
            "mcp protocol error code=\(code) method=\(method ?? "?") "
                + "origin=\(origin.rawValue): \(message)")
    }

    /// Test barrier: returns after every recorded event is on disk.
    func flushForTesting() {
        log.flushForTesting()
    }

    // MARK: - Result measurement

    private struct ResultShape {
        var textChars = 0
        var preview = ""
        var previewTruncated = false
        var images: [BrowserMCPTelemetryEvent.ImageInfo] = []
        var totalBytes = 0
        var errorMessage: String?
    }

    private static func measure(_ result: BrowserToolResult) -> ResultShape {
        var shape = ResultShape()
        for block in result.content {
            switch block {
            case .text(let text):
                shape.textChars += text.count
                shape.totalBytes += text.utf8.count
                if !shape.preview.isEmpty { shape.preview += "\n" }
                shape.preview += text
            case .image(let data, let mimeType):
                let dimensions = imageDimensions(data)
                shape.images.append(
                    BrowserMCPTelemetryEvent.ImageInfo(
                        width: dimensions?.width,
                        height: dimensions?.height,
                        bytes: data.count,
                        mimeType: mimeType
                    ))
                shape.totalBytes += data.count
            }
        }
        if result.isError {
            shape.errorMessage = String(shape.preview.prefix(maxErrorChars))
        }
        if shape.preview.count > maxPreviewChars {
            shape.preview = String(shape.preview.prefix(maxPreviewChars))
            shape.previewTruncated = true
        }
        return shape
    }

    /// Pixel dimensions without decoding the bitmap — ImageIO reads
    /// them from the header.
    nonisolated static func imageDimensions(_ data: Data) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        return (width, height)
    }

    // MARK: - Argument capping

    /// Cap every string value inside a JSON tree at `maxArgumentChars`
    /// so one huge `evaluate` script can't bloat the corpus. Returns
    /// whether anything was cut.
    nonisolated static func capped(_ value: JSONValue) -> (value: JSONValue, truncated: Bool) {
        switch value {
        case .string(let string):
            guard string.count > maxArgumentChars else { return (value, false) }
            return (.string(String(string.prefix(maxArgumentChars)) + "…[truncated]"), true)
        case .array(let items):
            var truncated = false
            let capped = items.map { item -> JSONValue in
                let (value, cut) = Self.capped(item)
                truncated = truncated || cut
                return value
            }
            return (.array(capped), truncated)
        case .object(let object):
            var truncated = false
            var capped: [String: JSONValue] = [:]
            for (key, item) in object {
                let (value, cut) = Self.capped(item)
                truncated = truncated || cut
                capped[key] = value
            }
            return (.object(capped), truncated)
        case .null, .bool, .int, .double:
            return (value, false)
        }
    }

    // MARK: - Helpers

    private func append(_ event: BrowserMCPTelemetryEvent) {
        log.append(event)
    }

    private func now() -> Double {
        Date().timeIntervalSince1970
    }

    private func encodedSize(_ value: JSONValue) -> Int {
        (try? JSONEncoder().encode(value))?.count ?? 0
    }

    private func short(_ sessionID: String) -> String {
        String(sessionID.prefix(8))
    }

    nonisolated static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1_000
            + Double(duration.components.attoseconds) / 1e15
    }
}
