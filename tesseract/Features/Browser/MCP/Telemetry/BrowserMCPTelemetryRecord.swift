//
//  BrowserMCPTelemetryRecord.swift
//  tesseract
//
//  The on-disk schema for Browser MCP tool telemetry (ADR-0031): one
//  JSONL event per MCP interaction — session lifecycle, every tool
//  call with its arguments, latency, outcome, and result shape
//  (including screenshot pixel dimensions), and protocol-level errors.
//  Field semantics follow the OpenTelemetry GenAI `execute_tool`
//  conventions (tool name, arguments, error type, session id) so the
//  corpus stays convertible to OTLP later; the encoding is the repo's
//  own discriminated-line JSONL (`CompletionTraceLine` precedent), not
//  the OTLP envelope.
//
//  Everything here is local-only product telemetry: it never leaves
//  the Mac, and exists so real agent transcripts can be analyzed
//  offline to improve the tools (fewer calls, clearer results, right
//  screenshot resolution).
//

import Foundation
import MLXLMCommon

// MARK: - MCPClientOrigin

/// Which entry path a request arrived through — the load-bearing
/// dimension for analysis, since the same server serves both:
/// - `inProcess`: Tesseract's own agent over `InProcessMCPTransport`
///   (ADR-0027 dogfood; no socket).
/// - `http`: an external client (OpenCode, Claude Code, …) over the
///   loopback `/mcp` listener.
nonisolated enum MCPClientOrigin: String, Codable, Sendable {
    case inProcess = "in_process"
    case http
}

// MARK: - Header

/// First line of every telemetry file: the schema contract readers
/// gate on before analyzing.
nonisolated struct BrowserMCPTelemetryHeader: Codable, Sendable, Equatable {
    let schemaVersion: Int
    let serverName: String
    let serverVersion: String
    /// Seconds since 1970 when the file was opened.
    let createdAt: Double
}

// MARK: - Event

/// One telemetry event. A single wide record discriminated by `kind`;
/// optional fields are omitted from the JSON when nil, so each kind
/// serializes only its own section.
nonisolated struct BrowserMCPTelemetryEvent: Codable, Sendable, Equatable {
    static let currentSchemaVersion = 1

    nonisolated enum Kind: String, Codable, Sendable {
        /// `initialize` handled — a Browser Session was minted.
        case sessionStart = "session_start"
        /// `tools/list` served.
        case toolsList = "tools_list"
        /// One `tools/call` executed (the analysis workhorse).
        case toolCall = "tool_call"
        /// A JSON-RPC error was returned (parse, unknown method,
        /// missing session, bad params …).
        case protocolError = "protocol_error"
        /// A session was closed by the client (`DELETE`).
        case sessionEnd = "session_end"
        /// All sessions were closed (server stop / app termination).
        case serverShutdown = "server_shutdown"
    }

    nonisolated enum Outcome: String, Codable, Sendable {
        case ok
        /// The tool returned `isError: true` (tool-level failure —
        /// timeouts included; see `errorMessage`).
        case error
    }

    /// Pixel-level shape of one image content block — the evidence for
    /// "are our screenshots too small for the model?".
    nonisolated struct ImageInfo: Codable, Sendable, Equatable {
        let width: Int?
        let height: Int?
        /// Encoded (pre-base64) byte count.
        let bytes: Int
        let mimeType: String
    }

    let kind: Kind
    /// Seconds since 1970 (event completion time).
    let timestamp: Double

    // Identity / correlation
    let sessionID: String?
    let origin: MCPClientOrigin?
    /// From `initialize` `clientInfo` — `"tesseract-agent"` for the
    /// in-app agent, the client's own name (e.g. `"opencode"`) over HTTP.
    let clientName: String?
    let clientVersion: String?
    /// Per-session tool-call ordinal (1-based), so call sequences can
    /// be reconstructed per client session.
    let seq: Int?

    // Tool call
    let tool: String?
    /// The call's arguments, verbatim except long string values capped
    /// at `BrowserMCPTelemetryRecorder.maxArgumentChars`.
    let arguments: JSONValue?
    /// Encoded size of the *original* (uncapped) arguments.
    let argumentsBytes: Int?
    let argumentsTruncated: Bool?
    let durationMS: Double?
    let outcome: Outcome?
    let errorMessage: String?

    // Result shape
    let resultTextChars: Int?
    let resultTextPreview: String?
    let resultTextTruncated: Bool?
    let resultImages: [ImageInfo]?
    /// Total content payload (text UTF-8 + image bytes, pre-base64).
    let resultBytes: Int?

    // Protocol errors / non-call methods
    let method: String?
    let errorCode: Int?

    init(
        kind: Kind,
        timestamp: Double,
        sessionID: String? = nil,
        origin: MCPClientOrigin? = nil,
        clientName: String? = nil,
        clientVersion: String? = nil,
        seq: Int? = nil,
        tool: String? = nil,
        arguments: JSONValue? = nil,
        argumentsBytes: Int? = nil,
        argumentsTruncated: Bool? = nil,
        durationMS: Double? = nil,
        outcome: Outcome? = nil,
        errorMessage: String? = nil,
        resultTextChars: Int? = nil,
        resultTextPreview: String? = nil,
        resultTextTruncated: Bool? = nil,
        resultImages: [ImageInfo]? = nil,
        resultBytes: Int? = nil,
        method: String? = nil,
        errorCode: Int? = nil
    ) {
        self.kind = kind
        self.timestamp = timestamp
        self.sessionID = sessionID
        self.origin = origin
        self.clientName = clientName
        self.clientVersion = clientVersion
        self.seq = seq
        self.tool = tool
        self.arguments = arguments
        self.argumentsBytes = argumentsBytes
        self.argumentsTruncated = argumentsTruncated
        self.durationMS = durationMS
        self.outcome = outcome
        self.errorMessage = errorMessage
        self.resultTextChars = resultTextChars
        self.resultTextPreview = resultTextPreview
        self.resultTextTruncated = resultTextTruncated
        self.resultImages = resultImages
        self.resultBytes = resultBytes
        self.method = method
        self.errorCode = errorCode
    }
}

// MARK: - Line

/// One JSONL line, discriminated by `kind` so the format can grow new
/// line kinds without breaking old readers (`CompletionTraceLine`
/// precedent).
nonisolated enum BrowserMCPTelemetryLine: Codable, Sendable, Equatable {
    case header(BrowserMCPTelemetryHeader)
    case event(BrowserMCPTelemetryEvent)

    private enum CodingKeys: String, CodingKey {
        case kind
        case header
        case event
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(String.self, forKey: .kind)
        switch kind {
        case "header":
            self = .header(try container.decode(BrowserMCPTelemetryHeader.self, forKey: .header))
        case "event":
            self = .event(try container.decode(BrowserMCPTelemetryEvent.self, forKey: .event))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .kind,
                in: container,
                debugDescription: "unknown telemetry line kind '\(kind)'"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .header(let header):
            try container.encode("header", forKey: .kind)
            try container.encode(header, forKey: .header)
        case .event(let event):
            try container.encode("event", forKey: .kind)
            try container.encode(event, forKey: .event)
        }
    }
}
