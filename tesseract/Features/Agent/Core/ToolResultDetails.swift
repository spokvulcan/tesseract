import Foundation

// MARK: - ToolResultDetails

/// Typed details of a tool execution, carried on the persisted
/// `ToolResultMessage` (PRD #200). Tools that compute structured facts about
/// what they did (the edit tool's diff, the read tool's truncation) surface
/// them here so the transcript's Tool Panels can project from data instead of
/// re-parsing result text. Optional everywhere: legacy conversations and
/// tools without details render the generic panel.
///
/// Encodes as a single-key object (`{"edit": {…}}`). Decoding an unknown key
/// throws — `ToolResultMessage` catches that and degrades to `nil` details,
/// so a future case never costs the whole message.
nonisolated enum ToolResultDetails: Sendable, Codable, Hashable {
    case edit(EditToolDetails)
    case read(ReadToolDetails)
    case ls(LsToolDetails)

    private enum CodingKeys: String, CodingKey {
        case edit, read, ls
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let details = try container.decodeIfPresent(EditToolDetails.self, forKey: .edit) {
            self = .edit(details)
        } else if let details = try container.decodeIfPresent(ReadToolDetails.self, forKey: .read) {
            self = .read(details)
        } else if let details = try container.decodeIfPresent(LsToolDetails.self, forKey: .ls) {
            self = .ls(details)
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Unknown ToolResultDetails shape"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .edit(let details): try container.encode(details, forKey: .edit)
        case .read(let details): try container.encode(details, forKey: .read)
        case .ls(let details): try container.encode(details, forKey: .ls)
        }
    }

    // MARK: Case accessors

    var editDetails: EditToolDetails? {
        if case .edit(let details) = self { details } else { nil }
    }

    var readDetails: ReadToolDetails? {
        if case .read(let details) = self { details } else { nil }
    }

    var lsDetails: LsToolDetails? {
        if case .ls(let details) = self { details } else { nil }
    }
}
