import Foundation

// MARK: - SkillExecution

/// The **Skill Execution** leaf (ADR-0045 continuation, #408): the chat's
/// collaborator for firing a **Skill**. It owns what a fire *is* — assembling
/// the argument text from the drained composer draft, rendering the **Skill
/// Envelope** injection block (#401) around the skill body, and recording the
/// user-initiated invocation for the **Skill Usage Ranking** — while the send
/// itself stays on the Chat Session spine.
///
/// A fire returns the ``Injection`` the spine sends; the leaf touches no
/// `Agent`, no arbiter, and never the error banner (load failure is a nil
/// render the spine surfaces). Held by the Chat Session — the default no-op
/// leaf in tests, the container-wired one (assembly + recording on a
/// `SkillPillController`) in production.
@MainActor
final class SkillExecution {

    /// A rendered skill injection ready for the spine to send: the
    /// `<skill>`-wrapped body with assembled arguments appended, plus the
    /// images that ride the fire.
    struct Injection: Equatable {
        let message: String
        let images: [ImageAttachment]
    }

    /// Argument assembly, provided by the view-owned **Skill Pill** leaf
    /// (identity in tests): the drained composer text becomes the argument
    /// text appended after the injected block (`translate`'s configured target
    /// language is the one wired special case).
    private let assembleArguments: @MainActor (_ skillName: String, _ userText: String) -> String
    /// Usage recording, provided by the same leaf (no-op in tests).
    private let recordInvocation: @MainActor (_ skillName: String) -> Void
    /// The skill-file read (the registry read): file bytes → body content with
    /// frontmatter stripped, nil when the file can't be loaded. Injected so the
    /// render is testable without a real file on disk.
    private let loadSkillBody: @MainActor (_ filePath: String) -> String?

    init(
        assembleArguments: @MainActor @escaping (String, String) -> String = { _, text in text },
        recordInvocation: @MainActor @escaping (String) -> Void = { _ in },
        loadSkillBody: @MainActor @escaping (String) -> String? = SkillExecution.readSkillBody
    ) {
        self.assembleArguments = assembleArguments
        self.recordInvocation = recordInvocation
        self.loadSkillBody = loadSkillBody
    }

    /// Render the injection for a fired skill: load the body, wrap it in the
    /// **Skill Envelope** injection, and append the assembled argument text
    /// outside the block. Returns nil when the skill file can't be read — the
    /// spine surfaces the load error and restores the draft. Byte-for-byte the
    /// message the pre-#408 inline `executeSkill` built.
    func render(
        skillName: String, filePath: String, userText: String, images: [ImageAttachment]
    ) -> Injection? {
        guard let body = loadSkillBody(filePath) else { return nil }
        var message = SkillEnvelope.injection(name: skillName, location: filePath, body: body)
        let arguments = assembleArguments(skillName, userText)
        if !arguments.isEmpty {
            message += "\n\n\(arguments)"
        }
        return Injection(message: message, images: images)
    }

    /// Record a user-initiated invocation for the ranking. The spine calls this
    /// once the rendered injection has been sent (never on a failed load).
    func recordFired(_ skillName: String) {
        recordInvocation(skillName)
    }

    /// The default skill-file read: bytes → ``SkillRegistry/bodyContent(of:)``
    /// (YAML frontmatter stripped), nil when the file can't be loaded.
    static func readSkillBody(_ filePath: String) -> String? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)),
            let fullText = String(data: data, encoding: .utf8)
        else {
            return nil
        }
        return SkillRegistry.bodyContent(of: fullText)
    }
}
