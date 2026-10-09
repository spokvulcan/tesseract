import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The catalog gate (ADR-0070): every chat model the catalog serves must
/// measure its **Generation Prompt**, by default and with thinking off, so
/// no shipped model reaches the unknown state. Runs against the models
/// downloaded under `~/Library/Application Support/models` (override with
/// `TESSERACT_MODELS_ROOT`), loaded through the production tokenizer loader.
/// A gate that measured nothing has not passed, so the suite is skipped
/// where no catalog chat model is downloaded. The directory alone doesn't
/// count: the app creates it at launch, so a CI runner has one with nothing
/// in it. It prints which models it checked and which it skipped.
@Suite(.cpuBound)
struct GenerationPromptCatalogRealTests {

    nonisolated static var modelsRoot: URL {
        URL(
            fileURLWithPath: NSString(
                string: ProcessInfo.processInfo.environment["TESSERACT_MODELS_ROOT"]
                    ?? "~/Library/Application Support/models"
            ).expandingTildeInPath)
    }

    /// The catalog entries a chat template drives.
    @MainActor
    static var chatModels: [ModelDefinition] {
        ModelDefinition.all.filter { $0.category == .agent || $0.category == .proofread }
    }

    nonisolated static func hasTokenizer(_ directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tokenizer_config.json").path)
    }

    @MainActor
    static var anyChatModelDownloaded: Bool {
        chatModels.contains { model in
            model.cacheSubdirectory.map { hasTokenizer(modelsRoot.appendingPathComponent($0)) }
                ?? false
        }
    }

    /// The request context for `enable_thinking`, resolved as the completion
    /// handler resolves it against the model's own template.
    static func context(enableThinking: Bool, identity: ModelIdentity) -> TemplateRenderContext {
        TemplateRenderContext.resolve(
            requestKwargs: [TemplateRenderFlag.enableThinking.rawValue: enableThinking],
            appDesired: [:],
            declaredFlags: identity.declaredTemplateFlags,
            templateDefaults: identity.templateFlagDefaults)
    }

    /// What a family's template is known to append. Only the templates
    /// checked against their own tokenizer get a think-block expectation;
    /// every other catalog model must still measure.
    enum Family {
        /// Qwen3.5-0.8B: a closed empty block unless thinking is asked for.
        case thinksWhenAsked
        /// Qwen3.5, Qwen3.8 and Bonsai 2 (the Qwen3.8 template): an open
        /// block, a closed empty one with thinking off.
        case thinksByDefault
        case measuresOnly
    }

    @MainActor
    static func family(of id: String) -> Family {
        if id == ModelDefinition.defaultProofreadModelID { return .thinksWhenAsked }
        if ["qwen3.5-", "qwen3.8-", "bonsai-2-"].contains(where: id.hasPrefix) {
            return .thinksByDefault
        }
        return .measuresOnly
    }

    /// What the gate reads of a catalog chat model, taken on the main actor
    /// where the catalog lives, so the loads and renders can run off it.
    struct Entry: Sendable {
        let id: String
        let subdirectory: String?
        let family: Family
    }

    @MainActor
    static var entries: [Entry] {
        chatModels.map {
            Entry(id: $0.id, subdirectory: $0.cacheSubdirectory, family: family(of: $0.id))
        }
    }

    static func measured(
        _ tokenizer: any Tokenizer, _ context: TemplateRenderContext
    ) throws -> GenerationPrompt {
        let fed = try tokenizer.applyChatTemplate(
            messages: GenerationPromptProbeTests.conversation, tools: nil,
            additionalContext: context.additionalContext())
        return ConversationRender.generationPromptProbe(
            tokenizer: tokenizer, renderContext: context, modelFingerprint: nil,
            cache: RenderTokenCache()
        ).checked(against: fed)
    }

    /// Off the main actor: each model's tokenizer load and renders take a
    /// second or more in a Debug build, and in a parallel run the main actor
    /// is every other suite's too. The models are checked three at a time;
    /// one after another, the loads made this the slowest test in the target.
    @concurrent
    @Test(.enabled("no catalog chat model is downloaded") { await anyChatModelDownloaded })
    func everyDownloadedCatalogModelMeasures() async throws {
        var downloaded: [(model: Entry, directory: URL)] = []
        var skipped: [String] = []
        for model in await Self.entries {
            guard let subdirectory = model.subdirectory else { continue }
            let directory = Self.modelsRoot.appendingPathComponent(subdirectory)
            guard Self.hasTokenizer(directory) else {
                skipped.append(model.id)
                continue
            }
            downloaded.append((model, directory))
        }
        let checked = try await withThrowingTaskGroup(of: String.self) { group in
            var checked: [String] = []
            var next = 0
            func startNext() {
                guard next < downloaded.count else { return }
                let (model, directory) = downloaded[next]
                next += 1
                group.addTask { try await Self.check(model, in: directory) }
            }
            for _ in 0..<3 { startNext() }
            while let id = try await group.next() {
                checked.append(id)
                startNext()
            }
            return checked
        }
        print(
            "generation-prompt catalog gate: checked=\(checked.sorted().joined(separator: ",")) "
                + "skipped=\(skipped.joined(separator: ","))")
        #expect(!checked.isEmpty, "no catalog chat model is downloaded; the gate checked nothing")
    }

    /// One model's measurements, by default and with thinking off (and on,
    /// for the family that thinks only when asked). Returns the model's ID.
    @concurrent
    static func check(_ model: Entry, in directory: URL) async throws -> String {
        let tokenizer = try await AppTokenizerLoader().load(from: directory)
        let identity = ModelIdentity(directory: directory)
        let byDefault = try Self.measured(tokenizer, .canonical)
        let off = try Self.measured(
            tokenizer, Self.context(enableThinking: false, identity: identity))
        #expect(byDefault.unknownReason == nil, "\(model.id) default: \(byDefault.traceValue)")
        #expect(off.unknownReason == nil, "\(model.id) thinking off: \(off.traceValue)")
        switch model.family {
        case .thinksWhenAsked:
            #expect(byDefault.thinkBlock == .closed, "\(model.id): \(byDefault.traceValue)")
            #expect(off.thinkBlock == .closed, "\(model.id) thinking off: \(off.traceValue)")
            let on = try Self.measured(
                tokenizer, Self.context(enableThinking: true, identity: identity))
            #expect(on.thinkBlock == .opens, "\(model.id) thinking on: \(on.traceValue)")
        case .thinksByDefault:
            #expect(byDefault.thinkBlock == .opens, "\(model.id): \(byDefault.traceValue)")
            #expect(off.thinkBlock == .closed, "\(model.id) thinking off: \(off.traceValue)")
        case .measuresOnly:
            break
        }
        return model.id
    }
}
