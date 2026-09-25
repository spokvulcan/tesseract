import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The catalog gate (ADR-0070): every chat model the catalog serves must
/// measure its **Generation Prompt**, by default and with thinking off, so
/// no shipped model reaches the unknown state. Runs against the models
/// downloaded under `~/Library/Application Support/models` (override with
/// `TESSERACT_MODELS_ROOT`), loaded through the production tokenizer loader;
/// the suite is skipped where that directory does not exist. It prints which
/// models it checked and which it skipped, and a run that checked none
/// fails: a gate that measured nothing has not passed.
struct GenerationPromptCatalogRealTests {

    nonisolated static var modelsRoot: URL {
        URL(
            fileURLWithPath: NSString(
                string: ProcessInfo.processInfo.environment["TESSERACT_MODELS_ROOT"]
                    ?? "~/Library/Application Support/models"
            ).expandingTildeInPath)
    }

    nonisolated static var modelsRootExists: Bool {
        FileManager.default.fileExists(atPath: modelsRoot.path)
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

    @MainActor
    @Test(.enabled(if: modelsRootExists))
    func everyDownloadedCatalogModelMeasures() async throws {
        var checked: [String] = []
        var skipped: [String] = []
        // The catalog entries a chat template drives.
        let chatModels = ModelDefinition.all.filter {
            $0.category == .agent || $0.category == .proofread
        }
        for model in chatModels {
            guard let subdirectory = model.cacheSubdirectory else { continue }
            let directory = Self.modelsRoot.appendingPathComponent(subdirectory)
            guard
                FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent("tokenizer_config.json").path)
            else {
                skipped.append(model.id)
                continue
            }
            let tokenizer = try await AppTokenizerLoader().load(from: directory)
            let identity = ModelIdentity(directory: directory)
            let byDefault = try Self.measured(tokenizer, .canonical)
            let off = try Self.measured(
                tokenizer, Self.context(enableThinking: false, identity: identity))
            #expect(byDefault.unknownReason == nil, "\(model.id) default: \(byDefault.traceValue)")
            #expect(off.unknownReason == nil, "\(model.id) thinking off: \(off.traceValue)")
            switch Self.family(of: model.id) {
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
            checked.append(model.id)
        }
        print(
            "generation-prompt catalog gate: checked=\(checked.joined(separator: ",")) "
                + "skipped=\(skipped.joined(separator: ","))")
        #expect(!checked.isEmpty, "no catalog chat model is downloaded; the gate checked nothing")
    }
}
