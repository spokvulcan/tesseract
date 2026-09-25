import Foundation
import MLXLMCommon

@testable import Tesseract_Agent

/// Requests for the suites that drive a Server Completion arm or the
/// Speculative Canonical Prefill directly. The request types can only be
/// built by Request Keying (ADR-0070), so these run it on the toy session
/// exactly as Server Completion does, and the arm under test reads facts
/// derived the one way production derives them.
nonisolated enum ToyRequestKeying {

    struct NotKeyedAsExpected: Error {}

    static func parameters(prefillStepSize: Int? = nil) -> GenerateParameters {
        var parameters = GenerateParameters(temperature: 0)
        parameters.prefill = PrefillParameters(stepSize: prefillStepSize)
        return parameters
    }

    /// The Keyed Request for `conversation` on `provider`'s toy session.
    static func keyedRequest(
        provider: ToyModelSessionProvider,
        conversation: HTTPPrefixCacheConversation,
        modelID: String = "toy/model",
        prefillStepSize: Int? = nil
    ) async throws -> KeyedRequest {
        try await provider.withSession { session in
            guard
                case .keyed(let request, _) = try await RequestKeyingPhase.run(
                    session: session, conversation: conversation, canonicalTools: nil,
                    renderContext: .canonical,
                    parameters: parameters(prefillStepSize: prefillStepSize), modelID: modelID,
                    modelFingerprint: nil, imageKeying: nil)
            else { throw NotKeyedAsExpected() }
            return request
        }
    }

    /// An Unkeyed Completion's request and prepared input, inside `session`:
    /// a one-message conversation carrying an image on a family with no
    /// image-placeholder identity, so key-space construction fails with
    /// `unrecognizedPlaceholderFamily`. A tokenizer that renders only text
    /// parts (`ToySequencingTokenizer`) prepares the text render alone.
    static func unkeyedRequest(
        in session: any ModelSession, userText: String
    ) async throws -> (request: UnkeyedRequest, input: LMInput) {
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(
                    role: .user, content: userText,
                    images: [HTTPPrefixCacheImage(data: ImageTestFixtures.tinyPNGData)])
            ])
        guard
            case .unkeyed(let request, let input) = try await RequestKeyingPhase.run(
                session: session, conversation: conversation, canonicalTools: nil,
                renderContext: .canonical, parameters: parameters(), modelID: "toy/model",
                modelFingerprint: nil, imageKeying: nil)
        else { throw NotKeyedAsExpected() }
        return (request, input)
    }
}
