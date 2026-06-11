import Foundation

/// Shared image fixtures for the conversation-shape tests. The cross-adapter
/// parity suites (`MessageConverterTests`, `AgentConversationBuilderTests`)
/// must exercise both edges with the SAME bytes — a divergent fixture would
/// not fail loudly, it would just silently test different payloads.
enum ImageTestFixtures {
    /// 1×1 valid PNG.
    static let tinyPNGBase64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

    static var tinyPNGData: Data { Data(base64Encoded: tinyPNGBase64)! }
}
