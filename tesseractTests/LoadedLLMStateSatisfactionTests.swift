import Testing

@testable import Tesseract_Agent

/// The ADR-0008 satisfaction rule on the `.llm` slot's loaded state: loads
/// upgrade, never downgrade. A loaded vision container serves text-only
/// demands for the same model, so chat (toggle off) and HTTP callers sharing
/// the slot cannot thrash reloads — and the warm prefix cache survives.
struct LoadedLLMStateSatisfactionTests {

    private typealias State = InferenceArbiter.LoadedLLMState

    @Test func exactMatchSatisfies() {
        let text = State(modelID: "m", visionMode: false)
        let vision = State(modelID: "m", visionMode: true)

        #expect(text.satisfies(text))
        #expect(vision.satisfies(vision))
    }

    @Test func loadedVisionSatisfiesTextDemandForTheSameModel() {
        let loaded = State(modelID: "m", visionMode: true)
        let desired = State(modelID: "m", visionMode: false)

        #expect(loaded.satisfies(desired))
    }

    @Test func loadedTextDoesNotSatisfyVisionDemand() {
        let loaded = State(modelID: "m", visionMode: false)
        let desired = State(modelID: "m", visionMode: true)

        #expect(!loaded.satisfies(desired))
    }

    @Test func differentModelNeverSatisfies() {
        let loaded = State(modelID: "m", visionMode: true)

        #expect(!loaded.satisfies(State(modelID: "other", visionMode: false)))
        #expect(!loaded.satisfies(State(modelID: "other", visionMode: true)))
    }
}
