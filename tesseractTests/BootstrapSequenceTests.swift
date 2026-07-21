//
//  BootstrapSequenceTests.swift
//  tesseractTests
//
//  Exercises the pure validator and the production launch declaration
//  (`DependencyContainer.bootstrapStepNames` / `bootstrapInvariants`) without
//  constructing a container or running any step — the step closures are never
//  called. Reordering `setup()` shows up here as a failing invariant that names
//  the reason.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct BootstrapSequenceTests {

    // MARK: - Helpers

    /// Build a sequence with no-op step closures — validation never calls them.
    private func sequence(
        steps names: [String],
        invariants: [BootstrapSequence.Invariant]
    ) -> BootstrapSequence {
        BootstrapSequence(
            steps: names.map { BootstrapSequence.Step(name: $0, run: {}) },
            invariants: invariants
        )
    }

    private func invariant(
        _ before: String, before after: String, why: String = "because"
    ) -> BootstrapSequence.Invariant {
        .init(before: before, after: after, why: why)
    }

    // MARK: - Validator rows

    @Test
    func satisfiedInvariantPasses() {
        let seq = sequence(
            steps: ["a", "b", "c"],
            invariants: [invariant("a", before: "c")]
        )
        #expect(seq.validate().isEmpty)
    }

    @Test
    func violatedOrderIsDetected() {
        let inv = invariant("b", before: "a")
        let seq = sequence(steps: ["a", "b"], invariants: [inv])
        #expect(seq.validate() == [.violatedOrder(inv)])
    }

    @Test
    func adjacentEqualIndexIsAViolation() {
        // A step can't run before itself; before == after is a violation, not a pass.
        let inv = invariant("a", before: "a")
        let seq = sequence(steps: ["a", "b"], invariants: [inv])
        #expect(seq.validate() == [.violatedOrder(inv)])
    }

    @Test
    func unknownStepNameInInvariantIsDetected() {
        let inv = invariant("ghost", before: "a")
        let seq = sequence(steps: ["a", "b"], invariants: [inv])
        #expect(seq.validate() == [.unknownStep(name: "ghost", in: inv)])
    }

    @Test
    func duplicateStepNamesAreDetected() {
        let seq = sequence(steps: ["a", "a", "b"], invariants: [])
        #expect(seq.validate() == [.duplicateStepName("a")])
    }

    @Test
    func multipleFailuresAllReported() {
        let dup = "a"
        let unknown = invariant("nope", before: "b")
        let seq = sequence(steps: ["a", "a", "b"], invariants: [unknown])
        let failures = seq.validate()
        #expect(failures.contains(.duplicateStepName(dup)))
        #expect(failures.contains(.unknownStep(name: "nope", in: unknown)))
    }

    // MARK: - Production declaration

    @Test
    func productionDeclarationSatisfiesItsInvariants() {
        let seq = sequence(
            steps: DependencyContainer.bootstrapStepNames,
            invariants: DependencyContainer.bootstrapInvariants
        )
        #expect(seq.validate().isEmpty)
    }

    @Test
    func productionStepNamesAreUnique() {
        let names = DependencyContainer.bootstrapStepNames
        #expect(Set(names).count == names.count)
    }

    @Test
    func everyInvariantNamesRealSteps() {
        let names = Set(DependencyContainer.bootstrapStepNames)
        for inv in DependencyContainer.bootstrapInvariants {
            #expect(names.contains(inv.before))
            #expect(names.contains(inv.after))
        }
    }

    /// Golden order — any reorder of `setup()` is a conscious diff here, and the
    /// three known invariants are asserted directly against it.
    @Test
    func goldenLaunchOrder() {
        #expect(
            DependencyContainer.bootstrapStepNames == [
                "registerHotkeys",
                "attachDictationMeters",
                "registerMessageCodecs",
                "startHotkeyListening",
                "registerHTTPRoutes",
                "startAppBindings",
                "startCompanionLoop",
                "wirePerceptionCallbacks",
                "startCompanionPerception",
                "materializeAgent",
                "startMCPClient",
            ])
    }

    @Test
    func theThreeKnownInvariantsArePresent() {
        let pairs = Set(
            DependencyContainer.bootstrapInvariants.map { "\($0.before)>\($0.after)" }
        )
        #expect(pairs.contains("registerHTTPRoutes>startAppBindings"))
        #expect(pairs.contains("wirePerceptionCallbacks>startCompanionPerception"))
        #expect(pairs.contains("materializeAgent>startMCPClient"))
    }
}
