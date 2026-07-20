//
//  TurnReplayGuardTests.swift
//  tesseractTests
//
//  Unit coverage for the agent loop's replay breaker: what counts as an
//  identical turn (canonical arguments, key order, whitespace), what resets
//  the chain, and how consecutive replays are counted.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct TurnReplayGuardTests {

    private func turn(
        text: String = "Good — logging it now.",
        calls: [(name: String, argumentsJSON: String)] = [
            ("track", #"{"kind":"observation","payload":{"domain":"body"}}"#)
        ]
    ) -> AssistantMessage {
        AssistantMessage.create(
            content: text,
            toolCalls: calls.map {
                ToolCallInfo(id: UUID().uuidString, name: $0.name, argumentsJSON: $0.argumentsJSON)
            }
        )
    }

    // MARK: - Verdicts

    @Test func firstOccurrenceIsFresh() {
        var guardState = TurnReplayGuard()
        #expect(guardState.observe(turn()) == .fresh)
    }

    @Test func identicalTurnsCountConsecutiveReplays() {
        var guardState = TurnReplayGuard()
        #expect(guardState.observe(turn()) == .fresh)
        #expect(guardState.observe(turn()) == .replay(consecutive: 1))
        #expect(guardState.observe(turn()) == .replay(consecutive: 2))
    }

    @Test func differentTextIsFreshAndRestartsTheChain() {
        var guardState = TurnReplayGuard()
        #expect(guardState.observe(turn(text: "a")) == .fresh)
        #expect(guardState.observe(turn(text: "b")) == .fresh)
        #expect(guardState.observe(turn(text: "b")) == .replay(consecutive: 1))
    }

    @Test func differentArgumentsAreFresh() {
        var guardState = TurnReplayGuard()
        #expect(guardState.observe(turn(calls: [("track", #"{"v":1}"#)])) == .fresh)
        #expect(guardState.observe(turn(calls: [("track", #"{"v":2}"#)])) == .fresh)
    }

    @Test func differentCallOrderIsFresh() {
        var guardState = TurnReplayGuard()
        #expect(
            guardState.observe(turn(calls: [("track", "{}"), ("report_back", "{}")])) == .fresh)
        #expect(
            guardState.observe(turn(calls: [("report_back", "{}"), ("track", "{}")])) == .fresh)
    }

    /// The live transcript's exact shape: the replaying model re-emits the
    /// same call with shuffled JSON key order. Same call, still a replay.
    @Test func shuffledArgumentKeyOrderIsStillAReplay() {
        var guardState = TurnReplayGuard()
        let first = turn(
            calls: [
                ("track", #"{"kind":"observation","payload":{"domain":"body","kind":"exercise"}}"#)
            ])
        let shuffled = turn(
            calls: [
                ("track", #"{"payload":{"kind":"exercise","domain":"body"},"kind":"observation"}"#)
            ])
        #expect(guardState.observe(first) == .fresh)
        #expect(guardState.observe(shuffled) == .replay(consecutive: 1))
    }

    @Test func surroundingWhitespaceInTextDoesNotBreakIdentity() {
        var guardState = TurnReplayGuard()
        #expect(guardState.observe(turn(text: "\n\nDone.\n")) == .fresh)
        #expect(guardState.observe(turn(text: "Done.")) == .replay(consecutive: 1))
    }

    // MARK: - Resets

    @Test func toolLessTurnResetsTheChain() {
        var guardState = TurnReplayGuard()
        #expect(guardState.observe(turn()) == .fresh)
        #expect(guardState.observe(turn(calls: [])) == .fresh)
        #expect(guardState.observe(turn()) == .fresh)
    }

    @Test func explicitResetForgetsTheStandingTurn() {
        var guardState = TurnReplayGuard()
        #expect(guardState.observe(turn()) == .fresh)
        guardState.reset()
        #expect(guardState.observe(turn()) == .fresh)
    }

    // MARK: - Signature

    @Test func signatureIsNilWithoutToolCalls() {
        #expect(TurnReplayGuard.signature(of: turn(calls: [])) == nil)
    }

    @Test func signatureIgnoresToolCallIDs() {
        // `turn` mints a fresh UUID per call — equal signatures despite them.
        #expect(TurnReplayGuard.signature(of: turn()) == TurnReplayGuard.signature(of: turn()))
    }
}
