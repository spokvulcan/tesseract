//
//  CompanionNotifierTests.swift
//  tesseractTests
//
//  A scratch launch's notifier (ADR-0073). The dev build shares the installed
//  app's notification center (one bundle id), and the runtime cancels every
//  scheduled nudge its Agenda doesn't list, so a scratch launch's empty Agenda
//  would cancel the owner's real nudges. Its notifier keeps nudges in memory
//  instead: scheduled, listed and cancelled there, none delivered, and it
//  never asks for permission.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct CompanionNotifierTests {

    @Test func aScratchLaunchKeepsItsNudgesInMemory() async {
        let notifier = CompanionNotifier(usesOS: false)
        #expect(await notifier.activate() == false)

        let standup = Nudge(
            id: "nudge.event.standup.1", eventID: "standup",
            fireAt: Date(timeIntervalSinceNow: 600),
            title: "Standup", body: "In 10 min")
        let review = Nudge(
            id: "nudge.event.review.1", eventID: "review",
            fireAt: Date(timeIntervalSinceNow: 3_600),
            title: "Design review", body: "In 10 min")
        await notifier.schedule(standup)
        await notifier.schedule(review)
        #expect(await notifier.scheduledNudgeIDs() == [standup.id, review.id])

        notifier.cancel(nudgeIDs: [standup.id])
        #expect(await notifier.scheduledNudgeIDs() == [review.id])
        #expect(await notifier.deliveredNudges().isEmpty)
    }
}
