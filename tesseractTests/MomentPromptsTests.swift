//
//  MomentPromptsTests.swift
//  tesseractTests
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct MomentPromptsTests {

    static let facts = TimelineBuilderTests.facts(now: TimelineBuilderTests.local(30, 7, 30))

    @Test func theDayOpeningCarriesProfileAreasAgendaAndNote() {
        let opening = MomentPrompts.dayOpening(
            facts: Self.facts, profile: ["Works on Tesseract in the evenings."],
            carryOver: "Finish the spec first.")
        #expect(opening.hasPrefix("[Day Opening — Wednesday 30 September]"))
        #expect(opening.contains("- Works on Tesseract in the evenings."))
        #expect(opening.contains("Areas: Health, Life, Work"))
        #expect(opening.contains("Carried over from last night: Finish the spec first."))
        #expect(opening.contains("- 09:30–10:00 Standup"))
        #expect(!opening.contains("first draft"))
    }

    @Test func theDayOpeningCarriesLastNightsDraftOfTheDay() {
        let opening = MomentPrompts.dayOpening(
            facts: Self.facts, profile: [], carryOver: "A good close.",
            draft: ["Send Anna the request early", "Documents ready for the ID check"])
        #expect(
            opening.contains(
                """
                Last night's first draft of today:
                - Send Anna the request early
                - Documents ready for the ID check
                """))
    }

    @Test func theMorningPlanListsTheEventsToLeaveFor() {
        let text = MomentPrompts.morningPlan(facts: Self.facts)
        #expect(text.contains("Events still ahead (id · when — title · place):"))
        #expect(text.contains("- e1 · 09:30 — Standup"))
        #expect(text.contains("- e2 · 13:00 — 1:1"))
        #expect(text.contains(#""leave": [{"event": "<event id>", "at": "HH:MM"}]"#))
        #expect(MomentPrompts.leavingEvents(Self.facts).map(\.id) == ["E1", "E2"])
    }

    @Test func undatedTasksComeTheLikeliestFirstAndCollectionsAreSummed() {
        func reminder(_ id: String, list: String) -> AgendaReminder {
            AgendaReminder(id: id, title: "Task \(id)", listID: list, listTitle: list.capitalized)
        }
        // Twenty films, then an Inbox capture, then a task in a list that
        // also holds dated work.
        let films = (1...20).map { reminder("film\($0)", list: "movies") }
        let facts = DayFacts(
            now: TimelineBuilderTests.local(30, 7, 30),
            dueOrOverdue: [
                AgendaReminder(
                    id: "spec", title: "Write the spec", listID: "work", listTitle: "Work",
                    due: TimelineBuilderTests.local(30, 0))
            ],
            undated: films + [reminder("idea", list: "inbox"), reminder("deck", list: "work")],
            inboxListID: "inbox")
        #expect(
            MomentPrompts.plannableUndated(facts).prefix(3).map(\.id) == ["idea", "deck", "film1"])
        let lines = MomentPrompts.agendaLines(facts)
        let listed = lines.filter { $0.hasPrefix("- ") && $0.contains("Task ") }
        #expect(listed.count == MomentPrompts.undatedShown)
        #expect(lines.contains("- …and 7 more, in Movies"))
    }

    @Test func theMorningPlanAsksForTheCardWithIDs() {
        let text = MomentPrompts.morningPlan(facts: Self.facts)
        #expect(text.hasPrefix("[Morning Plan]"))
        #expect(text.contains("- dentist · Health · 11:00 — Call the dentist"))
        #expect(text.contains("- passport · Life · overdue — Renew passport"))
        #expect(text.contains("- gym · Health — Gym"))
        #expect(text.contains("Free before the first meeting: 07:30–09:30 (2 h)."))
        #expect(text.contains(#""must_do""#))
    }

    @Test func theWrapUpNeverAsksAboutMisses() {
        let text = MomentPrompts.eveningWrapUp(facts: Self.facts, leftovers: [])
        #expect(text.contains("Done today: Answer mail"))
        #expect(text.contains("Nothing left over from today."))
        #expect(text.contains("Never call anything missed or failed."))
    }
}
