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

    /// A Monday-first week, so Sunday is the week's last day whatever the
    /// machine's locale.
    static var mondayFirst: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.firstWeekday = 2
        return calendar
    }

    static func weekFacts(day: Int, focus: String? = nil) -> DayFacts {
        let calendar = mondayFirst
        let at = { (d: Int, h: Int) in
            calendar.date(from: DateComponents(year: 2026, month: 10, day: d, hour: h))!
        }
        func done(_ id: String, list: String) -> AgendaReminder {
            AgendaReminder(
                id: id, title: id, listID: list, listTitle: list.capitalized, isCompleted: true,
                completedAt: at(day - 1, 10))
        }
        return DayFacts(
            now: at(day, 21), calendar: calendar,
            doneThisWeek: [
                done("a", list: "work"), done("b", list: "work"), done("c", list: "daily"),
            ],
            weekFocus: focus)
    }

    @Test func onTheWeeksLastDayTheWrapUpLooksBackAndAsksForAFocus() {
        // Sunday 4 October.
        let sunday = MomentPrompts.eveningWrapUp(
            facts: Self.weekFacts(day: 4, focus: "the job search"), leftovers: [])
        #expect(sunday.contains("It's the week's last day: look back on the week too."))
        #expect(sunday.contains("This week: 3 done — Work 2, Daily 1."))
        #expect(sunday.contains("This week's focus was: the job search."))
        var withMustDos = Self.weekFacts(day: 4)
        withMustDos.mustDoDays = ["2026-10-01": true, "2026-10-02": false, "2026-10-03": true]
        #expect(
            MomentPrompts.eveningWrapUp(facts: withMustDos, leftovers: []).contains(
                "The must-do got done on 2 of the 3 days it was set."))
        #expect(sunday.contains(#""focus": "<next week's one focus, a few words>""#))
        // Saturday: an ordinary evening.
        let saturday = MomentPrompts.eveningWrapUp(facts: Self.weekFacts(day: 3), leftovers: [])
        #expect(!saturday.contains("week's last day"))
        #expect(!saturday.contains(#""focus""#))
    }

    @Test func theWeeksLastWrapUpAlsoAsksAboutWhatHasWaitedLongest() throws {
        let calendar = Self.mondayFirst
        let at = { (d: Int, h: Int) in
            calendar.date(from: DateComponents(year: 2026, month: 10, day: d, hour: h))!
        }
        func task(_ id: String, due day: Int) -> AgendaReminder {
            AgendaReminder(
                id: id, title: id.capitalized, listID: "life", listTitle: "Life",
                due: at(day, 0))
        }
        // Sunday 4 October: one task due today, seven overdue (3 Oct back to
        // 27 Sep).
        var sunday = Self.weekFacts(day: 4)
        sunday.dueOrOverdue = [task("bins", due: 4)] + (1...7).map { task("old\($0)", due: 4 - $0) }
        let leftovers = DayEngine.leftovers(sunday)
        #expect(leftovers.map(\.id) == ["bins", "old7", "old6", "old5", "old4", "old3"])
        // On any other day, only today's.
        var saturday = Self.weekFacts(day: 3)
        saturday.dueOrOverdue = [task("bins", due: 3), task("old", due: 1)]
        #expect(DayEngine.leftovers(saturday).map(\.id) == ["bins"])

        let text = MomentPrompts.eveningWrapUp(facts: sunday, leftovers: leftovers)
        #expect(text.contains("Still open from today (id — title):\n- bins — Bins"))
        #expect(text.contains("Waiting since an earlier day (id — title — due):"))
        #expect(text.contains("- old7 — Old7 — Sunday 27 September"))
        #expect(text.contains(#""drop" lets it go."#))

        // Without the model: what waited is kept undated, today's rolls on.
        let card = FallbackCards.eveningWrapUp(facts: sunday, leftovers: leftovers)
        #expect(card.leftovers.first?.suggestion == .tomorrow)
        #expect(card.leftovers.first?.since == nil)
        let oldest = try #require(card.leftovers.dropFirst().first)
        #expect(oldest.suggestion == .later)
        #expect(oldest.since == at(-3, 0))
        #expect(oldest.waiting?.hasPrefix("Waiting since") == true)
        #expect(card.leftoversHeading == "Still open")
        #expect(
            FallbackCards.eveningWrapUp(facts: saturday, leftovers: DayEngine.leftovers(saturday))
                .leftoversHeading == "Left from today")
        // A leftover saved before it could have waited still loads.
        let saved = #"{"reminderID": "x", "title": "X", "suggestion": "tomorrow"}"#
        #expect(try JSONDecoder().decode(Leftover.self, from: Data(saved.utf8)).since == nil)
    }

    @Test func theWeeksFocusOpensTheDayAndGuidesThePlan() {
        let facts = Self.weekFacts(day: 5, focus: "the job search")
        #expect(
            MomentPrompts.dayOpening(facts: facts, profile: [], carryOver: nil)
                .contains("This week's focus: the job search"))
        #expect(
            MomentPrompts.morningPlan(facts: facts).contains(
                "This week's focus: the job search. When a task serves it, let it be the must-do."))
    }

    @Test func theNightReflectionAsksForTasksTheDayShowed() {
        let text = MomentPrompts.nightReflection(facts: Self.facts, profile: [])
        #expect(text.contains(#""tasks": [{"title":"#))
        #expect(text.contains("At most three; most nights none."))
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
