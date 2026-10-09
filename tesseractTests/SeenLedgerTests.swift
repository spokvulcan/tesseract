//
//  SeenLedgerTests.swift
//  tesseractTests
//
//  The seen ledger and the owner's rules: what counts as seen, what stays
//  unresolved, what a rule does — before any model sees a notification.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SeenLedgerTests {

    static let base = Date(timeIntervalSince1970: 1_790_760_000)

    static func notification(
        _ id: String, app: String = "Slack", title: String = "Anna", body: String = "can you look?",
        minutes: Double = 0
    )
        -> ObservedNotification
    {
        ObservedNotification(
            id: id, app: app, title: title, subtitle: "", body: body,
            arrivedAt: base.addingTimeInterval(minutes * 60))
    }

    @Test func aBannerSeenTwiceIsAdmittedOnce() {
        var ledger = SeenLedger()
        let first = ledger.arrived(Self.notification("a"), present: true, rules: [])
        let again = ledger.arrived(Self.notification("a"), present: true, rules: [])
        #expect(first)
        #expect(!again)
        #expect(ledger.entries.count == 1)
    }

    @Test func openingTheAppSoonAfterMeansSeen() {
        var ledger = SeenLedger()
        ledger.arrived(Self.notification("a"), present: true, rules: [])
        ledger.arrived(Self.notification("b", app: "Mail"), present: true, rules: [])
        let marked = ledger.appActivated("slack", at: Self.base.addingTimeInterval(5 * 60))
        #expect(marked == ["a"])
        #expect(ledger.unresolved(now: Self.base.addingTimeInterval(6 * 60)).map(\.id) == ["b"])
    }

    @Test func openingTheAppMuchLaterDoesNotCountWhilePresent() {
        var ledger = SeenLedger()
        ledger.arrived(Self.notification("a"), present: true, rules: [])
        let marked = ledger.appActivated("Slack", at: Self.base.addingTimeInterval(40 * 60))
        #expect(marked.isEmpty)
    }

    @Test func whatArrivedWhileAwayIsSeenWhenTheAppIsOpened() {
        var ledger = SeenLedger()
        ledger.arrived(Self.notification("a"), present: false, rules: [])
        let marked = ledger.appActivated("Slack", at: Self.base.addingTimeInterval(3 * 3600))
        #expect(marked == ["a"])
    }

    @Test func unresolvedItemsExpireAfterADay() {
        var ledger = SeenLedger()
        ledger.arrived(Self.notification("a"), present: true, rules: [])
        #expect(ledger.unresolved(now: Self.base.addingTimeInterval(23 * 3600)).count == 1)
        #expect(ledger.unresolved(now: Self.base.addingTimeInterval(25 * 3600)).isEmpty)
    }

    @Test func rulesIgnoreHoldAndRaise() {
        let rules = [
            TriageRule(
                app: "GitHub", keywords: ["passed"], action: .ignore, phrase: "never CI passing"),
            TriageRule(app: "Mail", action: .hold, phrase: "mail can wait"),
            TriageRule(sender: "Anna", action: .raise, phrase: "always Anna"),
        ]
        var ledger = SeenLedger()
        ledger.arrived(
            Self.notification("ci", app: "GitHub", title: "CI", body: "All checks passed"),
            present: true, rules: rules)
        ledger.arrived(
            Self.notification("ci-fail", app: "GitHub", title: "CI", body: "Build failed"),
            present: true, rules: rules)
        ledger.arrived(
            Self.notification("mail", app: "Mail", title: "Newsletter"), present: true, rules: rules
        )
        ledger.arrived(
            Self.notification("anna", app: "Slack", title: "Anna"), present: true, rules: rules)
        let now = Self.base.addingTimeInterval(60)
        #expect(ledger.entry("ci")?.rule == .ignore)
        #expect(ledger.entry("ci-fail")?.rule == nil)
        #expect(ledger.entry("anna")?.rule == .raise)
        // Ignored never shows; held waits for a Breakpoint but is never triaged.
        #expect(Set(ledger.unresolved(now: now).map(\.id)) == ["ci-fail", "mail", "anna"])
        #expect(Set(ledger.untriaged(now: now).map(\.id)) == ["ci-fail", "anna"])
    }

    @Test func aRuleMustNarrowSomething() {
        let everything = TriageRule(action: .ignore, phrase: "")
        #expect(!everything.isSpecific)
        #expect(!everything.matches(Self.notification("a")))
    }

    @Test func rulesRoundTripThroughTheSetting() {
        let rules = [
            TriageRule(id: "r1", app: "GitHub", keywords: ["passed"], action: .ignore, phrase: "CI")
        ]
        #expect(TriageRules.decode(TriageRules.encode(rules)) == rules)
        #expect(TriageRules.decode("not json").isEmpty)
    }

    @Test func tesseractsOwnBannersNeverCount() {
        let own = CapturedNotification(app: "Tesseract Agent", title: "Jarvis", body: "hi")
        #expect(own.admitted(selfDisplayNames: ["Tesseract Agent"]) == nil)
        let stacked = CapturedNotification(app: "Stacked summary", title: "Tesseract Agent: Jarvis")
        #expect(stacked.admitted(selfDisplayNames: ["Tesseract Agent"]) == nil)
        let other = CapturedNotification(app: "Slack", title: "Anna", body: "hi", uuid: "U1")
        #expect(other.admitted(selfDisplayNames: ["Tesseract Agent"])?.id == "notification:U1")
    }
}

struct DeliveryLadderTests {

    static func snapshot(
        hour: Int, present: Bool = true, speaks: Bool = true, frontmost: String? = nil,
        game: Bool = false
    ) -> DaySnapshot {
        var settings = DaySettings()
        settings.speaks = speaks
        let now = Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: 30, hour: hour))!
        return DaySnapshot(
            now: now, settings: settings, agenda: .empty, ownerPresent: present,
            frontmostBundleID: frontmost, frontmostIsGame: game)
    }

    @Test func sittingDownEndsTheMorningOfQuietHoursThatStartAfterMidnight() {
        let at = { (hour: Int, minute: Int) in
            Calendar.current.date(
                from: DateComponents(year: 2026, month: 9, day: 30, hour: hour, minute: minute))!
        }
        var settings = DaySettings()
        settings.quietStartMinutes = 60
        settings.quietEndMinutes = 9 * 60
        let morning = DaySnapshot(
            now: at(7, 30), settings: settings, agenda: .empty, ownerPresent: true)
        #expect(DeliveryLadder.quietHoursAreNight(settings))
        #expect(DeliveryLadder.dayStarted(at(7, 10), snapshot: morning))
        // A daytime window holds, sat down or not.
        settings.quietStartMinutes = 13 * 60
        settings.quietEndMinutes = 15 * 60
        let afternoon = DaySnapshot(
            now: at(13, 30), settings: settings, agenda: .empty, ownerPresent: true)
        #expect(!DeliveryLadder.quietHoursAreNight(settings))
        #expect(!DeliveryLadder.dayStarted(at(7, 10), snapshot: afternoon))
        // An evening window has a bedtime but no morning end: it holds.
        settings.quietStartMinutes = 20 * 60
        settings.quietEndMinutes = 23 * 60
        let evening = DaySnapshot(
            now: at(21, 0), settings: settings, agenda: .empty, ownerPresent: true)
        #expect(DeliveryLadder.quietHoursAreNight(settings))
        #expect(!DeliveryLadder.dayStarted(at(9, 30), snapshot: evening))
        // A sit-down after the morning lifts nothing that night.
        settings.quietStartMinutes = 23 * 60
        settings.quietEndMinutes = 8 * 60
        let night = DaySnapshot(
            now: Calendar.current.date(
                from: DateComponents(year: 2026, month: 10, day: 1, hour: 0, minute: 20))!,
            settings: settings, agenda: .empty, ownerPresent: true)
        #expect(!DeliveryLadder.dayStarted(at(14, 43), snapshot: night))
    }

    @Test func aGameInFrontGetsNoPanelAndNoVoice() {
        let playing = Self.snapshot(hour: 20, frontmost: "com.valvesoftware.dota2", game: true)
        #expect(DeliveryLadder.rungs(for: .normal, snapshot: playing) == [.today])
        #expect(DeliveryLadder.rungs(for: .urgent, snapshot: playing) == [.banner])
    }

    @Test(arguments: [
        (14, true, true, nil as String?, Importance.normal, [DeliveryRung.panel]),
        (14, true, true, nil, .urgent, [.panel, .voice]),
        (14, true, false, nil, .urgent, [.panel, .banner]),
        (14, false, true, nil, .normal, [.today]),
        (14, false, true, nil, .urgent, [.banner]),
        (14, true, true, "us.zoom.xos", .normal, [.today]),
        (14, true, true, "us.zoom.xos", .urgent, [.banner]),
        (23, true, true, nil, .urgent, [.today]),
        (7, true, true, nil, .urgent, [.today]),
    ])
    func rung(
        hour: Int, present: Bool, speaks: Bool, frontmost: String?, importance: Importance,
        expected: [DeliveryRung]
    ) {
        let snapshot = Self.snapshot(
            hour: hour, present: present, speaks: speaks, frontmost: frontmost)
        #expect(DeliveryLadder.rungs(for: importance, snapshot: snapshot) == expected)
    }

    @Test(arguments: [
        // The reflection runs on power, or on a battery at least half full
        // with a cool Mac.
        (
            MomentKind.nightReflection,
            PowerState(onACPower: false, batteryPercent: 90, thermal: .nominal), false
        ),
        (
            .nightReflection, PowerState(onACPower: false, batteryPercent: 49, thermal: .nominal),
            true
        ),
        (.nightReflection, PowerState(onACPower: false, batteryPercent: 90, thermal: .fair), true),
        (.nightReflection, PowerState(onACPower: true, batteryPercent: 90, thermal: .fair), false),
        (
            .nightReflection, PowerState(onACPower: true, batteryPercent: 90, thermal: .serious),
            true
        ),
        (
            .nightReflection, PowerState(onACPower: true, batteryPercent: 90, thermal: .nominal),
            false
        ),
        (.triage, PowerState(onACPower: false, batteryPercent: 15, thermal: .nominal), true),
        (.triage, PowerState(onACPower: false, batteryPercent: 60, thermal: .fair), false),
        (.triage, PowerState(onACPower: true, batteryPercent: nil, thermal: .serious), true),
        (.breakpoint, PowerState(onACPower: false, batteryPercent: 5, thermal: .critical), false),
    ])
    func governor(kind: MomentKind, power: PowerState, deferred: Bool) {
        #expect((Governor.deferral(for: kind, power: power) != nil) == deferred)
    }
}
