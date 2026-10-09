//
//  NotificationSourceTests.swift
//  tesseractTests
//
//  Who a banner is from, decided by code before a model sees it: a person
//  goes to Triage, an app's news waits for the next Breakpoint, noise (the
//  system, a game) is never shown — and an owner rule beats all three.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct NotificationSourceTests {

    static func banner(_ app: String, title: String = "", subtitle: String = "", body: String = "")
        -> ObservedNotification
    {
        ObservedNotification(
            id: "notification:\(app)|\(title)", app: app, title: title, subtitle: subtitle,
            body: body, arrivedAt: Date(timeIntervalSince1970: 1_790_880_000))
    }

    static let dota = AppIdentity(
        name: "Dota 2", bundleID: "com.valvesoftware.dota2",
        bundlePath:
            "/Users/owner/Library/Application Support/Steam/steamapps/common/dota 2 beta/game/bin/osx64/dota2.app"
    )
    static let slack = AppIdentity(name: "Slack", bundleID: "com.tinyspeck.slackmacgap")

    struct Row: Sendable, CustomTestStringConvertible {
        let name: String
        let notification: ObservedNotification
        let app: AppIdentity?
        let source: NotificationSource
        var testDescription: String { name }
    }

    static let rows: [Row] = [
        Row(
            name: "Game Mode is the system's",
            notification: banner("Game Mode", title: "Game Mode: On"),
            app: nil, source: .noise),
        Row(
            name: "a game is noise", notification: banner("Dota 2", title: "Your game is ready"),
            app: dota, source: .noise),
        Row(
            name: "Slack is a person", notification: banner("Slack", title: "Anna"), app: slack,
            source: .person),
        Row(
            name: "Slack by name alone", notification: banner("Slack", title: "Anna"), app: nil,
            source: .person),
        Row(
            name: "Discord is a person", notification: banner("Discord", title: "rtm"), app: nil,
            source: .person),
        Row(
            name: "a site in Chrome is an app",
            notification: banner(
                "Google Chrome", title: "Claude has a question", subtitle: "claude.ai"),
            app: nil, source: .app),
        Row(
            name: "mail in Chrome is a person",
            notification: banner("Google Chrome", title: "Anna", subtitle: "mail.google.com"),
            app: nil, source: .person),
        Row(
            name: "an image is ready is an app's news",
            notification: banner("ChatGPT", title: "Your image is ready to review"), app: nil,
            source: .app),
        Row(
            name: "a bot's message in Slack is an app's news",
            notification: banner(
                "Slack", title: "Acme", subtitle: "Jira",
                body: "🔔 Jira bot commented on a Sub-task you are assigned to"),
            app: slack, source: .app),
        Row(
            name: "a bot posting in a Slack channel is an app's news",
            notification: banner(
                "Slack", title: "Acme", subtitle: "#builds", body: "CircleCI: build 812 passed"),
            app: slack, source: .app),
        Row(
            name: "a person in a Slack channel is a person",
            notification: banner(
                "Slack", title: "Acme", subtitle: "#qa", body: "Roman: moved the ticket to test"),
            app: slack, source: .person),
        Row(
            name: "a person's direct message is a person",
            notification: banner("Slack", title: "Acme", subtitle: "Anna", body: "Got a minute?"),
            app: slack, source: .person),
        Row(
            name: "a workspace named like a tool is still people",
            notification: banner(
                "Slack", title: "GitHub", subtitle: "Anna", body: "Can you review my PR?"),
            app: slack, source: .person),
        Row(
            name: "a page can't wait",
            notification: banner(
                "Slack", title: "Acme", subtitle: "PagerDuty", body: "Triggered: API is down"),
            app: slack, source: .person),
    ]

    @Test(arguments: rows)
    func classify(_ row: Row) {
        #expect(NotificationSources.classify(row.notification, app: row.app) == row.source)
    }

    @Test func gamesAreKnownByCategoryStoreOrPublisher() {
        #expect(AppIdentity(name: "Chess", category: "public.app-category.games").isGame)
        #expect(AppIdentity(name: "Doom", category: "public.app-category.action-games").isGame)
        #expect(AppIdentity(name: "Arcade", supportsGameMode: true).isGame)
        #expect(Self.dota.isGame)
        #expect(AppIdentity(name: "Steam", bundleID: "com.valvesoftware.steam").isGame)
        #expect(!Self.slack.isGame)
        #expect(
            !AppIdentity(
                name: "Safari", bundleID: "com.apple.Safari",
                bundlePath: "/Applications/Safari.app",
                category: "public.app-category.productivity"
            ).isGame)
    }

    @Test func theLedgerIgnoresNoiseAndHoldsAppNewsForTheBreakpoint() {
        var ledger = SeenLedger()
        let gameMode = Self.banner("Game Mode", title: "Game Mode: On").classified(.noise)
        let image = Self.banner("ChatGPT", title: "Your image is ready").classified(.app)
        let anna = Self.banner("Slack", title: "Anna").classified(.person)
        for notification in [gameMode, image, anna] {
            ledger.arrived(notification, present: true, rules: [])
        }
        let now = Date(timeIntervalSince1970: 1_790_880_060)
        // Noise is never shown; an app's news waits; only the person is triaged.
        #expect(ledger.unresolved(now: now).map(\.id) == [image.id, anna.id])
        #expect(ledger.untriaged(now: now).map(\.id) == [anna.id])
    }

    @Test func anOwnerRuleBeatsTheSource() {
        var ledger = SeenLedger()
        let image = Self.banner("ChatGPT", title: "Your image is ready").classified(.app)
        let raise = TriageRule(
            app: "ChatGPT", action: .raise, phrase: "always tell me about images")
        ledger.arrived(image, present: true, rules: [raise])
        #expect(ledger.entry(image.id)?.rule == .raise)
    }

    @Test func aBannerSavedBeforeSourcesExistedIsTriagedAsBefore() {
        var ledger = SeenLedger()
        let old = Self.banner("GitHub", title: "CI")
        ledger.arrived(old, present: true, rules: [])
        #expect(ledger.untriaged(now: Date(timeIntervalSince1970: 1_790_880_060)).count == 1)
    }
}
