//
//  NotificationSource.swift
//  tesseract
//
//  Who a banner is from, decided by code before anything reaches the model:
//  a person (a chat, a mail, a mention), an app's own news (a download
//  finished, an image is ready), or noise (the system, a game). Only people
//  are worth a Triage; an app's news waits for the next Breakpoint; noise is
//  never shown. The owner's own rules still come first.
//
//  Pure: the runtime looks the app up (its bundle id, category and path) and
//  hands the facts in.
//

import Foundation

nonisolated enum NotificationSource: String, Sendable, Equatable, Hashable, Codable {
    /// A message from a person: a chat, a mail, a call, a mention.
    case person
    /// An app's own news: it can wait for the next Breakpoint.
    case app
    /// The system or a game: never Jarvis's business.
    case noise
}

/// What the runtime knows about an app: the one behind a banner, or the one
/// in front.
nonisolated struct AppIdentity: Sendable, Equatable {
    var name: String
    var bundleID: String?
    /// Where the app lives; games from a store sit in its library folder.
    var bundlePath: String?
    /// `LSApplicationCategoryType` from the app's Info.plist.
    var category: String?
    /// `GCSupportsGameMode` from the app's Info.plist.
    var supportsGameMode: Bool = false

    init(
        name: String, bundleID: String? = nil, bundlePath: String? = nil,
        category: String? = nil, supportsGameMode: Bool = false
    ) {
        self.name = name
        self.bundleID = bundleID
        self.bundlePath = bundlePath
        self.category = category
        self.supportsGameMode = supportsGameMode
    }

    /// A game, by its own declaration or by where it came from. Many games
    /// declare no category (many store games don't), so the store's library folder
    /// and the publisher's bundle id count too.
    var isGame: Bool {
        if let category, category.lowercased().contains("games") { return true }
        if supportsGameMode { return true }
        if let bundlePath, GameApps.libraryFolders.contains(where: bundlePath.contains) {
            return true
        }
        if let bundleID = bundleID?.lowercased(),
            GameApps.publisherPrefixes.contains(where: bundleID.hasPrefix)
        {
            return true
        }
        return false
    }
}

nonisolated enum GameApps {
    /// Folders game stores install into.
    static let libraryFolders = ["/steamapps/common/", "/Epic Games/", "/GOG Games/"]
    /// Bundle-id prefixes of game publishers and stores.
    static let publisherPrefixes = [
        "com.valvesoftware.", "com.blizzard.", "net.battle.", "com.epicgames.",
        "com.riotgames.", "com.feralinteractive.", "com.aspyr.", "com.paradoxinteractive.",
        "com.ea.", "com.ubisoft.", "com.cdprojektred.", "com.gog.", "unity.",
    ]
}

nonisolated enum NotificationSources {

    /// Messaging apps: what they notify about is someone talking to the owner.
    static let personBundleIDs: Set<String> = [
        "com.tinyspeck.slackmacgap", "com.apple.MobileSMS", "com.apple.mail",
        "com.apple.FaceTime", "com.hnc.Discord", "ru.keepcoder.Telegram",
        "org.telegram.desktop", "net.whatsapp.WhatsApp", "desktop.WhatsApp",
        "org.whispersystems.signal-desktop", "com.microsoft.teams2", "com.microsoft.teams",
        "com.microsoft.Outlook", "com.readdle.smartemail-Mac", "com.readdle.SparkDesktop",
        "com.mimestream.Mimestream", "com.superhuman.electron", "us.zoom.xos",
        "com.facebook.archon", "com.facebook.archon.developerID", "com.viber.osx",
        "im.riot.app", "com.skype.skype",
    ]

    /// The same apps by display name, for a banner whose app isn't running.
    static let personAppNames: Set<String> = [
        "slack", "messages", "mail", "facetime", "discord", "telegram", "whatsapp", "signal",
        "microsoft teams", "teams", "microsoft outlook", "outlook", "spark", "spark desktop",
        "mimestream", "superhuman", "zoom", "zoom.us", "messenger", "viber", "element", "skype",
    ]

    /// Browsers: a site's banner is a person only when the site is for messages.
    static let browserNames: Set<String> = [
        "google chrome", "safari", "arc", "firefox", "microsoft edge", "brave browser", "dia",
        "chromium", "opera", "vivaldi",
    ]

    static let messagingSites = [
        "mail.google.com", "gmail", "web.whatsapp.com", "slack.com", "discord.com",
        "teams.microsoft.com", "teams.live.com", "web.telegram.org", "messenger.com",
        "outlook.live.com", "outlook.office.com", "outlook.office365.com", "chat.google.com",
        "linkedin.com", "mail.proton.me", "fastmail.com",
    ]

    /// The system's own banners, which are never about the owner's day.
    static let systemNoiseNames: Set<String> = [
        "game mode", "software update", "time machine", "login items",
        "background items added", "screen time", "game center",
    ]

    static func classify(_ notification: ObservedNotification, app: AppIdentity?)
        -> NotificationSource
    {
        let name = notification.app.lowercased()
        if systemNoiseNames.contains(name) { return .noise }
        if app?.isGame == true { return .noise }
        if let bundleID = app?.bundleID, personBundleIDs.contains(bundleID) { return .person }
        if personAppNames.contains(name) { return .person }
        if browserNames.contains(name) {
            let site = "\(notification.subtitle) \(notification.title)".lowercased()
            return messagingSites.contains(where: site.contains) ? .person : .app
        }
        return .app
    }

    /// What code does with a banner no owner rule matched: noise is never
    /// shown, an app's news waits for the next Breakpoint, a person (or a
    /// banner admitted before sources existed) goes to Triage.
    static func defaultAction(for source: NotificationSource?) -> TriageRule.Action? {
        switch source {
        case .noise: .ignore
        case .app: .hold
        case .person, nil: nil
        }
    }
}
