//
//  MenuBarClockGalleryTests.swift
//  tesseractTests
//
//  The menu bar's clock beside the glyph, drawn with the status item's own
//  views over a stand-in for the menu bar: a started step's time left
//  ("12m"), an event coming up ("in 12m") and the time to leave for one
//  ("leave in 12m"). With PANEL_GALLERY_DIR set
//  (TEST_RUNNER_PANEL_GALLERY_DIR through xcodebuild), each is also written
//  there as a PNG, in dark and light, for judging by eye.
//

import AppKit
import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct MenuBarClockGalleryTests {

    static let directory = ProcessInfo.processInfo.environment["PANEL_GALLERY_DIR"].map {
        URL(fileURLWithPath: $0, isDirectory: true)
    }

    @Test(arguments: [MenuBarClock.Kind.focus, .event, .leave])
    func theClockSitsBesideTheGlyph(_ kind: MenuBarClock.Kind) throws {
        let now = Date(timeIntervalSinceReferenceDate: 0)
        let clock = MenuBarClock(
            kind: kind, title: "Design review", until: now.addingTimeInterval(12 * 60))
        for dark in [false, true] {
            let rep = try render(MenuBarClockText.label(clock, now: now), dark: dark)
            #expect(rep.pixelsWide > 0)
            guard let directory = Self.directory else { continue }
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let data = try #require(rep.representation(using: .png, properties: [:]))
            try data.write(
                to: directory.appendingPathComponent(
                    "menubar-\(kind)-\(dark ? "dark" : "light").png"))
        }
    }

    @Test func theMenuOpensOnTheDayWithJarvisOn() throws {
        let manager = MenuBarManager(settings: SettingsManager(store: InMemorySettingsStore()))
        let soon = Date().addingTimeInterval(12 * 60 + 30)
        manager.updateClock(MenuBarClock(kind: .event, title: "Design review", until: soon))
        var on = true
        manager.companionOn = { on }
        let menu = NSMenu()
        manager.menuNeedsUpdate(menu)
        let titles = menu.items.prefix(4).map(\.title)
        #expect(titles.first == "Jarvis")
        #expect(titles.dropFirst().first?.hasPrefix("Design review at ") == true)
        #expect(titles.dropFirst().first?.hasSuffix("— in 13 min") == true)
        #expect(Array(titles.suffix(2)) == ["Open Today", "Write a Thought Down…"])
        // With the Companion off, the menu is as it was.
        on = false
        manager.menuNeedsUpdate(menu)
        #expect(!menu.items.contains { $0.title == "Open Today" })
    }

    /// A strip of menu bar, the status item's content left in it.
    private func render(_ text: String, dark: Bool) throws -> NSBitmapImageRep {
        let strip = NSView(frame: NSRect(x: 0, y: 0, width: 160, height: 24))
        strip.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        strip.wantsLayer = true
        strip.layer?.backgroundColor =
            (dark ? NSColor(white: 0.16, alpha: 1) : NSColor(white: 0.94, alpha: 1)).cgColor
        let (stack, icon, label) = MenuBarManager.makeStatusContent()
        let config = NSImage.SymbolConfiguration(pointSize: 16, weight: .medium)
        let image = try #require(
            NSImage(systemSymbolName: "waveform", accessibilityDescription: nil)?
                .withSymbolConfiguration(config))
        image.isTemplate = true
        icon.image = image
        icon.contentTintColor = .labelColor
        label.stringValue = text
        label.isHidden = false
        strip.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: strip.leadingAnchor, constant: 10),
            stack.centerYAnchor.constraint(equalTo: strip.centerYAnchor),
        ])
        strip.layoutSubtreeIfNeeded()
        let rep = try #require(strip.bitmapImageRepForCachingDisplay(in: strip.bounds))
        strip.cacheDisplay(in: strip.bounds, to: rep)
        return rep
    }
}
