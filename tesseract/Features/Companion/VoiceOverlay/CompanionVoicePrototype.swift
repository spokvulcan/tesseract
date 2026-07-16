//
//  CompanionVoicePrototype.swift
//  tesseract
//

import AppKit
import SwiftUI

// MARK: - Concepts

/// PROTOTYPE (map #301, ticket #328) — one voice-overlay concept: a placement,
/// a panel level, and a view over the shared `CompanionVoiceFeed`. The
/// concepts answer the ticket's form question from structurally different
/// directions; the Settings picker chooses which one the demo scenes perform
/// on. Reaction round 1 (2026-07-16) killed Horizon and kept both survivors.
@MainActor
struct CompanionVoiceConcept: Identifiable {
    let id: String
    let displayName: String
    /// One-line thesis shown under the Settings picker, so flipping concepts
    /// carries its argument along.
    let thesis: String
    let placement: OverlayPlacement
    /// Notch-and-edge concepts draw over the menu-bar region, which needs a
    /// level above the status bar (the TTS notch precedent).
    let panelLevel: NSWindow.Level
    let makeView: @MainActor (CompanionVoiceFeed, CompanionVoiceActions) -> AnyView
}

@MainActor
enum CompanionVoiceConcepts {
    static let emissary = CompanionVoiceConcept(
        id: "emissary", displayName: "Emissary",
        thesis:
            "The Companion as a physical presence: a glass sphere that knocks where notifications live, then unfolds into a conversation card.",
        placement: .companionEmissary,
        panelLevel: .statusBar
    ) { feed, actions in
        AnyView(EmissaryConceptView(feed: feed, actions: actions))
    }

    static let proscenium = CompanionVoiceConcept(
        id: "proscenium", displayName: "Proscenium",
        thesis:
            "The Companion on stage: the notch grows a glass lip that opens into a word-tracked speaking stage (the TTS notch, absorbed).",
        placement: .companionProscenium,
        panelLevel: .screenSaver
    ) { feed, actions in
        AnyView(ProsceniumConceptView(feed: feed, actions: actions))
    }

    static let all: [CompanionVoiceConcept] = [emissary, proscenium]

    /// Unknown ids (a deleted exploration surviving in defaults — Horizon was
    /// killed in reaction round 1) fall back to Emissary.
    static func concept(for id: String) -> CompanionVoiceConcept {
        all.first { $0.id == id } ?? emissary
    }
}

// MARK: - Placements

nonisolated extension OverlayPlacement {
    /// Emissary: the top-right corner below the menu bar — where
    /// notification banners land, deliberately (the knock replaces the
    /// banner). Tall enough for the unfolded conversation card.
    static let companionEmissary = OverlayPlacement(
        frame: { geometry in
            let size = CGSize(width: 420, height: 600)
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.maxX - size.width - 12,
                y: visible.maxY - size.height - 12,
                width: size.width,
                height: size.height
            )
        }
    )

    /// Proscenium: top-center spanning the notch/menu-bar band, wide enough
    /// for the opened stage.
    static let companionProscenium = OverlayPlacement(
        frame: { geometry in
            let size = CGSize(width: 800, height: 340)
            return NSRect(
                x: geometry.frame.midX - size.width / 2,
                y: geometry.frame.maxY - size.height,
                width: size.width,
                height: size.height
            )
        }
    )
}

// MARK: - Controller

/// PROTOTYPE (map #301, ticket #328) — owns the demo lifecycle: reads the
/// concept picker at play time, raises a throwaway overlay panel, hands the
/// scene to the scripted driver, and tears the panel down when the scene
/// dissolves. Unlike dictation's always-resident `OverlayPanel`, this panel
/// exists only while a scene runs — a prototype leaves nothing behind.
@MainActor
final class CompanionVoicePrototype {

    let feed = CompanionVoiceFeed()

    private let settings: SettingsManager
    private let openChat: @MainActor () -> Void
    private lazy var driver: CompanionVoiceDemoDriver = {
        let driver = CompanionVoiceDemoDriver(feed: feed)
        driver.onActiveChange = { [weak self] active in
            if !active { self?.scheduleTeardown() }
        }
        return driver
    }()

    private var panel: NSPanel?
    private var teardownGeneration = 0

    init(settings: SettingsManager, openChat: @escaping @MainActor () -> Void) {
        self.settings = settings
        self.openChat = openChat
    }

    /// The Settings preview buttons. Re-reads the concept picker every time,
    /// so flipping concept + replaying is the whole comparison loop.
    func play(_ scene: CompanionVoiceScene) {
        let concept = CompanionVoiceConcepts.concept(for: settings.companionVoiceConceptRaw)
        driver.stop()
        raisePanel(for: concept)
        driver.play(scene)
    }

    func stopScene() {
        driver.stop()
        tearDownPanel()
    }

    // MARK: - Panel lifecycle

    private func raisePanel(for concept: CompanionVoiceConcept) {
        tearDownPanel()
        guard let screen = OverlayScreenLocator.preferredScreen() else { return }
        let geometry = ScreenGeometry(frame: screen.frame, visibleFrame: screen.visibleFrame)

        let panel = NSPanel(
            contentRect: concept.placement.frame(geometry),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.level = concept.panelLevel
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        panel.isReleasedWhenClosed = false
        // Interactive from the start — the summons must be clickable. The
        // panel is transparent and nonactivating: clicks land only where a
        // concept draws, and never steal focus (per-pixel hit testing on
        // transparent borderless windows; the TTS notch relies on the same).
        panel.ignoresMouseEvents = false
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.hidesOnDeactivate = false

        let actions = CompanionVoiceActions(
            engage: { [weak self] in self?.driver.engage() },
            bargeIn: { [weak self] in self?.driver.bargeIn() },
            dismiss: { [weak self] in self?.driver.dismiss() },
            openChat: { [weak self] in
                self?.openChat()
                self?.driver.dismiss()
            }
        )
        let hosting = NSHostingView(rootView: concept.makeView(feed, actions))
        hosting.frame = panel.contentView?.bounds ?? .zero
        hosting.autoresizingMask = [.width, .height]
        panel.contentView?.addSubview(hosting)

        panel.orderFrontRegardless()
        self.panel = panel
    }

    /// Dissolve room: the views animate out on `.idle`, then the panel goes.
    private func scheduleTeardown() {
        teardownGeneration += 1
        let generation = teardownGeneration
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
            guard let self, self.teardownGeneration == generation else { return }
            self.tearDownPanel()
        }
    }

    private func tearDownPanel() {
        teardownGeneration += 1
        panel?.orderOut(nil)
        panel = nil
    }
}
