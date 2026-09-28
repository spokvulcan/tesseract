//
//  DictationPrototypeHost.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  Five Dictation pages on the existing Dictation route, switched by the
//  floating pill in the toolbar (or ← → with no text field focused), plus
//  today's page for comparison. Each variant is a whole flow: the page, what
//  the overlay shows after the words land, and what ⌃⌥Space does. They share
//  the lab, so what one learned the others know. The choice survives a
//  relaunch; what was learned does not.
//

import SwiftUI

struct DictationPrototypeHost<Current: View>: View {
    @ViewBuilder let current: () -> Current
    @AppStorage("dictationPrototype.variant") private var variantIndex = 0
    @Environment(SettingsManager.self) private var settings
    private let lab = DictationLab.shared

    var body: some View {
        let variant = DictationPrototypeVariant(rawValue: variantIndex) ?? .fixBar
        Group {
            switch variant {
            case .fixBar: FixBarPage(lab: lab)
            case .wordCards: WordCardsPage(lab: lab)
            case .teach: TeachPage(lab: lab)
            case .justEdit: JustEditPage(lab: lab)
            case .sayItAgain: SayItAgainPage(lab: lab)
            case .current: current()
            }
        }
        .frame(minWidth: 640, maxWidth: .infinity, minHeight: 480, maxHeight: .infinity)
        .id(variant)
        .navigationTitle("Dictation")
        .toolbar {
            ToolbarItem(placement: .principal) {
                PrototypeSwitcher(
                    labels: DictationPrototypeVariant.allCases.map(\.label), index: $variantIndex)
            }
            .sharedBackgroundVisibility(.hidden)
            ToolbarItem {
                LabMenu(lab: lab)
            }
        }
        .onChange(of: variantIndex, initial: true) { _, index in
            let variant = DictationPrototypeVariant(rawValue: index) ?? .fixBar
            lab.variant = variant
            LabOverlay.apply(variant, settings: settings)
        }
    }
}

/// The lab's own switches, kept out of the pages.
private struct LabMenu: View {
    @Bindable var lab: DictationLab

    var body: some View {
        Menu {
            Toggle("Lean Whisper toward learned words", isOn: $lab.biasesRecognizer)
            Toggle("Run the old Proofread Pass", isOn: $lab.usesProofreadPass)
        } label: {
            Label("Lab", systemImage: "flask")
        }
        .help("Prototype switches")
    }
}

/// Keeps the recording pill in the overlay while a prototype is on, and
/// puts back the owner's own overlay for "Current page".
@MainActor
enum LabOverlay {
    static let variantID = "prototype-lab"
    private static let savedKey = "dictationPrototype.savedOverlayVariant"

    static func apply(_ variant: DictationPrototypeVariant, settings: SettingsManager) {
        let defaults = UserDefaults.standard
        if variant == .current {
            if settings.overlayVariantRaw == variantID {
                settings.overlayVariantRaw = defaults.string(forKey: savedKey) ?? "classic"
            }
        } else if settings.overlayVariantRaw != variantID {
            defaults.set(settings.overlayVariantRaw, forKey: savedKey)
            settings.overlayVariantRaw = variantID
        }
    }

    static let variant = OverlayVariant(
        id: variantID, displayName: "Prototype (lab)", placement: .pill
    ) { feed, _ in
        AnyView(LabPillOverlay(feed: feed))
    }
}

// MARK: - Dispatch

extension DictationLab {
    /// ⌃⌥Space pressed: what it means depends on the variant.
    func shortcutDown() {
        switch variant {
        case .fixBar, .justEdit: FixBarFlow.open(self)
        case .wordCards: WordCardsFlow.showCard(self, take: lastTake, linger: .seconds(12))
        case .teach: Task { await TeachFlow.open(self) }
        case .sayItAgain: voice?.begin()
        case .current: break
        }
    }

    func shortcutUp() {
        if variant == .sayItAgain { voice?.end() }
    }

    /// A take just landed: each variant's overlay card.
    func showCommitCard(for take: LabTake) {
        switch variant {
        case .fixBar: FixBarFlow.showHint(self, take: take)
        case .wordCards: WordCardsFlow.showCard(self, take: take, linger: .seconds(8))
        case .teach: TeachFlow.showApplied(self, take: take)
        case .justEdit: JustEditFlow.showWatching(self, take: take)
        case .sayItAgain: SayItAgainFlow.showHint(self, take: take)
        case .current: break
        }
    }

    /// Learned something: the toast card, over any app.
    func showToastCard() {
        guard variant != .current, let toast else { return }
        panels.showCard(
            LabToastCard(lab: self, toast: toast), size: CGSize(width: 460, height: 76),
            duration: .seconds(6))
    }
}
