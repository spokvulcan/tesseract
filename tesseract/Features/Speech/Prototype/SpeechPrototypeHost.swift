//
//  SpeechPrototypeHost.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  Five structurally different Speech pages on the existing Speech route,
//  switched by the floating bar in the toolbar (or ← → with no text field
//  focused), plus today's page for comparison. The choice survives a
//  relaunch. Each variant owns its whole layout; they share only the lab
//  (text, takes, voice, overlay) so flipping between them keeps your work.
//

import SwiftUI

enum SpeechPrototypeVariant: Int, CaseIterable {
    case studio, reader, voiceLab, takes, feed, current

    var label: String {
        switch self {
        case .studio: "A · Studio"
        case .reader: "B · Reader"
        case .voiceLab: "C · Voice Lab"
        case .takes: "D · Takes"
        case .feed: "E · Feed"
        case .current: "Current page"
        }
    }
}

struct SpeechPrototypeHost<Current: View>: View {
    @ViewBuilder let current: () -> Current
    @AppStorage("speechPrototype.variant") private var variantIndex = 0

    var body: some View {
        let variant = SpeechPrototypeVariant(rawValue: variantIndex) ?? .studio
        Group {
            switch variant {
            case .studio: StudioVariant()
            case .reader: ReaderVariant()
            case .voiceLab: VoiceLabVariant()
            case .takes: TakesVariant()
            case .feed: FeedVariant()
            case .current: current()
            }
        }
        .id(variant)
        .navigationTitle("Speech")
        .toolbar {
            ToolbarItem(placement: .principal) {
                PrototypeSwitcher(
                    labels: SpeechPrototypeVariant.allCases.map(\.label), index: $variantIndex)
            }
            .sharedBackgroundVisibility(.hidden)
        }
    }
}
