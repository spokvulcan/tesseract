//
//  ProsceniumConceptView.swift
//  tesseract
//

import AppKit
import SwiftUI

// MARK: - Proscenium

/// PROTOTYPE (map #301, ticket #328) — Concept 3: **Proscenium**.
///
/// The Companion on *stage*: the notch grows a glass lip that swells
/// downward for a summons, then opens into a wide teleprompter stage when
/// engaged — the Companion's words play across it large and word-lit, your
/// turns flip the stage lighting to a waveform and your own words, thinking
/// narrows the whole stage to a shimmering sliver. This is the TTS notch's
/// real estate absorbed into the new design language (the ticket's explicit
/// sub-question): if this concept wins, `TTSNotchOverlayView` retires into
/// it. The form argues that the voice already has a home on this screen —
/// the top center — and the Companion should inherit it.
struct ProsceniumConceptView: View {
    var feed: CompanionVoiceFeed
    var actions: CompanionVoiceActions = .none

    @Namespace private var glassSpace
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The menu-bar band the island hangs through — resolved once per
    /// mount; the demo never migrates screens mid-scene (prototype).
    private let menuBarHeight: CGFloat = {
        guard let screen = OverlayScreenLocator.preferredScreen() else { return 38 }
        return screen.frame.maxY - screen.visibleFrame.maxY
    }()

    var body: some View {
        ZStack(alignment: .top) {
            GlassEffectContainer(spacing: 18) {
                if feed.isActive {
                    stage
                        .transition(
                            .asymmetric(
                                insertion: .move(edge: .top).combined(with: .opacity),
                                removal: .move(edge: .top).combined(with: .opacity)))
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    // MARK: - The stage

    private var stage: some View {
        VStack(spacing: 0) {
            // The island: menu-bar band + the visible lip below it. All state
            // motion is this one shape resizing — the morph IS the concept.
            VStack(spacing: 0) {
                Color.clear.frame(height: menuBarHeight)
                stageContent
                    .frame(height: lipHeight)
                    .frame(maxWidth: .infinity)
            }
            .frame(width: stageWidth)
            .glassEffect(
                .regular.interactive(),
                in: DynamicIslandShape(topInset: 14, bottomRadius: 22)
            )
            .glassEffectID("stage", in: glassSpace)
            .contentShape(DynamicIslandShape(topInset: 14, bottomRadius: 22))
            .onTapGesture { handleTap() }

            if let contract = feed.contract {
                ProsceniumContractChip(text: contract)
                    .padding(.top, 10)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.spring(duration: 0.55, bounce: 0.18), value: stageWidth)
        .animation(.spring(duration: 0.55, bounce: 0.18), value: lipHeight)
    }

    private var stageWidth: CGFloat {
        switch feed.state {
        case .idle: 220
        case .summoning(let escalation): 300 + CGFloat(escalation) * 40
        case .thinking: 260
        case .listening: 620
        case .speaking: 700
        }
    }

    private var lipHeight: CGFloat {
        switch feed.state {
        case .idle: 0
        case .summoning(let escalation): 34 + CGFloat(escalation) * 12
        case .thinking: 26
        case .listening: 96
        case .speaking: 118
        }
    }

    @ViewBuilder
    private var stageContent: some View {
        switch feed.state {
        case .summoning(let escalation):
            summonsLip(escalation: escalation)
        case .thinking:
            ProsceniumShimmer(paused: reduceMotion)
                .padding(.horizontal, 28)
        case .speaking:
            speakingStage
        case .listening:
            listeningStage
        default:
            EmptyView()
        }
    }

    // MARK: - Faces

    private func summonsLip(escalation: Int) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "theatermasks.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
                .symbolEffect(
                    .pulse, options: .repeating.speed(escalation > 0 ? 1.4 : 0.6),
                    isActive: !reduceMotion)
            Text(feed.summonsLine)
                .font(.system(size: 12.5 + CGFloat(escalation) * 0.5, weight: .medium))
                .lineLimit(1)
        }
        .padding(.horizontal, 20)
    }

    /// The teleprompter: words arrive large and center-stage; the light sits
    /// on the newest ones while the said dims behind them, and a hairline
    /// tracks progress along the bottom lip.
    private var speakingStage: some View {
        VStack(spacing: 10) {
            wordLitLine
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.horizontal, 32)
            Capsule()
                .fill(.secondary)
                .frame(width: progressWidth, height: 2)
                .frame(maxWidth: 320, alignment: .leading)
                .animation(.linear(duration: 0.25), value: progressWidth)
            stageRail
        }
        .padding(.vertical, 10)
    }

    private var wordLitLine: some View {
        let words = feed.spokenWords.prefix(feed.revealedWordCount)
        let dimmed = words.dropLast(3).joined(separator: " ")
        let lit = words.suffix(3).joined(separator: " ")
        return
            (Text(dimmed.isEmpty ? "" : dimmed + " ").foregroundStyle(.tertiary)
            + Text(lit).foregroundStyle(.primary))
            .font(.system(size: 19, weight: .semibold))
            .multilineTextAlignment(.center)
            .lineLimit(3)
    }

    private var progressWidth: CGFloat {
        guard !feed.spokenWords.isEmpty else { return 0 }
        return 320 * CGFloat(feed.revealedWordCount) / CGFloat(feed.spokenWords.count)
    }

    /// Your turn: the stage lighting flips — waveform takes center, your
    /// words build underneath in a quieter register.
    private var listeningStage: some View {
        VStack(spacing: 8) {
            HStack(spacing: 3.5) {
                ForEach(feed.spectrum.indices, id: \.self) { index in
                    Capsule()
                        .fill(.primary.opacity(0.8))
                        .frame(width: 3.5, height: 6 + CGFloat(feed.spectrum[index]) * 26)
                }
            }
            .animation(.linear(duration: 0.08), value: feed.spectrum)
            Text(feed.partial ?? "Your stage…")
                .font(.system(size: 13))
                .foregroundStyle(feed.partial == nil ? .tertiary : .secondary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            stageRail
        }
        .padding(.vertical, 10)
    }

    /// The quiet affordance rail on the bottom lip: hand-off left, dismissal
    /// right, both ghosted so the stage stays the show.
    private var stageRail: some View {
        HStack {
            Button {
                actions.openChat()
            } label: {
                Image(systemName: "text.bubble")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.quaternary)
            }
            .buttonStyle(.plain)
            .help("Continue in chat")
            Spacer()
            Button {
                actions.dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.quaternary)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
        .padding(.horizontal, 24)
    }

    private func handleTap() {
        switch feed.state {
        case .summoning: actions.engage()
        case .speaking: actions.bargeIn()
        default: break
        }
    }
}

// MARK: - Pieces

/// The thinking sliver: a light travelling the narrowed stage.
private struct ProsceniumShimmer: View {
    let paused: Bool

    var body: some View {
        TimelineView(.animation(paused: paused)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            GeometryReader { proxy in
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [.clear, .white.opacity(0.75), .clear],
                            startPoint: .leading, endPoint: .trailing)
                    )
                    .frame(width: 48, height: 3)
                    .position(
                        x: proxy.size.width * (0.5 + 0.38 * sin(t * 2.2)),
                        y: proxy.size.height / 2)
            }
        }
        .frame(height: 12)
    }
}

/// The contract, stamped below the stage like a playbill line.
private struct ProsceniumContractChip: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(text)
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(1)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .glassEffect(.regular, in: .capsule)
    }
}
