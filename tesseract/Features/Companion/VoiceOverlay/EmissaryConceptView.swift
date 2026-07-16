//
//  EmissaryConceptView.swift
//  tesseract
//

import SwiftUI

// MARK: - Emissary

/// PROTOTYPE (map #301, ticket #328) — Concept 2: **Emissary**.
///
/// The Companion as a *physical presence*: a small glass sphere that
/// condenses exactly where notification banners land — the summons replaces
/// the banner with something that can knock. Unanswered, it knocks harder
/// and grows. Engaging unfolds the sphere into a **conversation card**: an
/// orb face on top, the running exchange beneath it, your live words and a
/// waveform at the bottom while you talk. The form argues the opposite of
/// Horizon: a voice conversation deserves a *body* and a visible transcript,
/// not just a line of light.
struct EmissaryConceptView: View {
    var feed: CompanionVoiceFeed
    var actions: CompanionVoiceActions = .none

    @Namespace private var glassSpace
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack(alignment: .topTrailing) {
            GlassEffectContainer(spacing: 20) {
                switch feed.state {
                case .idle:
                    EmptyView()
                case .summoning(let escalation):
                    knockingSphere(escalation: escalation)
                case .listening, .thinking, .speaking:
                    conversationCard
                }
            }
            .padding(.top, 16)
            .padding(.trailing, 16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
    }

    // MARK: - The knock

    private func knockingSphere(escalation: Int) -> some View {
        let diameter: CGFloat = [56, 64, 72][min(escalation, 2)]
        let period: Double = [3.0, 2.2, 1.5][min(escalation, 2)]
        return HStack(spacing: 12) {
            Text(feed.summonsLine)
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .lineLimit(2)
                .multilineTextAlignment(.trailing)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .glassEffect(.regular, in: .rect(cornerRadius: 14))
                .transition(.move(edge: .trailing).combined(with: .opacity))

            TimelineView(.animation(paused: reduceMotion)) { context in
                let t = context.date.timeIntervalSinceReferenceDate
                sphereFace(diameter: diameter)
                    .offset(x: knockOffset(time: t, period: period))
            }
        }
        .padding(4)
        .contentShape(Rectangle())
        .onTapGesture { actions.engage() }
        .transition(.scale(scale: 0.4).combined(with: .opacity))
    }

    /// Two quick raps, then quiet, on a fixed cycle — a knock, not a wobble.
    private func knockOffset(time: TimeInterval, period: Double) -> CGFloat {
        let phase = time.truncatingRemainder(dividingBy: period)
        guard phase < 0.5 else { return 0 }
        return CGFloat(-6 * sin(phase * .pi * 4) * (1 - phase / 0.5))
    }

    private func sphereFace(diameter: CGFloat) -> some View {
        ZStack {
            Circle()
                .fill(.clear)
                .glassEffect(.regular.interactive(), in: .circle)
                .glassEffectID("emissary", in: glassSpace)
            Image(systemName: "waveform")
                .font(.system(size: diameter * 0.32, weight: .medium))
                .foregroundStyle(.secondary)
                .symbolEffect(
                    .variableColor.iterative, options: .repeating, isActive: !reduceMotion)
        }
        .frame(width: diameter, height: diameter)
    }

    // MARK: - The card

    private var conversationCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            cardHeader
            if let contract = feed.contract {
                EmissaryContractChip(text: contract)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
            transcriptBlock
            liveBlock
        }
        .padding(16)
        .frame(width: 380, alignment: .leading)
        .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 24))
        .glassEffectID("emissary", in: glassSpace)
        .transition(.scale(scale: 0.85, anchor: .topTrailing).combined(with: .opacity))
    }

    private var cardHeader: some View {
        HStack(spacing: 10) {
            orbFace
            Text(feed.sceneTitle)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                actions.openChat()
            } label: {
                Image(systemName: "bubble.left.and.text.bubble.right")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
            .help("Continue in chat")
            Button {
                actions.dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
    }

    /// The face keeps state legible at a glance even mid-scroll: a ring that
    /// breathes with whoever is talking, a slow orbit while it thinks.
    private var orbFace: some View {
        ZStack {
            Circle()
                .strokeBorder(.secondary.opacity(0.6), lineWidth: 1.5)
            switch feed.state {
            case .thinking:
                TimelineView(.animation(paused: reduceMotion)) { context in
                    let t = context.date.timeIntervalSinceReferenceDate
                    Circle()
                        .fill(.primary)
                        .frame(width: 4, height: 4)
                        .offset(
                            x: 9 * cos(t * 2.6),
                            y: 9 * sin(t * 2.6))
                }
            default:
                Circle()
                    .fill(.primary.opacity(0.85))
                    .frame(
                        width: 8 + CGFloat(feed.level) * 14,
                        height: 8 + CGFloat(feed.level) * 14
                    )
                    .animation(.linear(duration: 0.08), value: feed.level)
            }
        }
        .frame(width: 30, height: 30)
    }

    private var transcriptBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(visibleTranscript) { line in
                transcriptRow(line)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// The last few settled exchanges, oldest dimmest — the card shows a
    /// *conversation*, not a status.
    private var visibleTranscript: [CompanionVoiceFeed.Line] {
        Array(feed.transcript.suffix(4))
    }

    private func transcriptRow(_ line: CompanionVoiceFeed.Line) -> some View {
        let index = visibleTranscript.firstIndex(of: line) ?? 0
        let age = visibleTranscript.count - 1 - index
        return HStack {
            if line.role == .owner { Spacer(minLength: 32) }
            Text(line.text)
                .font(.system(size: 12.5, design: .rounded))
                .foregroundStyle(
                    line.role == .companion ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary)
                )
                .multilineTextAlignment(line.role == .companion ? .leading : .trailing)
            if line.role == .companion { Spacer(minLength: 32) }
        }
        .opacity(1 - Double(age) * 0.18)
    }

    // MARK: - The live line

    @ViewBuilder
    private var liveBlock: some View {
        switch feed.state {
        case .speaking:
            speakingLine
        case .listening:
            listeningWell
        case .thinking:
            HStack(spacing: 5) {
                ForEach(0..<3, id: \.self) { index in
                    Circle()
                        .fill(.tertiary)
                        .frame(width: 5, height: 5)
                        .breathing(index: index, paused: reduceMotion)
                }
                Spacer()
            }
            .padding(.top, 2)
        default:
            EmptyView()
        }
    }

    private var speakingLine: some View {
        let words = feed.spokenWords.prefix(feed.revealedWordCount)
        let settled = words.dropLast().joined(separator: " ")
        let newest = words.last ?? ""
        return
            (Text(settled.isEmpty ? "" : settled + " ").foregroundStyle(.secondary)
            + Text(newest).foregroundStyle(.primary))
            .font(.system(size: 13.5, weight: .medium, design: .rounded))
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .onTapGesture { actions.bargeIn() }
    }

    private var listeningWell: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(feed.partial ?? "Listening…")
                .font(.system(size: 13, design: .rounded))
                .italic(feed.partial == nil)
                .foregroundStyle(feed.partial == nil ? .tertiary : .secondary)
                .frame(maxWidth: .infinity, alignment: .trailing)
                .multilineTextAlignment(.trailing)
            HStack(spacing: 3) {
                Spacer()
                ForEach(feed.spectrum.indices, id: \.self) { index in
                    Capsule()
                        .fill(.secondary)
                        .frame(width: 3, height: 4 + CGFloat(feed.spectrum[index]) * 18)
                }
            }
            .animation(.linear(duration: 0.08), value: feed.spectrum)
        }
    }
}

// MARK: - Pieces

private struct EmissaryContractChip: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 12, weight: .semibold))
            Text(text)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .lineLimit(2)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 10))
    }
}

extension View {
    /// A staggered breathing pulse for the thinking dots.
    fileprivate func breathing(index: Int, paused: Bool) -> some View {
        modifier(BreathingDot(delay: Double(index) * 0.2, paused: paused))
    }
}

private struct BreathingDot: ViewModifier {
    let delay: Double
    let paused: Bool

    func body(content: Content) -> some View {
        TimelineView(.animation(paused: paused)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            content
                .opacity(0.4 + 0.6 * max(0, sin((t - delay) * 3)))
        }
    }
}
