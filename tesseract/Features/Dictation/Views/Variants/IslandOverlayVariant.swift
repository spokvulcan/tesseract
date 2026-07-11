//
//  IslandOverlayVariant.swift
//  tesseract
//

import AppKit
import SwiftUI

// MARK: - Metrics

/// The island variant's sizes — same contract as ``PillMetrics``: per-stage
/// sizes are *content layout* inside the fixed ``canvasSize`` panel; SwiftUI
/// springs the capsule between them and the window never moves (map #283).
///
/// `nonisolated` so ``OverlayPlacement/island`` (and placement tests) can read
/// the canvas off the main actor.
nonisolated enum IslandMetrics {
    /// Compact capsule while recording (dot + elapsed + mini spectrum).
    static let recordingSize = CGSize(width: 180, height: 30)
    /// The compact capsule breathes up to `breatheSteps * breatheStep` pt
    /// wider with loudness — quantized so the glass re-lays-out at bucket
    /// crossings, never at meter cadence.
    static let breatheStep: CGFloat = 4
    static let breatheSteps = 2
    /// The slightly-wider morph for the thinking stages (dots / wand).
    static let processingSize = CGSize(width: 208, height: 32)
    /// The expanded island for a lingering committed take: the text line
    /// plus the chips-and-affordances row.
    static let committedSize = CGSize(width: 380, height: 64)
    /// The expanded island for a lingering rejected take: one row.
    static let rejectedSize = CGSize(width: 380, height: 44)
    /// The expanded island for an error: description plus recovery line.
    static let errorSize = CGSize(width: 380, height: 68)

    /// The fixed panel canvas: fits the widest island plus breathing and
    /// antialiasing headroom. Content is top-anchored inside it, so every
    /// island size hangs from the same top edge.
    static let canvasSize = CGSize(width: 420, height: 120)
    /// The canvas hangs this far below the visible frame's top (menu bar).
    static let topInset: CGFloat = 8
}

// MARK: - Placement

nonisolated extension OverlayPlacement {
    /// The island's canvas: centred horizontally, its top edge a fixed inset
    /// below the visible frame's top — the capsule hangs just under the menu
    /// bar, the macOS notch fantasy. Pure rect math, off-main-testable.
    static let island = OverlayPlacement(
        frame: { geometry in
            let size = IslandMetrics.canvasSize
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.midX - size.width / 2,
                y: visible.maxY - size.height - IslandMetrics.topInset,
                width: size.width,
                height: size.height
            )
        }
    )
}

// MARK: - View

/// The **island** Overlay Variant (map #283): a Dynamic-Island-style capsule
/// top-centre under the menu bar. Unlike the classic pill's view swaps, ONE
/// Liquid Glass capsule persists across every stage and morphs its frame —
/// compact meter while recording, a thinking pill while processing and
/// proofreading, a wide island for lingering beats and errors. One spring,
/// one glass surface, zero window motion.
///
/// Same glass discipline as `GlobalOverlayHUD`: pure untinted `.regular`
/// glass; state reads through vibrant content (red meters, green polish,
/// orange trouble), never through a tint.
struct IslandOverlayView: View {
    /// The Overlay Feed — the only pipeline surface this view sees.
    var feed: DictationFeed
    /// The overlay action surface (ticket #289) — invoked only from the
    /// lingering-beat affordances, the panel's sole clickable window.
    var actions: OverlayActions = .none

    /// Phase/beat mirrors of the feed, mutated inside `withAnimation` so the
    /// capsule frame, the content crossfade, and mount/unmount all ride the
    /// one morph spring.
    @State private var shownPhase: DictationFeed.Phase = .idle
    @State private var shownBeat: DictationFeed.Beat?
    /// Quantized loudness (0…`IslandMetrics.breatheSteps`) reported by the
    /// recording dot — the only meter signal allowed to touch the capsule
    /// frame, and only at bucket crossings.
    @State private var breatheBucket = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The one spring system — every island morph rides this.
    private static let morphSpring: Animation = .spring(response: 0.25, dampingFraction: 0.8)

    /// Reduce Motion path: crossfade only — a short ease so content fades
    /// while the frame settles gently (entrances/exits are pure `.opacity`).
    private var stateAnimation: Animation {
        reduceMotion ? .easeInOut(duration: 0.15) : Self.morphSpring
    }

    var body: some View {
        GlassEffectContainer {
            ZStack(alignment: .top) {
                if let stage {
                    islandCapsule(for: stage)
                        .transition(
                            reduceMotion
                                ? .opacity
                                : .scale(scale: 0.8, anchor: .top).combined(with: .opacity)
                        )
                }
            }
            .frame(
                width: IslandMetrics.canvasSize.width,
                height: IslandMetrics.canvasSize.height,
                alignment: .top
            )
        }
        .onChange(of: feed.phase) { _, newPhase in
            apply(newPhase)
        }
        .onChange(of: feed.beat) { _, beat in
            applyBeat(beat)
        }
        .onAppear {
            apply(feed.phase, animated: false)
        }
    }

    // MARK: - Stage machine

    /// What the island renders right now — one value so live phases and
    /// lingering beats share a single capsule identity (a morph between
    /// sizes, never a swap between two glass surfaces).
    private enum Stage: Equatable {
        case recording
        case processing
        case proofreading
        case error(DictationError)
        case committed(text: String, edits: [WordEdit])
        case rejected
    }

    private var stage: Stage? {
        switch shownPhase {
        case .recording: return .recording
        case .processing: return .processing
        case .proofreading: return .proofreading
        case .error(let error): return .error(error)
        case .idle:
            switch shownBeat?.outcome {
            case .committed(let text, _, let edits):
                return .committed(text: text, edits: edits)
            case .rejected:
                return .rejected
            case .empty, .cancelled, .superseded, nil:
                return nil
            }
        }
    }

    private func apply(_ newPhase: DictationFeed.Phase, animated: Bool = true) {
        guard shownPhase != newPhase else { return }
        withAnimation(animated ? stateAnimation : nil) {
            // A live phase always outranks a lingering beat, and any hop away
            // from recording rests the breathing width for the next take.
            if newPhase != .idle { shownBeat = nil }
            if newPhase != .recording { breatheBucket = 0 }
            shownPhase = newPhase
        }
    }

    private func applyBeat(_ beat: DictationFeed.Beat?) {
        guard let beat else { return }
        switch beat.outcome {
        case .committed, .rejected:
            break
        case .empty, .cancelled, .superseded:
            return
        }
        withAnimation(stateAnimation) {
            shownBeat = beat
        }
        Task {
            try? await Task.sleep(for: DictationFeed.affordanceLinger)
            dismissBeat(ifStill: beat.id)
        }
    }

    private func dismissBeat(ifStill id: UInt64? = nil) {
        if let id, shownBeat?.id != id { return }
        withAnimation(stateAnimation) {
            shownBeat = nil
        }
    }

    // MARK: - The capsule

    /// The one glass capsule. Its *frame* morphs between the per-stage sizes
    /// (the panel is fixed, so this is pure content layout); the ZStack
    /// overlays outgoing and incoming stage content, so stage hops read as a
    /// crossfade inside the stretching glass.
    private func islandCapsule(for stage: Stage) -> some View {
        let size = size(for: stage)
        return ZStack {
            stageContent(for: stage)
        }
        .frame(width: size.width, height: size.height)
        .glassEffect(.regular, in: .capsule)
    }

    private func size(for stage: Stage) -> CGSize {
        switch stage {
        case .recording:
            return CGSize(
                width: IslandMetrics.recordingSize.width
                    + CGFloat(breatheBucket) * IslandMetrics.breatheStep,
                height: IslandMetrics.recordingSize.height)
        case .processing, .proofreading:
            return IslandMetrics.processingSize
        case .committed:
            return IslandMetrics.committedSize
        case .rejected:
            return IslandMetrics.rejectedSize
        case .error:
            return IslandMetrics.errorSize
        }
    }

    @ViewBuilder
    private func stageContent(for stage: Stage) -> some View {
        switch stage {
        case .recording:
            recordingContent
        case .processing:
            processingContent
        case .proofreading:
            proofreadingContent
        case .error(let error):
            errorContent(for: error)
        case .committed(let text, let edits):
            committedContent(text: text, edits: edits)
        case .rejected:
            rejectedContent
        }
    }

    // MARK: - Stage content

    /// Compact recording capsule: pulsing dot, elapsed time, mini spectrum.
    /// The dot and spectrum read the feed themselves, so meter cadence
    /// (~47 Hz) invalidates only those leaves — this row, the capsule frame,
    /// and the glass re-diff only at breathe-bucket crossings.
    private var recordingContent: some View {
        HStack(spacing: 8) {
            RecordingPulseDot(feed: feed) { bucket in
                guard !reduceMotion else { return }
                withAnimation(Self.morphSpring) { breatheBucket = bucket }
            }
            Spacer(minLength: 4)
            ElapsedTimeView(feed: feed)
            Spacer(minLength: 4)
            MiniSpectrumView(feed: feed)
        }
        .padding(.horizontal, 12)
        .accessibilityLabel("Recording")
    }

    /// Thinking stage: the dots ride inside the TimelineView, so the 60 fps
    /// closure re-evaluates only the dots row — never the capsule or glass.
    private var processingContent: some View {
        Group {
            if reduceMotion {
                ProgressView()
                    .controlSize(.small)
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                    ProcessingDotsView(time: timeline.date.timeIntervalSinceReferenceDate)
                        .frame(height: 12)
                }
            }
        }
        .accessibilityLabel("Transcribing")
    }

    /// The Proofread Pass, narrated (map #283): a twinkling wand and its
    /// name — unmistakably different from the transcription dots.
    private var proofreadingContent: some View {
        HStack(spacing: 7) {
            Image(systemName: "wand.and.sparkles")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
                .symbolEffect(.variableColor.iterative, isActive: !reduceMotion)
            Text("Polishing…")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
        }
        .accessibilityLabel("Proofreading")
    }

    /// Errors get the full island: what went wrong, and how to fix it when
    /// the error knows. Read-only — the panel is click-through outside the
    /// beat linger, so an error never carries affordances.
    private func errorContent(for error: DictationError) -> some View {
        let message = error.errorDescription ?? "Something went wrong."
        let suggestion = error.recoverySuggestion
        return HStack(alignment: .center, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text(message)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(suggestion == nil ? 2 : 1)
                    .minimumScaleFactor(0.9)
                if let suggestion {
                    Text(suggestion)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            .multilineTextAlignment(.leading)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 22)
        .accessibilityLabel(suggestion.map { "\(message) \($0)" } ?? message)
    }

    /// The expanded committed island (ticket #289): the take's text on top,
    /// the Proofread Pass's polish (word-swap chips) and the flywheel
    /// affordances below. The panel is clickable for exactly this linger.
    private func committedContent(text: String, edits: [WordEdit]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 7) {
                Image(systemName: edits.isEmpty ? "checkmark" : "wand.and.sparkles")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.green)
                Text(text)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 0)
            }
            HStack(spacing: 6) {
                if edits.isEmpty {
                    Text("Inserted")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)
                } else {
                    // Show the polish, don't just count it: up to two swaps
                    // as chips, the rest folded into a +n.
                    ForEach(Array(edits.prefix(2).enumerated()), id: \.offset) { _, edit in
                        editChip(edit)
                    }
                    if edits.count > 2 {
                        Text("+\(edits.count - 2)")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 8)
                affordance("flag", label: "Flag as wrong") {
                    actions.flagLastTakeWrong()
                    dismissBeat()
                }
                affordance("pencil", label: "Edit in history") {
                    actions.editLastTake()
                    dismissBeat()
                }
            }
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 9)
        .accessibilityLabel(
            edits.isEmpty
                ? "Inserted: \(text)"
                : "Inserted, proofreader polished \(edits.count) words: \(text)")
    }

    /// The lingering rejected island. Passive-first: the press is the retry;
    /// the one affordance inserts the raw take anyway (which also flags the
    /// pass as wrong).
    private var rejectedContent: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.uturn.backward")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.orange)
            Text("Didn't catch that — hold to retry")
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
            Spacer(minLength: 8)
            Button {
                actions.insertRawAnyway()
                dismissBeat()
            } label: {
                Text("Insert anyway")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Insert the raw transcription anyway")
        }
        .padding(.horizontal, 22)
        .accessibilityLabel("Transcription rejected — hold the hotkey to retry")
    }

    // MARK: - Pieces

    /// One word-swap chip ("peace → piece"). Insertions read "+ word",
    /// deletions strike the word through; long words truncate inside a
    /// capped chip so two always fit beside the affordances.
    private func editChip(_ edit: WordEdit) -> some View {
        HStack(spacing: 3) {
            if edit.original.isEmpty {
                Text("+ \(edit.replacement)")
                    .foregroundStyle(.primary)
            } else if edit.replacement.isEmpty {
                Text(edit.original)
                    .strikethrough()
                    .foregroundStyle(.secondary)
            } else {
                Text(edit.original)
                    .foregroundStyle(.secondary)
                Image(systemName: "arrow.right")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundStyle(.tertiary)
                Text(edit.replacement)
                    .foregroundStyle(.primary)
            }
        }
        .font(.system(size: 10, weight: .medium))
        .lineLimit(1)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(.quaternary, in: .capsule)
        .frame(maxWidth: 130)
        .accessibilityLabel(chipLabel(for: edit))
    }

    private func chipLabel(for edit: WordEdit) -> String {
        if edit.original.isEmpty { return "Added \(edit.replacement)" }
        if edit.replacement.isEmpty { return "Removed \(edit.original)" }
        return "\(edit.original) changed to \(edit.replacement)"
    }

    /// One icon affordance on the lingering island: plain style (the glass
    /// is the chrome), secondary ink, a generous circular hit target.
    private func affordance(
        _ systemName: String, label: String, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 24, height: 24)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

// MARK: - Meter leaves

/// The pulsing record dot — the one place raw `level` moves geometry. Reads
/// the feed itself so meter cadence invalidates only this leaf; reports the
/// *quantized* loudness bucket upward so the capsule's breathing (a glass
/// re-layout) changes only at bucket crossings.
private struct RecordingPulseDot: View {
    var feed: DictationFeed
    /// Called on bucket crossings with 0…`IslandMetrics.breatheSteps`.
    let onLevelBucket: @MainActor (Int) -> Void

    @State private var lastBucket = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let level = CGFloat(min(max(feed.level, 0), 1))
        Circle()
            .fill(.red)
            .frame(width: 9, height: 9)
            .scaleEffect(reduceMotion ? 1 : 1 + level * 0.35)
            .animation(reduceMotion ? nil : .linear(duration: 0.05), value: level)
            .onChange(of: feed.level) { _, newLevel in
                let bucket = min(
                    IslandMetrics.breatheSteps,
                    Int(newLevel * Float(IslandMetrics.breatheSteps + 1)))
                guard bucket != lastBucket else { return }
                lastBucket = bucket
                onLevelBucket(bucket)
            }
            .accessibilityHidden(true)
    }
}

/// The compact live spectrum: `MeterFrame.bandCount` capsule bars growing
/// from the vertical centre. Reads the feed itself (meter cadence stays in
/// this leaf); height and brightness both track the band, so quiet reads
/// dim, not empty.
private struct MiniSpectrumView: View {
    var feed: DictationFeed
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let barWidth: CGFloat = 2.5
    private let barSpacing: CGFloat = 2
    private let maxBarHeight: CGFloat = 14
    private let minBarHeight: CGFloat = 3

    var body: some View {
        let bands = feed.spectrum
        HStack(alignment: .center, spacing: barSpacing) {
            ForEach(0..<MeterFrame.bandCount, id: \.self) { index in
                let value = index < bands.count ? CGFloat(min(max(bands[index], 0), 1)) : 0
                Capsule()
                    .fill(.red.opacity(0.45 + 0.55 * value))
                    .frame(
                        width: barWidth,
                        height: minBarHeight + (maxBarHeight - minBarHeight) * value)
            }
        }
        // Data display, so the bars keep moving under Reduce Motion — only
        // the interpolation between frames is dropped.
        .animation(reduceMotion ? nil : .linear(duration: 0.06), value: bands)
        .accessibilityHidden(true)
    }
}

/// Elapsed take time, mm:ss. The periodic timeline is aligned to the take's
/// start, so this leaf re-evaluates exactly once a second — never at meter
/// cadence.
private struct ElapsedTimeView: View {
    var feed: DictationFeed

    var body: some View {
        // recordingStarted is set once per take; nil only in mount races,
        // where 00:00 is the right thing to show anyway.
        let start = feed.recordingStarted ?? Date()
        TimelineView(.periodic(from: start, by: 1)) { context in
            Text(Self.label(from: start, to: context.date))
                .font(.system(size: 11, weight: .semibold))
                .monospacedDigit()
        }
    }

    private static func label(from start: Date, to now: Date) -> String {
        let seconds = max(0, Int(now.timeIntervalSince(start).rounded()))
        return String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}

// MARK: - Registration

extension OverlayVariants {
    /// The top-centre dynamic island (map #283).
    static let island = OverlayVariant(
        id: "island",
        displayName: "Island",
        placement: .island
    ) { feed, actions in
        AnyView(IslandOverlayView(feed: feed, actions: actions))
    }
}

// MARK: - Previews

private extension View {
    /// Busy backdrop for judging glass legibility in previews.
    func islandPreviewBackdrop() -> some View {
        padding(30)
            .background(
                LinearGradient(
                    colors: [.blue, .purple, .orange],
                    startPoint: .topLeading, endPoint: .bottomTrailing
                ))
    }
}

#Preview("Recording") {
    let feed = DictationFeed()
    feed.setPhase(.recording)
    return IslandOverlayView(feed: feed)
        .islandPreviewBackdrop()
        .task {
            // Synthesized speech-shaped meters so the dot, the breathing,
            // and the spectrum all move in canvas.
            var tick = 0.0
            while !Task.isCancelled {
                tick += 1.0 / 30.0
                let level = 0.45 + 0.3 * sin(tick * 2.2) + 0.2 * sin(tick * 5.7)
                let bands = (0..<MeterFrame.bandCount).map { band -> Float in
                    let phase = Double(band)
                    let wave: Double = 0.5 + 0.45 * sin(tick * (1.3 + phase * 0.6) + phase)
                    return Float(max(0, wave))
                }
                feed.apply(MeterFrame(level: Float(min(max(level, 0), 1)), bands: bands))
                try? await Task.sleep(for: .milliseconds(33))
            }
        }
}

#Preview("Processing") {
    let feed = DictationFeed()
    feed.setPhase(.processing)
    return IslandOverlayView(feed: feed)
        .islandPreviewBackdrop()
}

#Preview("Proofreading") {
    let feed = DictationFeed()
    feed.setPhase(.proofreading)
    return IslandOverlayView(feed: feed)
        .islandPreviewBackdrop()
}

#Preview("Committed + edits") {
    let feed = DictationFeed()
    return IslandOverlayView(feed: feed)
        .islandPreviewBackdrop()
        .task {
            // Re-emit past the linger so the island replays while designing.
            while !Task.isCancelled {
                feed.emit(
                    .committed(
                        text: "Let's grab a piece of the market before their launch.",
                        duration: 3.8,
                        edits: [
                            WordEdit(original: "peace", replacement: "piece"),
                            WordEdit(original: "there", replacement: "their"),
                        ]))
                try? await Task.sleep(for: .seconds(4))
            }
        }
}

#Preview("Rejected") {
    let feed = DictationFeed()
    return IslandOverlayView(feed: feed)
        .islandPreviewBackdrop()
        .task {
            while !Task.isCancelled {
                feed.emit(
                    .rejected(
                        raw: "uh so um the thing with the stuff",
                        reason: "Unintelligible transcription"))
                try? await Task.sleep(for: .seconds(4))
            }
        }
}

#Preview("Error") {
    let feed = DictationFeed()
    feed.setPhase(.error(.microphonePermissionDenied))
    return IslandOverlayView(feed: feed)
        .islandPreviewBackdrop()
}
