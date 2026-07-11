//
//  WhisperOverlayVariant.swift
//  tesseract
//

import AppKit
import SwiftUI

// MARK: - Metrics

/// The whisper variant's sizes — the underline, the sweep stage, and the
/// linger chips, all inside one fixed canvas.
///
/// Per-state sizes are *content layout* consumed by `WhisperOverlayView`; the
/// panel's window is fixed at ``canvasSize`` and never resizes (map #283) —
/// SwiftUI animates the line and chips between these sizes inside that canvas.
///
/// `nonisolated` so it escapes the build's MainActor default isolation — the
/// same contract as `PillMetrics`, which is what lets ``OverlayPlacement/whisper``
/// (and its tests) run off the main actor.
nonisolated enum WhisperMetrics {
    /// The resting hairline: the error flash's width, and the Reduce Motion
    /// recording width.
    static let lineRestWidth: CGFloat = 140
    /// Recording width span — the line undulates between these with `level`.
    static let lineMinWidth: CGFloat = 100
    static let lineMaxWidth: CGFloat = 180
    static let lineHeight: CGFloat = 5

    /// Processing/proofreading: the line contracts to this width…
    static let sweepLineWidth: CGFloat = 60
    /// …and its centre glides this far to each side of the canvas centre.
    static let sweepTravel: CGFloat = 40
    /// One full left-right-left metronome cycle.
    static let sweepPeriod: TimeInterval = 1.8
    /// The sweep stage's own bounds: the sweep extent plus sparkle headroom.
    static let sweepStageSize = CGSize(width: 220, height: 20)

    /// The linger chips (ticket #289): tiny by design — no committed text.
    static let chipHeight: CGFloat = 28
    static let committedChipWidth: CGFloat = 164
    static let rejectedChipWidth: CGFloat = 140
    static let errorChipWidth: CGFloat = 220
    /// Gap between the error chip and the flashing line beneath it.
    static let errorStackSpacing: CGFloat = 6

    /// Content floats this far above the canvas bottom (glass-rim
    /// antialiasing headroom); the line's baseline never moves between states.
    static let bottomPadding: CGFloat = 4

    /// The fixed panel canvas: fits the tallest moment (the error chip
    /// stacked above the line) plus entrance-scale headroom. Content is
    /// bottom-anchored, so the visual bottom inset is constant across states.
    static let canvasSize = CGSize(width: 300, height: 56)
}

// MARK: - Placement

nonisolated extension OverlayPlacement {
    /// The whisper underline's canvas: centred horizontally, hugging the
    /// bottom of the visible frame far tighter than the pill — the line
    /// should read as part of the desktop's floor, not as a floating HUD.
    static let whisper = OverlayPlacement(
        frame: { geometry in
            let size = WhisperMetrics.canvasSize
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.midX - size.width / 2,
                y: visible.minY + whisperBottomInset,
                width: size.width,
                height: size.height
            )
        }
    )

    /// The whisper canvas floats this far above the bottom of the visible
    /// frame.
    private static let whisperBottomInset: CGFloat = 20
}

// MARK: - Ink & glass identities

/// The variant's ink constants. State reads through vibrant CONTENT on pure
/// untinted glass — never through a glass tint (the classic pill's
/// discipline, kept).
private nonisolated enum WhisperInk {
    /// Recording red — the same hot family as `AudioBarsView`'s bars.
    static let recording = Color(red: 0.95, green: 0.3, blue: 0.25)
    /// Polish gold — the proofread sparkles and the committed wand.
    static let gold = Color(red: 0.95, green: 0.78, blue: 0.35)
}

/// Glass identities for the morph (`glassEffectID`): every rendition of the
/// line and the linger chip share ``surface``, so recording → processing →
/// bloom-into-chip is one glass element reshaping, never chrome popping in
/// and out. The error chip is a second, independent element.
private nonisolated enum WhisperGlassID {
    static let surface = "whisper.surface"
    static let errorChip = "whisper.errorChip"
}

// MARK: - View

/// The whisper — the near-invisible **Overlay Variant** (map #283): a thin
/// glass underline just above the bottom of the screen, for users who find
/// any pill intrusive. It communicates through the line's motion and ink
/// alone — no text while dictating, and even the committed linger chip
/// (ticket #289) carries only glyphs; the words already landed in the target
/// app, so repeating them here would be noise.
///
/// The variant owns *all* motion inside the panel's fixed canvas — the panel
/// never fades or resizes. Hot-path work is fenced into leaf subtrees: the
/// recording line reads the feed itself (meter cadence invalidates only that
/// leaf), and the sweep's TimelineView wraps only the moving capsule — the
/// container and chips never re-diff per frame.
struct WhisperOverlayView: View {
    /// The Overlay Feed — the only pipeline surface this view sees.
    var feed: DictationFeed
    /// The overlay action surface (ticket #289) — what the lingering chip's
    /// click affordances invoke. Clicks only; the overlay stays keyboard-free.
    var actions: OverlayActions = .none

    /// The phase the line currently renders — updated inside `withAnimation`,
    /// so stage swaps and the shared-id glass morph ride one spring.
    @State private var shownPhase: DictationFeed.Phase = .idle
    /// A lingering terminal beat (map #283): committed and rejected takes
    /// hold the tiny chip for ``DictationFeed/affordanceLinger`` after the
    /// phase returned to idle. Passive: it collapses on its own; the press is
    /// the retry.
    @State private var shownBeat: DictationFeed.Beat?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// One namespace for the morphing glass surface — the line and the chip
    /// share an id inside the single `GlassEffectContainer`.
    @Namespace private var glassSpace

    var body: some View {
        GlassEffectContainer {
            ZStack(alignment: .bottom) {
                switch shownPhase {
                case .recording:
                    WhisperRecordingLine(feed: feed, glassSpace: glassSpace)
                        .transition(stageTransition)
                case .processing:
                    sweepStage(proofreading: false)
                        .transition(stageTransition)
                case .proofreading:
                    sweepStage(proofreading: true)
                        .transition(stageTransition)
                case .error(let error):
                    WhisperErrorStage(error: error, glassSpace: glassSpace)
                        .transition(stageTransition)
                case .idle:
                    if let shownBeat {
                        beatChip(for: shownBeat)
                            // Keyed per beat so two equal outcomes in a row
                            // still restart the chip (and the wand's glow).
                            .id(shownBeat.id)
                            .transition(stageTransition)
                    }
                }
            }
            .padding(.bottom, WhisperMetrics.bottomPadding)
            .frame(
                width: WhisperMetrics.canvasSize.width,
                height: WhisperMetrics.canvasSize.height,
                alignment: .bottom
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
            // Also seed a pre-existing beat: previews emit before the view
            // attaches; production panels mount with a nil beat, so this is
            // inert there.
            applyBeat(feed.beat)
        }
    }

    // MARK: State application (shared timing discipline with GlobalOverlayHUD)

    private func apply(_ newPhase: DictationFeed.Phase, animated: Bool = true) {
        guard shownPhase != newPhase else { return }
        if newPhase != .idle {
            // A live phase always outranks a lingering beat.
            shownBeat = nil
        }
        if reduceMotion || !animated {
            shownPhase = newPhase
            return
        }
        // Snappy in (the line must read as "listening" within ~100 ms of the
        // press), slightly softer out — the classic pill's timing contract,
        // kept so switching variants never changes the perceived latency.
        let showing = newPhase != .idle
        withAnimation(
            .spring(response: showing ? 0.2 : 0.25, dampingFraction: showing ? 0.75 : 0.8)
        ) {
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
        withAnimation(reduceMotion ? nil : .spring(response: 0.25, dampingFraction: 0.8)) {
            shownBeat = beat
        }
        Task {
            try? await Task.sleep(for: DictationFeed.affordanceLinger)
            dismissBeat(ifStill: beat.id)
        }
    }

    /// Collapses the chip back to nothing — on the linger expiring or on any
    /// affordance click.
    private func dismissBeat(ifStill id: UInt64? = nil) {
        if let id, shownBeat?.id != id { return }
        withAnimation(reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.85)) {
            shownBeat = nil
        }
    }

    /// Small and quiet: a slight bottom-anchored settle. The drama of the
    /// bloom lives in the shared-id glass morph, not the transition. Under
    /// Reduce Motion everything is a fade.
    private var stageTransition: AnyTransition {
        reduceMotion
            ? .opacity
            : .scale(scale: 0.85, anchor: .bottom).combined(with: .opacity)
    }

    // MARK: Processing / proofreading — the metronome sweep

    /// True in the instant a committed-with-edits beat has landed but the
    /// phase hasn't returned to idle yet: the still-visible line takes full
    /// polish gold — "the line glows gold" — before blooming into the chip.
    /// If the coordinator flips the phase in the same frame as the beat, the
    /// chip's wand glow carries the same signal instead, so the polish is
    /// surfaced regardless of event ordering.
    private var pendingPolishGold: Bool {
        if case .committed(_, _, let edits) = shownBeat?.outcome, !edits.isEmpty {
            return true
        }
        return false
    }

    /// The contracted gliding line. The TimelineView is scoped tight: only
    /// the moving capsule (and the proofread sparkles) live inside the 60 fps
    /// closure. Under Reduce Motion the sweep is a static line — processing
    /// in neutral ink, proofreading in gold: distinct without movement.
    private func sweepStage(proofreading: Bool) -> some View {
        Group {
            if reduceMotion {
                sweepBody(time: nil, proofreading: proofreading)
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                    sweepBody(
                        time: timeline.date.timeIntervalSinceReferenceDate,
                        proofreading: proofreading)
                }
            }
        }
        .frame(
            width: WhisperMetrics.sweepStageSize.width,
            height: WhisperMetrics.sweepStageSize.height,
            alignment: .bottom
        )
        .accessibilityLabel(proofreading ? "Proofreading transcription" : "Processing")
    }

    private func sweepBody(time: TimeInterval?, proofreading: Bool) -> some View {
        // A pure sine is the metronome — it eases at the extremes for free.
        let sweepPhase = (time ?? 0) * (2 * Double.pi) / WhisperMetrics.sweepPeriod
        let offsetX = time == nil ? 0 : CGFloat(sin(sweepPhase)) * WhisperMetrics.sweepTravel
        return ZStack(alignment: .bottom) {
            if proofreading, let time {
                WhisperSparkleTrail(time: time, sweepPhase: sweepPhase)
            }
            Capsule()
                .fill(sweepInk(proofreading: proofreading))
                .frame(width: WhisperMetrics.sweepLineWidth, height: WhisperMetrics.lineHeight)
                .glassEffect(.regular, in: .capsule)
                .glassEffectID(WhisperGlassID.surface, in: glassSpace)
                .offset(x: offsetX)
        }
    }

    private func sweepInk(proofreading: Bool) -> AnyShapeStyle {
        if pendingPolishGold {
            // The commit-with-edits instant: full gold before the bloom.
            return AnyShapeStyle(WhisperInk.gold)
        }
        if proofreading {
            return AnyShapeStyle(WhisperInk.gold.opacity(0.75))
        }
        return AnyShapeStyle(.secondary)
    }

    // MARK: Linger chips (ticket #289)

    /// The panel is clickable for exactly the beat-linger window (one App
    /// Bindings rule), so these chips are the variant's only interactive
    /// moments — one-click, no focus steal.
    @ViewBuilder
    private func beatChip(for beat: DictationFeed.Beat) -> some View {
        switch beat.outcome {
        case .committed(_, _, let edits):
            WhisperCommittedChip(
                edits: edits, actions: actions, glassSpace: glassSpace,
                dismiss: { dismissBeat() })
        case .rejected:
            rejectedChip
        case .empty, .cancelled, .superseded:
            EmptyView()
        }
    }

    /// The lingering rejected-take chip: an orange dot and the one
    /// affordance. No explanation text — the dot is the whole sentence in
    /// this variant; the accessibility label says the rest.
    private var rejectedChip: some View {
        HStack(alignment: .center, spacing: 8) {
            Circle()
                .fill(.orange)
                .frame(width: 8, height: 8)
            Button {
                actions.insertRawAnyway()
                dismissBeat()
            } label: {
                Text("Insert anyway")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.9)
                    .frame(height: 22)
                    .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Insert the raw transcription anyway")
        }
        .padding(.horizontal, 12)
        .frame(width: WhisperMetrics.rejectedChipWidth, height: WhisperMetrics.chipHeight)
        .glassEffect(.regular.interactive(), in: .capsule)
        .glassEffectID(WhisperGlassID.surface, in: glassSpace)
        .accessibilityLabel("Transcription rejected — hold the hotkey to retry")
    }
}

// MARK: - Recording line

/// The living underline: width springs after `level`, and the spectrum rides
/// the line as a leading→trailing intensity gradient — the eight bands become
/// gradient stops, bars fused into one line.
///
/// Reads the feed directly (not passed as values) so meter-cadence updates
/// invalidate only this leaf — never the container or the chips (the same
/// discipline as `AudioBarsView`).
private struct WhisperRecordingLine: View {
    var feed: DictationFeed
    let glassSpace: Namespace.ID

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// Spring-smoothed width; retargeted every meter frame, so the spring
    /// never settles while speech is live — that *is* the undulation.
    @State private var lineWidth: CGFloat = WhisperMetrics.lineRestWidth

    var body: some View {
        Capsule()
            .fill(spectrumGradient)
            .frame(
                width: reduceMotion ? WhisperMetrics.lineRestWidth : lineWidth,
                height: WhisperMetrics.lineHeight
            )
            .glassEffect(.regular, in: .capsule)
            .glassEffectID(WhisperGlassID.surface, in: glassSpace)
            .onChange(of: feed.level) { _, level in
                retarget(level)
            }
            .onAppear {
                if !reduceMotion { lineWidth = Self.width(for: feed.level) }
            }
            .accessibilityLabel("Recording")
    }

    /// The spectrum as ink: each band becomes a gradient stop's opacity, so
    /// the red intensity travels along the line with the voice. Set directly
    /// at meter cadence (~47 Hz) — no animation stacked on top of the width
    /// spring. This stays live under Reduce Motion: it is the only signal
    /// that the microphone hears anything, and dictation without live
    /// feedback reads as a dead mic.
    private var spectrumGradient: LinearGradient {
        let bands = feed.spectrum
        guard !bands.isEmpty else {
            return LinearGradient(
                colors: [WhisperInk.recording.opacity(0.45)],
                startPoint: .leading, endPoint: .trailing)
        }
        let lastIndex = max(1, bands.count - 1)
        let stops = bands.enumerated().map { index, band in
            Gradient.Stop(
                color: WhisperInk.recording.opacity(
                    0.45 + 0.55 * Double(min(max(band, 0), 1))),
                location: Double(index) / Double(lastIndex)
            )
        }
        return LinearGradient(stops: stops, startPoint: .leading, endPoint: .trailing)
    }

    private func retarget(_ level: Float) {
        guard !reduceMotion else { return }
        // Underdamped on purpose: continuous retargeting at meter cadence
        // gives the line a slight organic overshoot — alive, not nervous.
        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
            lineWidth = Self.width(for: level)
        }
    }

    private static func width(for level: Float) -> CGFloat {
        let fraction = CGFloat(min(max(level, 0), 1))
        return WhisperMetrics.lineMinWidth
            + (WhisperMetrics.lineMaxWidth - WhisperMetrics.lineMinWidth) * fraction
    }
}

// MARK: - Proofread sparkles

/// The proofread twinkle: three tiny motes trailing the gliding line,
/// twinkling on deterministic sine phases — no random state, no particle
/// system, just a function of `time` drawn into a small Canvas each frame.
/// Bare content (no glass beneath): at ~2 pt these are glints, not chrome.
private struct WhisperSparkleTrail: View {
    let time: TimeInterval
    /// The sweep's current sine phase — shared with the line so the trail
    /// stays glued behind it.
    let sweepPhase: Double

    private static let moteCount = 3

    var body: some View {
        Canvas { context, size in
            let lineX = size.width / 2 + CGFloat(sin(sweepPhase)) * WhisperMetrics.sweepTravel
            // The trail lag scales with the sweep's velocity (the sine's
            // derivative), not its sign: motes stretch out at full glide and
            // gather under the line at each turn — no teleport when the
            // direction flips.
            let velocity = CGFloat(cos(sweepPhase))
            let baselineY = size.height - WhisperMetrics.lineHeight - 5
            for index in 0..<Self.moteCount {
                let order = CGFloat(index)
                let x = lineX - velocity * (order * 10 + 12)
                let bob = CGFloat(sin(time * 6.5 + Double(index) * 2.1)) * 2.5
                let y = baselineY + bob
                let twinkle = 0.5 + 0.5 * sin(time * 9 + Double(index) * 1.7)
                let alpha = (0.6 - Double(index) * 0.16) * twinkle
                let radius = 1.6 - order * 0.35
                // A soft gold halo under a bright core — legible over any
                // desktop without needing glass beneath.
                context.fill(
                    Path(
                        ellipseIn: CGRect(
                            x: x - radius * 2, y: y - radius * 2,
                            width: radius * 4, height: radius * 4)),
                    with: .color(WhisperInk.gold.opacity(alpha * 0.45)))
                context.fill(
                    Path(
                        ellipseIn: CGRect(
                            x: x - radius, y: y - radius,
                            width: radius * 2, height: radius * 2)),
                    with: .color(.white.opacity(alpha)))
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

// MARK: - Error stage

/// The error moment, compressed whisper-style: the line itself flashes
/// orange twice (attention without a sound), and a small chip above it
/// carries the short error text. No affordances — the panel stays
/// click-through during errors; recovery guidance lives in the app, not the
/// overlay.
private struct WhisperErrorStage: View {
    let error: DictationError
    let glassSpace: Namespace.ID

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// Starts hot; three autoreversed legs (hot→base→hot→base) render as two
    /// flashes settling at the base ink — one animation, no timers. Under
    /// Reduce Motion the line is static base orange.
    @State private var flashHot = true

    var body: some View {
        let message = error.errorDescription ?? "Something went wrong."
        VStack(spacing: WhisperMetrics.errorStackSpacing) {
            HStack(alignment: .center, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.orange)
                Text(message)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .minimumScaleFactor(0.9)
            }
            .padding(.horizontal, 12)
            .frame(width: WhisperMetrics.errorChipWidth, height: WhisperMetrics.chipHeight)
            .glassEffect(.regular, in: .capsule)
            .glassEffectID(WhisperGlassID.errorChip, in: glassSpace)
            .accessibilityLabel(message)

            Capsule()
                .fill(Color.orange.opacity(flashHot ? 1.0 : 0.55))
                .frame(
                    width: WhisperMetrics.lineRestWidth,
                    height: WhisperMetrics.lineHeight
                )
                .glassEffect(.regular, in: .capsule)
                .glassEffectID(WhisperGlassID.surface, in: glassSpace)
                .accessibilityHidden(true)
        }
        .onAppear {
            if reduceMotion {
                flashHot = false
            } else {
                withAnimation(.easeInOut(duration: 0.2).repeatCount(3, autoreverses: true)) {
                    flashHot = false
                }
            }
        }
    }
}

// MARK: - Committed chip

/// The bloom: the line morphs into this tiny capsule for the linger — a
/// glyph and two affordances, never the committed text (quietness is the
/// variant's identity). When the Proofread Pass edited, the checkmark
/// becomes a gold wand with the edit count and the wand's halo dissolves
/// over the first second — the quietest possible "polishing happened".
private struct WhisperCommittedChip: View {
    let edits: [WordEdit]
    let actions: OverlayActions
    let glassSpace: Namespace.ID
    let dismiss: @MainActor () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// The entrance halo on the wand; owned here so every chip instance
    /// (`.id(beat.id)` upstream) replays it.
    @State private var polishGlow = true

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            if edits.isEmpty {
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.green)
            } else {
                Image(systemName: "wand.and.sparkles")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(WhisperInk.gold)
                    .shadow(
                        color: WhisperInk.gold.opacity(polishGlow ? 0.9 : 0),
                        radius: polishGlow ? 5 : 0)
                Text("\(edits.count)")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(WhisperInk.gold)
            }
            Spacer(minLength: 4)
            chipButton("flag", label: "Flag as wrong") {
                actions.flagLastTakeWrong()
                dismiss()
            }
            chipButton("pencil", label: "Edit in history") {
                actions.editLastTake()
                dismiss()
            }
        }
        .padding(.horizontal, 12)
        .frame(width: WhisperMetrics.committedChipWidth, height: WhisperMetrics.chipHeight)
        .glassEffect(.regular.interactive(), in: .capsule)
        .glassEffectID(WhisperGlassID.surface, in: glassSpace)
        .accessibilityLabel(
            edits.isEmpty
                ? "Inserted"
                : "Inserted, proofreader polished \(edits.count) words"
        )
        .onAppear {
            guard !edits.isEmpty else { return }
            if reduceMotion {
                polishGlow = false
            } else {
                // Let the bloom land first, then dissolve the halo.
                withAnimation(.easeOut(duration: 0.9).delay(0.2)) {
                    polishGlow = false
                }
            }
        }
    }

    /// One small icon affordance: plain style (the glass chip is the
    /// chrome), secondary ink, generous hit target.
    private func chipButton(
        _ systemName: String, label: String, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 22, height: 22)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

// MARK: - Registry entry

extension OverlayVariants {
    /// The whisper — the quietest exploration: an underline, not a pill.
    static let whisper = OverlayVariant(
        id: "whisper",
        displayName: "Whisper",
        placement: .whisper
    ) { feed, actions in
        AnyView(WhisperOverlayView(feed: feed, actions: actions))
    }
}

// MARK: - Previews

#Preview("Recording") {
    let feed = DictationFeed()
    feed.setPhase(.recording)
    feed.apply(MeterFrame(level: 0.55, bands: [0.2, 0.45, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1]))
    return WhisperOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Processing") {
    let feed = DictationFeed()
    feed.setPhase(.processing)
    return WhisperOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Committed — polished") {
    let feed = DictationFeed()
    feed.emit(
        .committed(
            text: "Their plan works",
            duration: 2.4,
            edits: [
                WordEdit(original: "there", replacement: "their"),
                WordEdit(original: "planned", replacement: "plan"),
            ]))
    return WhisperOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Rejected") {
    let feed = DictationFeed()
    feed.emit(.rejected(raw: "uh the um", reason: "Unintelligible transcription"))
    return WhisperOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}
