//
//  RibbonOverlayVariant.swift
//  tesseract
//

import AppKit
import SwiftUI

// MARK: - Metrics

/// The single source of truth for the ribbon variant's sizes.
///
/// The ribbon is the opposite philosophy of the pill: a full-width horizon
/// line hugging the bottom screen edge, not a floating object. Per-state strip
/// sizes are *content layout* inside the fixed ``canvasSize`` — the panel
/// window never resizes (map #283).
///
/// `nonisolated` so it escapes the build's MainActor default isolation — same
/// reasoning as ``PillMetrics``: the placement math must run off-main.
nonisolated enum RibbonMetrics {
    /// The glass strip's width in every state — the horizon line never
    /// changes length, only its height and content.
    static let stripWidth: CGFloat = 480
    /// The thin waveform strip (recording / processing / proofreading).
    static let waveHeight: CGFloat = 26
    /// The caption strip (lingering beats + errors) — tall enough for 12 pt
    /// text plus 24 pt hit targets, still a sliver.
    static let captionHeight: CGFloat = 34

    /// The *resting* strip bottom sits this far above the visible frame's
    /// bottom — the "few points above the bottom edge" of the design.
    static let bottomInset: CGFloat = 12
    /// The entrance rise distance. Doubles as the strip's bottom padding
    /// inside the canvas, so the rise starts below the resting line without
    /// ever leaving the panel window (the panel is fixed and would clip).
    static let entranceRise: CGFloat = 8

    /// Horizontal inset of the ripple inside the strip — keeps the waveform
    /// clear of the capsule's end caps (13 pt radius at ``waveHeight``).
    static let waveEndInset: CGFloat = 18
    /// Half-thickness of the resting waveform line: at silence the ribbon is
    /// a hairline, not empty glass.
    static let waveFloor: CGFloat = 1.4

    /// The fixed panel canvas: the caption strip plus the entrance-rise
    /// bottom padding plus antialiasing headroom. Strips are bottom-anchored
    /// inside it, so per-state heights grow upward from a constant baseline.
    static let canvasSize = CGSize(width: 500, height: 52)
}

// MARK: - Placement

nonisolated extension OverlayPlacement {
    /// The ribbon's canvas: centred horizontally, hugging the bottom of the
    /// visible frame. The canvas dips ``RibbonMetrics/entranceRise`` below
    /// the resting line so the entrance can start lower and rise — the
    /// resting strip bottom lands exactly ``RibbonMetrics/bottomInset``
    /// above the visible frame's bottom.
    static let ribbon = OverlayPlacement(
        frame: { geometry in
            let size = RibbonMetrics.canvasSize
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.midX - size.width / 2,
                y: visible.minY + RibbonMetrics.bottomInset - RibbonMetrics.entranceRise,
                width: size.width,
                height: size.height
            )
        }
    )
}

// MARK: - View

/// The **ribbon** Overlay Variant (map #283): an ultra-thin Liquid Glass
/// strip hugging the bottom screen edge — a horizon line, not a floating
/// object. One pure (untinted) `.glassEffect` capsule per strip with vibrant
/// content on top: the live spectrum ripples in red while recording, an
/// energy sweep travels the frozen ripple while processing, and terminal
/// beats swell the line into a caption strip carrying the committed text.
///
/// The variant owns *all* motion (map #283): entrance/exit is a small rise
/// from the bottom edge plus a fade, per-state strip heights are content
/// layout, and everything animates inside the panel's fixed canvas — the
/// panel never fades or resizes.
struct RibbonOverlayView: View {
    /// The Overlay Feed — the only pipeline surface this view sees.
    var feed: DictationFeed
    /// The overlay action surface (ticket #289) — what the lingering beat's
    /// click affordances invoke. Clicks only; the overlay stays keyboard-free.
    var actions: OverlayActions = .none

    /// The phase the ribbon currently renders — updated inside
    /// `withAnimation`, so mount/unmount, strip-height changes, and content
    /// swaps all ride one spring.
    @State private var shownPhase: DictationFeed.Phase = .idle
    /// A lingering terminal beat (map #283): committed and rejected takes
    /// hold the caption strip for ``DictationFeed/affordanceLinger`` after
    /// the phase returned to idle — flag/edit on a commit, insert-raw on a
    /// rejection. Passive: it slides away on its own; the press is the retry.
    @State private var shownBeat: DictationFeed.Beat?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// Every strip shares one glass id, so phase ↔ beat swaps morph the
    /// horizon line in place instead of crossfading two separate objects.
    @Namespace private var glassNamespace

    private static let stripGlassID = "ribbon.strip"

    var body: some View {
        GlassEffectContainer {
            ZStack(alignment: .bottom) {
                if shownPhase != .idle {
                    phaseStrip(for: shownPhase)
                        .transition(stripTransition)
                } else if let shownBeat {
                    beatStrip(for: shownBeat)
                        .transition(stripTransition)
                }
            }
            // Bottom padding == entrance rise: the entering strip's lowest
            // point exactly touches the canvas bottom, never clips past it.
            .padding(.bottom, RibbonMetrics.entranceRise)
            .frame(
                width: RibbonMetrics.canvasSize.width,
                height: RibbonMetrics.canvasSize.height,
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
        }
    }

    /// Rise-from-the-bottom-edge entrance, slide-away exit; opacity-only
    /// under Reduce Motion.
    private var stripTransition: AnyTransition {
        reduceMotion
            ? .opacity
            : .offset(y: RibbonMetrics.entranceRise).combined(with: .opacity)
    }

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
        // Snappy in (the ribbon must read as "on" within ~100 ms of the press
        // or dictation feels laggy), slightly softer out.
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

    private func dismissBeat(ifStill id: UInt64? = nil) {
        if let id, shownBeat?.id != id { return }
        withAnimation(reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.85)) {
            shownBeat = nil
        }
    }

    // MARK: Phase strips

    @ViewBuilder
    private func phaseStrip(for phase: DictationFeed.Phase) -> some View {
        switch phase {
        case .recording, .processing, .proofreading:
            activeStrip(for: phase)
        case .error(let error):
            errorStrip(for: error)
        case .idle:
            EmptyView()
        }
    }

    /// The thin waveform strip for the three live phases. One structural
    /// position for all three keeps ``RibbonWaveView``'s identity (and its
    /// smoothed bands) stable across the recording → processing flip — that
    /// persistence *is* the freeze.
    private func activeStrip(for phase: DictationFeed.Phase) -> some View {
        let recording = phase == .recording
        let proofreading = phase == .proofreading
        return ZStack {
            RibbonWaveView(feed: feed, live: recording)
                .opacity(recording ? 1 : 0.4)
            if !recording {
                if reduceMotion {
                    frozenIndicator(proofreading: proofreading)
                } else {
                    // The TimelineView wraps *only* the comet overlay: the
                    // glass, the frozen wave, and the strip chrome all sit
                    // outside the 60 fps closure.
                    TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                        RibbonSweepView(
                            time: timeline.date.timeIntervalSinceReferenceDate,
                            proofreading: proofreading)
                    }
                    .accessibilityHidden(true)
                }
            }
        }
        .frame(width: RibbonMetrics.stripWidth, height: RibbonMetrics.waveHeight)
        // The comet travels past the capsule's end caps — clip the content,
        // then let the glass render in the same shape behind it.
        .clipShape(.capsule)
        .glassEffect(.regular, in: .capsule)
        .glassEffectID(Self.stripGlassID, in: glassNamespace)
        .accessibilityLabel(
            recording ? "Recording" : proofreading ? "Proofreading" : "Processing")
    }

    /// Reduce Motion replacement for the sweep: the frozen ripple stays, the
    /// decorative comet becomes the system spinner (plus the wand while the
    /// Proofread Pass runs) — same family as the classic pill's fallback.
    private func frozenIndicator(proofreading: Bool) -> some View {
        HStack(spacing: 8) {
            if proofreading {
                Image(systemName: "wand.and.sparkles")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            ProgressView()
                .controlSize(.small)
        }
    }

    /// Errors ride the caption strip: the typed description always, the
    /// recovery suggestion when the width allows (the description wins the
    /// layout fight).
    private func errorStrip(for error: DictationError) -> some View {
        let message = error.errorDescription ?? "Something went wrong."
        let suggestion = error.recoverySuggestion
        return HStack(alignment: .center, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.orange)
            Text(message)
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.9)
                .layoutPriority(1)
            if let suggestion {
                Text(suggestion)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .frame(width: RibbonMetrics.stripWidth, height: RibbonMetrics.captionHeight)
        .glassEffect(.regular, in: .capsule)
        .glassEffectID(Self.stripGlassID, in: glassNamespace)
        .accessibilityLabel(suggestion.map { "\(message) \($0)" } ?? message)
    }

    // MARK: Beat strips

    /// The lingering terminal-beat caption strip (ticket #289): the panel is
    /// clickable for exactly this window (one App Bindings rule), so these
    /// are the overlay's only interactive moments — one-click, no focus steal.
    @ViewBuilder
    private func beatStrip(for beat: DictationFeed.Beat) -> some View {
        switch beat.outcome {
        case .committed(let text, _, let edits):
            committedStrip(text: text, edits: edits)
        case .rejected(let raw, _):
            rejectedStrip(raw: raw)
        case .empty, .cancelled, .superseded:
            EmptyView()
        }
    }

    /// Post-commit: the ribbon becomes a caption of what was inserted —
    /// middle-truncated so both the opening and the closing words survive —
    /// with the polish chip when the Proofread Pass edited, and the flywheel
    /// affordances (flag "wrong", edit in history) on the trailing end.
    private func committedStrip(text: String, edits: [WordEdit]) -> some View {
        HStack(alignment: .center, spacing: 8) {
            Image(systemName: "checkmark")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.green)
            Text(text)
                .font(.system(size: 12, weight: .medium))
                .lineLimit(1)
                .truncationMode(.middle)
            if !edits.isEmpty {
                polishChip(count: edits.count)
            }
            Spacer(minLength: 8)
            beatButton("flag", label: "Flag as wrong") {
                actions.flagLastTakeWrong()
                dismissBeat()
            }
            beatButton("pencil", label: "Edit in history") {
                actions.editLastTake()
                dismissBeat()
            }
        }
        .padding(.horizontal, 16)
        .frame(width: RibbonMetrics.stripWidth, height: RibbonMetrics.captionHeight)
        .glassEffect(.regular, in: .capsule)
        .glassEffectID(Self.stripGlassID, in: glassNamespace)
        .accessibilityLabel(
            edits.isEmpty
                ? "Inserted: \(text)"
                : "Inserted, proofreader polished \(polishPhrase(edits.count)): \(text)")
    }

    /// The wand-and-count chip — the committed strip's "polishing happened"
    /// mark when the Proofread Pass changed words. Detail lives in the
    /// history window; here it is one glanceable token.
    private func polishChip(count: Int) -> some View {
        HStack(spacing: 3) {
            Image(systemName: "wand.and.sparkles")
                .font(.system(size: 9, weight: .semibold))
            Text("\(count)")
                .font(.system(size: 10, weight: .semibold))
                .monospacedDigit()
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(.quaternary, in: .capsule)
        .accessibilityHidden(true)
    }

    private func polishPhrase(_ count: Int) -> String {
        count == 1 ? "1 word" : "\(count) words"
    }

    /// The lingering rejected-take strip. Passive-first: the press is the
    /// retry; the raw take is shown in tertiary so "insert anyway" is an
    /// informed click (which also flags the pass as wrong).
    private func rejectedStrip(raw: String) -> some View {
        HStack(alignment: .center, spacing: 8) {
            Image(systemName: "arrow.uturn.backward")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.orange)
            Text("Didn't catch that")
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(1)
                .layoutPriority(1)
            Text(raw)
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .truncationMode(.middle)
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
        .padding(.horizontal, 16)
        .frame(width: RibbonMetrics.stripWidth, height: RibbonMetrics.captionHeight)
        .glassEffect(.regular, in: .capsule)
        .glassEffectID(Self.stripGlassID, in: glassNamespace)
        .accessibilityLabel("Transcription rejected — hold the hotkey to retry")
    }

    /// One small icon affordance on the lingering strip: plain style (the
    /// glass strip is the chrome), secondary ink, generous hit target.
    private func beatButton(
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

// MARK: - Waveform

/// The ribbon's live spectrum ripple: the 8 log-spaced bands mirrored outward
/// from center (low bands — where speech energy lives — at the center),
/// cosine-interpolated into a smooth symmetric filled wave, tapered to a
/// hairline at the capsule's caps.
///
/// Reads the feed directly (never passed band values), so the ~47 Hz meter
/// cadence invalidates only this subtree — the glass and strip chrome never
/// re-diff during steady recording. The per-band attack/release smoothing
/// keeps the ripple continuous between meter frames; it also absorbs the one
/// zeroed frame the feed emits when capture stops, so the freeze into
/// `.processing` never collapses the shape. Meter-driven motion is
/// informational, not decorative, so it stays live under Reduce Motion —
/// same stance as `AudioBarsView`.
private struct RibbonWaveView: View {
    var feed: DictationFeed
    /// `false` freezes the last displayed ripple (processing/proofreading);
    /// this view's identity must be kept stable across the flip.
    let live: Bool

    @State private var displayed: [Float] = MeterFrame.zeroBands
    /// Smoothed overall level — breathes the ripple's amplitude on top of
    /// the per-band values.
    @State private var breathe: Float = 0

    var body: some View {
        // Read the state in body (not inside the Canvas closure) so the
        // dependency is registered during evaluation.
        let bands = displayed
        let amplitude = CGFloat(0.55 + 0.45 * breathe)
        Canvas { context, size in
            Self.drawRipple(bands: bands, amplitude: amplitude, in: &context, size: size)
        }
        .onChange(of: feed.spectrum) { _, target in
            guard live else { return }
            displayed = zip(displayed, target).map { shown, band in
                // Fast attack, slower release — the classic meter feel.
                let factor: Float = band > shown ? 0.55 : 0.25
                return shown + (band - shown) * factor
            }
        }
        .onChange(of: feed.level) { _, level in
            guard live else { return }
            breathe += (min(max(level, 0), 1) - breathe) * 0.3
        }
        .onAppear {
            // Seed from the feed so a mid-recording mount (variant switch)
            // starts at the live shape, not a flat line.
            displayed = feed.spectrum
            breathe = feed.level
        }
    }

    /// Pure geometry — `nonisolated` so the Canvas renderer carries no actor
    /// hops and the math is trivially testable.
    private nonisolated static func drawRipple(
        bands: [Float], amplitude: CGFloat, in context: inout GraphicsContext, size: CGSize
    ) {
        guard bands.count >= 2 else { return }
        let midY = size.height / 2
        let halfWidth = size.width / 2 - RibbonMetrics.waveEndInset
        guard halfWidth > 0 else { return }
        let maxAmp = size.height / 2 - 4

        // Half-amplitude at a mirrored distance from center (0 → 1 at edge).
        func rise(at distance: CGFloat) -> CGFloat {
            let pos = distance * CGFloat(bands.count - 1)
            let index = min(bands.count - 2, Int(pos))
            let fraction = pos - CGFloat(index)
            let smooth = (1 - cos(fraction * .pi)) / 2
            let value = CGFloat(bands[index]) * (1 - smooth) + CGFloat(bands[index + 1]) * smooth
            // Cubic taper pins the ripple to the hairline at the caps.
            let taper = 1 - distance * distance * distance
            return RibbonMetrics.waveFloor + value * maxAmp * amplitude * taper
        }

        let steps = 96
        var path = Path()
        for step in 0...steps {
            let t = CGFloat(step) / CGFloat(steps)
            let x = size.width / 2 + (t - 0.5) * 2 * halfWidth
            let y = midY - rise(at: abs(t - 0.5) * 2)
            if step == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        for step in stride(from: steps, through: 0, by: -1) {
            let t = CGFloat(step) / CGFloat(steps)
            let x = size.width / 2 + (t - 0.5) * 2 * halfWidth
            path.addLine(to: CGPoint(x: x, y: midY + rise(at: abs(t - 0.5) * 2)))
        }
        path.closeSubpath()

        // Red at the center fading toward the caps — vibrant content carries
        // the recording state; the glass stays untinted.
        let gradient = Gradient(stops: [
            .init(color: .red.opacity(0.30), location: 0),
            .init(color: .red.opacity(0.95), location: 0.35),
            .init(color: .red, location: 0.5),
            .init(color: .red.opacity(0.95), location: 0.65),
            .init(color: .red.opacity(0.30), location: 1),
        ])
        context.fill(
            path,
            with: .linearGradient(
                gradient,
                startPoint: CGPoint(x: 0, y: midY),
                endPoint: CGPoint(x: size.width, y: midY)))
    }
}

// MARK: - Sweep

/// The energy sweep over the frozen ripple: a soft red comet traveling
/// left → right. While the Proofread Pass runs it gains a brighter core and
/// a wand gliding at its head — the "polishing" character. Pure function of
/// the driving `time`, mirroring `ProcessingDotsView`'s no-state shape.
private struct RibbonSweepView: View {
    let time: TimeInterval
    let proofreading: Bool

    var body: some View {
        let period: TimeInterval = proofreading ? 2.6 : 2.0
        let progress = CGFloat((time / period).truncatingRemainder(dividingBy: 1))
        let cometWidth: CGFloat = proofreading ? 96 : 120
        let peak: Double = proofreading ? 0.65 : 0.4
        // Travels from fully off the left cap to fully off the right one —
        // the capsule clip makes the wrap seamless.
        let x = progress * (RibbonMetrics.stripWidth + cometWidth) - cometWidth

        ZStack(alignment: .leading) {
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0),
                    .init(color: .red.opacity(0.10), location: 0.45),
                    .init(color: .red.opacity(peak), location: 0.82),
                    .init(color: .clear, location: 1),
                ],
                startPoint: .leading, endPoint: .trailing
            )
            .frame(width: cometWidth, height: RibbonMetrics.waveHeight)
            .offset(x: x)
            if proofreading {
                Image(systemName: "wand.and.sparkles")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.primary)
                    .offset(x: x + cometWidth * 0.82 - 7)
            }
        }
        .frame(
            width: RibbonMetrics.stripWidth, height: RibbonMetrics.waveHeight,
            alignment: .leading)
    }
}

// MARK: - Registration

extension OverlayVariants {
    static let ribbon = OverlayVariant(
        id: "ribbon",
        displayName: "Ribbon",
        placement: .ribbon
    ) { feed, actions in
        AnyView(RibbonOverlayView(feed: feed, actions: actions))
    }
}

// MARK: - Previews

#Preview("Recording") {
    let feed = DictationFeed()
    feed.setPhase(.recording)
    return RibbonOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        )
        .task {
            // Synthetic speech-like meter frames at tap cadence.
            var t = 0.0
            while !Task.isCancelled {
                t += 0.033
                let bands = (0..<MeterFrame.bandCount).map { band -> Float in
                    let wave = sin(t * 2.6 + Double(band) * 0.9) * 0.5 + 0.5
                    let jitter = sin(t * 9 + Double(band) * 2.1) * 0.12
                    let falloff = 1.0 - Double(band) * 0.09
                    return Float(max(0, min(1, (0.18 + 0.55 * wave + jitter) * falloff)))
                }
                feed.apply(
                    MeterFrame(level: Float(0.45 + 0.3 * sin(t * 1.7)), bands: bands))
                try? await Task.sleep(for: .milliseconds(33))
            }
        }
}

#Preview("Processing") {
    let feed = DictationFeed()
    feed.apply(
        MeterFrame(level: 0.55, bands: [0.62, 0.78, 0.55, 0.44, 0.38, 0.26, 0.18, 0.10]))
    feed.setPhase(.processing)
    return RibbonOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Proofreading") {
    let feed = DictationFeed()
    feed.apply(
        MeterFrame(level: 0.55, bands: [0.62, 0.78, 0.55, 0.44, 0.38, 0.26, 0.18, 0.10]))
    feed.setPhase(.proofreading)
    return RibbonOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Committed — polished") {
    let feed = DictationFeed()
    return RibbonOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        )
        .task {
            // Re-emit on a loop so the entrance → linger → slide-away cycle
            // stays visible in the canvas.
            while !Task.isCancelled {
                feed.emit(
                    .committed(
                        text:
                            "Let's meet at half past nine to walk through the quarterly numbers together.",
                        duration: 6.4,
                        edits: [
                            WordEdit(original: "passed", replacement: "past"),
                            WordEdit(original: "quaterly", replacement: "quarterly"),
                        ]))
                try? await Task.sleep(for: .seconds(4))
            }
        }
}

#Preview("Rejected") {
    let feed = DictationFeed()
    return RibbonOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        )
        .task {
            while !Task.isCancelled {
                feed.emit(
                    .rejected(
                        raw: "uh the um thing with the the numbers thing",
                        reason: "Unintelligible transcription"))
                try? await Task.sleep(for: .seconds(4))
            }
        }
}

#Preview("Error") {
    let feed = DictationFeed()
    feed.setPhase(.error(.noSpeechDetected))
    return RibbonOverlayView(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}
