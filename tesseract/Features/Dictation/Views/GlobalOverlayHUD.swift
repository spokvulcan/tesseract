//
//  GlobalOverlayHUD.swift
//  tesseract
//

import SwiftUI

/// The classic dictation pill — the incumbent **Overlay Variant** (map #283).
/// A Liquid Glass pill floating on top of all applications: one pure (untinted)
/// `.glassEffect` capsule with vibrant content on top — the state reads through
/// the content (red bars recording, amber icon on error), never through a glass
/// tint, which over arbitrary desktops looks like a painted background instead
/// of glass. The system material adapts to light/dark, the Clear/Tinted
/// appearance setting, and Reduce Transparency on its own.
///
/// The variant owns *all* motion (map #283): entrance/exit and per-phase size
/// changes are one SwiftUI spring inside the panel's fixed canvas — the panel
/// never fades or resizes.
struct GlobalOverlayHUD: View {
    /// The Overlay Feed — the only pipeline surface this view sees.
    var feed: DictationFeed
    /// The overlay action surface (ticket #289) — what the lingering beat's
    /// click affordances invoke. Clicks only; the overlay stays keyboard-free.
    var actions: OverlayActions = .none

    /// The phase the pill currently renders — updated inside `withAnimation`,
    /// so mount/unmount, per-phase size, and content swaps all ride one spring.
    @State private var shownPhase: DictationFeed.Phase = .idle
    /// A lingering terminal beat (map #283): committed and rejected takes
    /// hold a small affordance pill for ``DictationFeed/affordanceLinger``
    /// after the phase returned to idle — flag/edit on a commit, insert-raw
    /// on a rejection. Passive: it fades on its own; the press is the retry.
    @State private var shownBeat: DictationFeed.Beat?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GlassEffectContainer {
            ZStack(alignment: .bottom) {
                if shownPhase != .idle {
                    pillContainer(for: shownPhase)
                        .transition(
                            reduceMotion
                                ? .opacity
                                : .scale(scale: 0.85, anchor: .bottom).combined(with: .opacity)
                        )
                } else if let shownBeat {
                    beatPill(for: shownBeat)
                        .transition(
                            reduceMotion
                                ? .opacity
                                : .scale(scale: 0.85, anchor: .bottom).combined(with: .opacity)
                        )
                }
            }
            .frame(
                width: PillMetrics.canvasSize.width,
                height: PillMetrics.canvasSize.height,
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
        // Snappy in (the pill must read as "on" within ~100 ms of the press or
        // dictation feels laggy), slightly softer out.
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

    @ViewBuilder
    private func hudContent(for phase: DictationFeed.Phase) -> some View {
        switch phase {
        case .recording:
            // AudioBarsView reads the level itself, so the meter cadence
            // invalidates only the bars subtree — the pill chrome and glass
            // never re-diff during steady recording.
            AudioBarsView(feed: feed)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
        case .processing, .proofreading:
            // The TimelineView lives *inside* the pill so the glass chrome sits
            // outside the 60 fps closure — only the dots row re-evaluates per frame.
            // Proofreading rides the same dots with a wand — one visual family.
            if reduceMotion {
                processingContent(time: nil, proofreading: phase == .proofreading)
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
                    processingContent(
                        time: timeline.date.timeIntervalSinceReferenceDate,
                        proofreading: phase == .proofreading)
                }
            }
        case .error(let error):
            errorContent(for: error)
        case .idle:
            EmptyView()
        }
    }

    private func processingContent(time: TimeInterval?, proofreading: Bool = false) -> some View {
        HStack(spacing: 8) {
            if proofreading {
                Image(systemName: "wand.and.sparkles")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("Proofreading")
            }
            if let time {
                ProcessingDotsView(time: time)
                    .frame(height: 12)
            } else {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    /// The lingering terminal-beat pill (ticket #289): the panel is clickable
    /// for exactly this window (one App Bindings rule), so these are the
    /// overlay's only interactive moments — one-click, no focus steal.
    @ViewBuilder
    private func beatPill(for beat: DictationFeed.Beat) -> some View {
        switch beat.outcome {
        case .committed(_, _, let edits):
            committedPill(edits: edits)
        case .rejected:
            rejectionPill
        case .empty, .cancelled, .superseded:
            EmptyView()
        }
    }

    /// Post-commit: names what happened (polish count when the Proofread
    /// Pass edited) and carries the flywheel affordances — flag "wrong" and
    /// "edit in history".
    private func committedPill(edits: [WordEdit]) -> some View {
        HStack(alignment: .center, spacing: 8) {
            Image(systemName: edits.isEmpty ? "checkmark" : "wand.and.sparkles")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.green)
            Text(edits.isEmpty ? "Inserted" : "Polished \(edits.count)")
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(1)
            Spacer(minLength: 4)
            beatButton("flag", label: "Flag as wrong") {
                actions.flagLastTakeWrong()
                dismissBeat()
            }
            beatButton("pencil", label: "Edit in history") {
                actions.editLastTake()
                dismissBeat()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(width: PillMetrics.beatSize.width, height: PillMetrics.beatSize.height)
        .glassEffect(.regular, in: .capsule)
        .accessibilityLabel(
            edits.isEmpty
                ? "Inserted"
                : "Inserted, proofreader polished \(edits.count) words")
    }

    /// The lingering rejected-take pill. Passive-first: the press is the
    /// retry; the one affordance inserts the raw take anyway (which also
    /// flags the pass as wrong).
    private var rejectionPill: some View {
        HStack(alignment: .center, spacing: 8) {
            Image(systemName: "arrow.uturn.backward")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.orange)
            Text("Didn't catch that")
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.9)
            Spacer(minLength: 4)
            Button {
                actions.insertRawAnyway()
                dismissBeat()
            } label: {
                Text("Insert anyway")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Insert the raw transcription anyway")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(width: PillMetrics.beatSize.width, height: PillMetrics.beatSize.height)
        .glassEffect(.regular, in: .capsule)
        .accessibilityLabel("Transcription rejected — hold the hotkey to retry")
    }

    /// One small icon affordance on a lingering pill: plain style (the glass
    /// pill is the chrome), secondary ink, generous hit target.
    private func beatButton(
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

    private func errorContent(for error: DictationError) -> some View {
        let message = error.errorDescription ?? "Something went wrong."
        return HStack(alignment: .center, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.orange)
            Text(message)
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(2)
                .multilineTextAlignment(.leading)
                .minimumScaleFactor(0.9)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .accessibilityLabel(message)
    }

    private func pillContainer(for phase: DictationFeed.Phase) -> some View {
        // Size from the shared PillMetrics keyed on the rendered phase. The
        // frame change animates under `apply`'s spring — the panel's canvas is
        // fixed, so this is pure content layout.
        let size = PillMetrics.size(for: phase)

        // Pure, untinted `.regular` glass — the material *is* the pill; state
        // color lives in the content (owner-selected over the `.clear` +
        // forced-light variant after on-hardware comparison). `.regular`
        // keeps the HIG legibility guarantees over arbitrary backdrops; in
        // dark appearance it renders as smoked glass rather than Control
        // Center's bright material — that material is a private variant no
        // third party can select.
        return hudContent(for: phase)
            .frame(width: size.width, height: size.height)
            .glassEffect(.regular, in: .capsule)
    }
}

#Preview("Recording") {
    let feed = DictationFeed()
    feed.setPhase(.recording)
    feed.apply(MeterFrame(level: 0.5, bands: MeterFrame.zeroBands))
    return GlobalOverlayHUD(feed: feed)
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
    return GlobalOverlayHUD(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}

#Preview("Error") {
    let feed = DictationFeed()
    feed.setPhase(.error(.noSpeechDetected))
    return GlobalOverlayHUD(feed: feed)
        .padding(50)
        .background(
            LinearGradient(
                colors: [.blue, .purple, .orange],
                startPoint: .topLeading, endPoint: .bottomTrailing
            ))
}
