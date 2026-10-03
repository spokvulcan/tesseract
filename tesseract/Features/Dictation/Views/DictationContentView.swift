//
//  DictationContentView.swift
//  tesseract
//
//  The Dictation page: the **catch record** (PRD #612). One sentence with
//  this week's catches and fixes, the week as a chart, one tile per
//  **Learned Word**, and today's takes, each one click from a fix in the
//  **Lens**. Recording, the full history and the **Correction Pair** export
//  live in the toolbar.
//

import AppKit
import SwiftUI

/// Dictation page surface constants (design language §2: one type size and
/// one spacing rhythm per surface; hierarchy comes from weight and color).
enum DictationPageStyle {
    static let bodySize: CGFloat = 15
    static let rhythm: CGFloat = 12
}

struct DictationContentView: View {
    @Environment(DictationCoordinator.self) private var coordinator
    @Environment(TranscriptionEngine.self) private var transcriptionEngine
    @Environment(TranscriptionHistory.self) private var history
    @Environment(CorrectionPairStore.self) private var pairs
    @Environment(LearnedWordStore.self) private var learnedWords
    @EnvironmentObject private var permissionsManager: PermissionsManager
    @Environment(SettingsManager.self) private var settings

    @State private var showsHistory = false
    /// The last Forget, kept for its Undo. A new Forget replaces it.
    @State private var forgotten: Forgotten?

    /// A forgotten word as the Undo line names it.
    struct Forgotten: Equatable {
        let receipt: LearnedWordStore.Receipt
        let meant: String
        let heard: [String]
        /// The word as the Forget left it: any later change to it (a new
        /// fix relearning it, or taking one of its heard forms) withdraws
        /// the Undo, which would bring back a stale copy.
        let afterForget: LearnedWord
    }

    var body: some View {
        // Every minute, so today and the week roll over at midnight.
        TimelineView(.everyMinute) { context in
            let record = CatchRecord(
                words: learnedWords.words, pairs: pairs.pairs, entries: history.entries,
                now: context.date)
            ScrollView {
                VStack(alignment: .leading, spacing: DictationPageStyle.rhythm) {
                    statusLine
                    hero(record)
                    CatchWeekChart(days: record.days)
                    if !record.tiles.isEmpty || pendingUndo != nil {
                        whatItCatches(record, now: context.date)
                    }
                    today(record)
                }
                .frame(maxWidth: Theme.Layout.contentMaxWidth, alignment: .leading)
                .padding(.horizontal, 24)
                .padding(.vertical, DictationPageStyle.rhythm)
                .frame(maxWidth: .infinity)
            }
        }
        .toolbar {
            ToolbarItem {
                Button {
                    coordinator.toggleRecording()
                } label: {
                    Label(
                        isRecording ? "Stop" : "Record",
                        systemImage: isRecording ? "stop.fill" : "mic")
                }
                .disabled(!canToggleRecording)
                .help(
                    isRecording
                        ? "Stop recording (\(settings.hotkey.displayString))"
                        : "Record a take (\(settings.hotkey.displayString) in any app)")
            }
            ToolbarItem {
                Button {
                    showsHistory = true
                } label: {
                    Label("History", systemImage: "clock")
                }
                .help("Recent takes, newest first")
            }
            ToolbarItem {
                Button {
                    exportCorrections()
                } label: {
                    Label("Export Corrections…", systemImage: "square.and.arrow.up")
                }
                .disabled(pairs.pairs.isEmpty)
                .help("Export the correction pairs as JSONL")
            }
        }
        .sheet(isPresented: $showsHistory) { TranscriptionHistorySheet() }
        .navigationTitle("Dictation")
    }

    // MARK: - Recording

    private var isRecording: Bool { coordinator.state == .recording }

    /// The toolbar button starts and stops a take; it can't while the model
    /// or the microphone isn't ready, or while a take is being transcribed.
    private var canToggleRecording: Bool {
        guard transcriptionEngine.isModelLoaded,
            permissionsManager.microphonePermission == .granted
        else { return false }
        switch coordinator.state {
        case .processing, .proofreading: return false
        case .idle, .recording, .error: return true
        }
    }

    /// Only when dictation can't run or is busy; nothing when it is idle and
    /// ready (design language §2: quiet loading).
    @ViewBuilder
    private var statusLine: some View {
        if permissionsManager.microphonePermission != .granted {
            StatusIndicator(
                badge: .dot(.secondary),
                title: "Microphone access needed",
                detail: "Grant access in System Settings › Privacy & Security › Microphone."
            )
            .frame(maxWidth: .infinity, alignment: .leading)
        } else if !transcriptionEngine.isModelLoaded {
            StatusIndicator(badge: .spinner, title: "Loading dictation model…", detail: nil)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else if coordinator.state != .idle {
            StatusIndicator(state: coordinator.state)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Hero

    private func hero(_ record: CatchRecord) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(record.headline)
                .font(.system(size: DictationPageStyle.bodySize, weight: .semibold))
                .foregroundStyle(.primary)
            Text(record.detail(fixHotkey: settings.fixHotkey.displayString))
                .font(.system(size: DictationPageStyle.bodySize))
                .foregroundStyle(.secondary)
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    // MARK: - What it catches

    /// The last Forget while its word is still as the Forget left it.
    private var pendingUndo: Forgotten? {
        guard let forgotten,
            learnedWords.word(withID: forgotten.receipt.wordID) == forgotten.afterForget
        else { return nil }
        return forgotten
    }

    private func whatItCatches(_ record: CatchRecord, now: Date) -> some View {
        VStack(alignment: .leading, spacing: DictationPageStyle.rhythm) {
            SectionHeader(
                title: "What it catches",
                detail: record.tiles.isEmpty ? nil : Self.wordCount(record.tiles.count))

            if let pendingUndo {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(Self.forgetNotice(meant: pendingUndo.meant, heard: pendingUndo.heard))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Undo") {
                        learnedWords.undo(pendingUndo.receipt)
                        forgotten = nil
                    }
                    .buttonStyle(.link)
                }
                .font(.system(size: DictationPageStyle.bodySize))
            }

            LazyVGrid(
                columns: [
                    GridItem(
                        .adaptive(minimum: 260), spacing: DictationPageStyle.rhythm,
                        alignment: .top)
                ],
                alignment: .leading,
                spacing: DictationPageStyle.rhythm
            ) {
                ForEach(record.tiles) { tile in
                    LearnedWordTile(tile: tile, now: now) { forget(tile) }
                }
            }
        }
    }

    private func forget(_ tile: CatchRecord.Tile) {
        guard let receipt = learnedWords.forget(tile.id),
            let word = learnedWords.word(withID: tile.id)
        else { return }
        forgotten = Forgotten(
            receipt: receipt, meant: tile.meant, heard: tile.heard, afterForget: word)
        // The tile VoiceOver was on is gone: say what happened and where
        // Undo is.
        NSAccessibility.post(
            element: NSApp as Any, notification: .announcementRequested,
            userInfo: [
                .announcement: Self.forgetNotice(meant: tile.meant, heard: tile.heard)
                    + " Undo is above the words.",
                .priority: NSAccessibilityPriorityLevel.high.rawValue,
            ])
    }

    /// "3 words".
    static func wordCount(_ count: Int) -> String {
        count == 1 ? "1 word" : "\(count) words"
    }

    /// The Undo line after a Forget: "Forgot Claude. “cloud” and “clawed”
    /// will come through as heard."
    static func forgetNotice(meant: String, heard: [String]) -> String {
        let forms = heard.map { "“\($0)”" }.formatted(.list(type: .and))
        return forms.isEmpty
            ? "Forgot \(meant)."
            : "Forgot \(meant). \(forms) will come through as heard."
    }

    // MARK: - Today

    private func today(_ record: CatchRecord) -> some View {
        VStack(alignment: .leading, spacing: DictationPageStyle.rhythm) {
            SectionHeader(
                title: "Today",
                detail: record.today.isEmpty ? nil : "Click a take to fix a word in it")

            if record.today.isEmpty {
                Text(Self.nothingToday(hotkey: settings.hotkey.displayString))
                    .font(.system(size: DictationPageStyle.bodySize))
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(record.today) { take in
                        TodayTakeRow(take: take)
                    }
                }
                // The rows' hover fill reaches past the column so their text
                // lines up with the headers.
                .padding(.horizontal, -10)
            }
        }
    }

    /// The Today section with no takes yet.
    static func nothingToday(hotkey: String) -> String {
        "Nothing dictated yet today. Press \(hotkey) to dictate in any app."
    }

    // MARK: - Export

    /// Writes the Correction Pair corpus as JSONL wherever the owner points:
    /// the flywheel's export half (fine-tuning itself is out of scope, #294).
    private func exportCorrections() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "correction-pairs.jsonl"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try pairs.exportJSONL().write(to: url, options: .atomic)
        } catch {
            Log.transcription.error("Correction export failed: \(error.localizedDescription)")
        }
    }
}

/// A section's header on the catch record: its name, and a quiet detail
/// beside it. Weight and color, never size (design language §2).
private struct SectionHeader: View {
    let title: String
    let detail: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(title)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
                .accessibilityAddTraits(.isHeader)
            if let detail {
                Text(detail)
                    .foregroundStyle(.tertiary)
            }
        }
        .font(.system(size: DictationPageStyle.bodySize))
        .padding(.top, DictationPageStyle.rhythm)
    }
}
