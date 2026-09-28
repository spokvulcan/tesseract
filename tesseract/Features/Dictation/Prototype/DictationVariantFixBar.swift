//
//  DictationVariantFixBar.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  A · Fix Bar. Keyboard first. After a dictation, ⌃⌥Space opens a bar with
//  the take, the word most likely wrong already selected: type the right
//  word, Return. The words are fixed where they landed and the app learns
//  the fix. The page is two panes: your recent takes, and your words.
//

import SwiftUI

@MainActor
enum FixBarFlow {
    static func open(_ lab: DictationLab, focusWord: Int? = nil) {
        guard let take = lab.lastTake else {
            lab.showToast(LabToast(title: "Nothing to fix yet", detail: "Dictate something first"))
            return
        }
        lab.panels.closeCard()
        lab.panels.showKey(
            LabFixField(lab: lab, take: take, focusWord: focusWord) { lab.panels.closeKey() },
            size: CGSize(width: 660, height: 210)
        ) {}
    }

    /// The quiet after-card: where the words went, and the one shortcut.
    static func showHint(_ lab: DictationLab, take: LabTake) {
        lab.panels.showCard(
            FixHintCard(take: take), size: CGSize(width: 520, height: 64),
            duration: take.applied.isEmpty ? .seconds(3) : .seconds(5))
    }
}

struct FixHintCard: View {
    let take: LabTake

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark")
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(.green)
            if take.applied.isEmpty {
                Text("Inserted in \(take.appName)")
                    .font(.system(size: 12, weight: .semibold))
            } else {
                ForEach(Array(take.applied.prefix(2).enumerated()), id: \.offset) { _, item in
                    HeardMeant(heard: item.heard, meant: item.term, size: 12)
                }
            }
            Spacer(minLength: 8)
            Text("\(LabStyle.fixShortcut) to fix")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .frame(height: 40)
        .glassEffect(.regular, in: .capsule)
        .padding(8)
    }
}

// MARK: - Page

struct FixBarPage: View {
    let lab: DictationLab
    @Environment(TranscriptionHistory.self) private var history

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top, spacing: 0) {
                TakesPane(lab: lab, entries: Array(history.entries.prefix(80)))
                    .frame(minWidth: 360, maxWidth: .infinity)
                Divider()
                WordsPane(lab: lab)
                    .frame(width: 300)
            }
            Divider()
            HStack {
                LabStatusLine(extra: "\(LabStyle.fixShortcut) fix the last one")
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
        }
    }
}

struct TakesPane: View {
    let lab: DictationLab
    let entries: [TranscriptionEntry]
    @State private var editing: UUID?
    @State private var draft = ""

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                Text("Recent")
                    .font(.system(size: LabStyle.body, weight: .semibold))
                    .padding(.bottom, LabStyle.rhythm)
                if entries.isEmpty {
                    Text("Press the dictation shortcut in any app. What you say lands here.")
                        .font(.system(size: LabStyle.body))
                        .foregroundStyle(.secondary)
                }
                ForEach(entries) { entry in
                    row(entry)
                    Divider().opacity(0.5)
                }
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private func row(_ entry: TranscriptionEntry) -> some View {
        let take = lab.takes.first { $0.pairID != nil && $0.pairID == entry.pairID }
        let text = take?.text ?? entry.text
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(TranscriptionHistory.formattedTime(for: entry.timestamp))
                .font(.system(size: LabStyle.body))
                .foregroundStyle(.tertiary)
                .monospacedDigit()
                .frame(width: 70, alignment: .trailing)
            VStack(alignment: .leading, spacing: 6) {
                if editing == entry.id {
                    TextField("", text: $draft, axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.system(size: LabStyle.body))
                        .onSubmit { save(entry, take: take, original: text) }
                        .onKeyPress(.escape) {
                            editing = nil
                            return .handled
                        }
                    Text("Return saves and teaches the app · esc cancels")
                        .font(.system(size: 12))
                        .foregroundStyle(.tertiary)
                } else {
                    Text(labAttributed(text, applied: take?.applied ?? []))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .onTapGesture(count: 2) {
            draft = text
            editing = entry.id
        }
        .contextMenu {
            Button("Fix…") {
                draft = text
                editing = entry.id
            }
            Button("Copy") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
            }
        }
    }

    private func save(_ entry: TranscriptionEntry, take: LabTake?, original: String) {
        let corrected = draft.replacingOccurrences(of: "\n", with: " ")
        editing = nil
        if let take {
            Task {
                await lab.fix(
                    take.id, to: corrected, source: .page, putBack: take.id == lab.lastTake?.id)
            }
        } else {
            lab.learn(before: original, after: corrected, source: .page, appName: "an earlier take")
        }
    }
}

struct WordsPane: View {
    let lab: DictationLab
    @State private var newTerm = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: LabStyle.rhythm) {
                Text("Your words")
                    .font(.system(size: LabStyle.body, weight: .semibold))
                TextField("Add a word or name", text: $newTerm)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        lab.add(term: newTerm)
                        newTerm = ""
                    }
                if lab.lexicon.terms.isEmpty {
                    Text(
                        "Nothing learned yet. After a dictation, press \(LabStyle.fixShortcut), fix the word, and the app learns it."
                    )
                    .font(.system(size: LabStyle.body))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(lab.lexicon.terms) { term in
                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            LearnedWord(term.term)
                            Spacer()
                            if term.applied > 0 {
                                Text("fixed \(term.applied)×")
                                    .font(.system(size: 12))
                                    .foregroundStyle(.tertiary)
                            }
                        }
                        if !term.heardAs.isEmpty {
                            Text(term.heardAs.joined(separator: "  "))
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                    .contextMenu {
                        ForEach(term.heardAs, id: \.self) { heard in
                            Button("Stop changing “\(heard)”") {
                                lab.removeForm(heard, from: term.id)
                            }
                        }
                        Button("Forget \(term.term)", role: .destructive) { lab.forget(term.id) }
                    }
                }
                if !lab.memorySuggestions.isEmpty {
                    Divider()
                    Text("From your memory")
                        .font(.system(size: LabStyle.body, weight: .semibold))
                    WordFlow(spacing: 6, lineSpacing: 6) {
                        ForEach(lab.memorySuggestions, id: \.self) { word in
                            Button {
                                lab.add(term: word, source: .memory)
                            } label: {
                                Label(word, systemImage: "plus")
                                    .font(.system(size: 12))
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .contextMenu {
                                Button("Not a word I use") { lab.dismissSuggestion(word) }
                            }
                        }
                    }
                }
            }
            .padding(20)
        }
    }
}
